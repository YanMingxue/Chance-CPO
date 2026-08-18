from asyncio.log import logger
import inspect
import numpy as np
import torch
from torch.optim import Adam
import safety_gym
from scipy.stats import beta
import gym
import time
import  core
from utils.logx import EpochLogger
from torch.nn.functional import softplus
torch.autograd.set_detect_anomaly(True)

import multiprocessing as mp
ctx = mp.get_context("spawn")


# multi-process rollout worker
def _rollout_worker(wid, env_name, actor_critic, ac_kwargs,
                    seed, steps, gamma, lam, max_ep_len, state_dict_cpu):
    # torch.manual_seed(seed + 1000 * wid)
    # np.random.seed(seed + 1000 * wid)

    def sample_traj_max_len():
        # T_Q ~ Geometric on {0,1,...}, P(T_Q=t)=(1-gamma)gamma^t
        # T_max = T_Q + 1
        p = 1.0 - gamma
        if p <= 0.0:
            return max_ep_len
        tq = np.random.geometric(p) - 1
        return min(max_ep_len, int(tq) + 1)


    env = gym.make(env_name)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))

    device = torch.device("cpu")
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs).to(device)
    ac.load_state_dict(state_dict_cpu, strict=True)
    ac.eval()

    buf = PPOBuffer(obs_dim, act_dim, steps, gamma, lam)
    ep_rets, ep_costs, ep_discount_costs, ep_lens = [], [], [], []
    o, ep_ret, ep_cret, ep_discount_cret, ep_len = env.reset(), 0.0, 0.0, 0.0, 0
    traj_max_ep_len = sample_traj_max_len()
    terminal_flag = False
    N_traj = 0

    for t in range(steps):
        a, v, vc, logp = ac.step(torch.as_tensor(o, dtype=torch.float32, device=device))
        next_o, r, d, info = env.step(a)
        c = info["cost"]
        ep_ret += r
        ep_cret += c
        ep_discount_cret += (gamma**t)*c
        ep_len += 1

        # timeout = (ep_len == max_ep_len)
        timeout = (ep_len == traj_max_ep_len)
        terminal = d or timeout
        epoch_ended = (t == steps - 1)
        if terminal or epoch_ended:
            terminal_flag = True
        # save and log
        buf.store(o, a, r, c, v, vc, logp,terminal_flag)
        o = next_o

        if terminal or epoch_ended:
            N_traj += 1 
            ep_rets.append(ep_ret)
            ep_costs.append(ep_cret)
            ep_discount_costs.append(ep_discount_cret)
            ep_lens.append(ep_len)
            # if epoch_ended and not(terminal):
            #     print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
            if timeout or epoch_ended:
                _, v, vc, _ = ac.step(torch.as_tensor(o, dtype=torch.float32, device=device))
            else:
                v, vc = 0, 0
            buf.finish_path(last_val=v, last_cval=vc)
            o, ep_ret, ep_cret, ep_discount_cret, ep_len = env.reset(), 0.0, 0.0, 0.0, 0
            traj_max_ep_len = sample_traj_max_len()
            terminal_flag = False

    return {"batch": buf.get_raw(), "ep_rets": np.asarray(ep_rets, dtype=np.float32), 
            "ep_costs": np.asarray(ep_costs, dtype=np.float32), 
            "ep_lens": np.asarray(ep_lens, dtype=np.int32), "N_traj": N_traj}


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.97):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.terminal_buf = np.zeros(size, dtype=np.bool_)

        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.cadv_buf = np.zeros(size, dtype=np.float32)

        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.crew_buf = np.zeros(size, dtype=np.float32)

        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.cret_buf = np.zeros(size, dtype=np.float32)

        self.val_buf = np.zeros(size, dtype=np.float32)
        self.cval_buf = np.zeros(size, dtype=np.float32)
        
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

        #buf.store(   o, a, r, c, v,vc, logp)
    def store(self, obs, act, rew, crew, val,cval, logp, terminal):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.crew_buf[self.ptr] = crew

        self.val_buf[self.ptr] = val
        self.cval_buf[self.ptr] = cval

        self.logp_buf[self.ptr] = logp
        self.terminal_buf[self.ptr] = terminal
        self.ptr += 1

    def finish_path(self, last_val=0, last_cval=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        crews = np.append(self.crew_buf[path_slice], last_cval)

        vals = np.append(self.val_buf[path_slice], last_val)
        cvals = np.append(self.cval_buf[path_slice], last_cval)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        cdeltas = crews[:-1] + self.gamma * cvals[1:] - cvals[:-1]

        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        self.cadv_buf[path_slice] = core.discount_cumsum(cdeltas, self.gamma * self.lam)

        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        self.cret_buf[path_slice] = core.discount_cumsum(crews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    # def get(self):
    #     assert self.ptr == self.max_size    # buffer has to be full before you can get
    #     self.ptr, self.path_start_idx = 0, 0
    #     # the next two lines implement the advantage normalization trick
    #     adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
    #     cadv_mean, cadv_std = mpi_statistics_scalar(self.cadv_buf)

    #     self.adv_buf = (self.adv_buf - adv_mean) / adv_std
    #     self.cadv_buf = (self.cadv_buf - cadv_mean) #/ adv_std

    #     data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf, cret=self.cret_buf,
    #                 adv=self.adv_buf, cadv=self.cadv_buf, logp=self.logp_buf)
    #     return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}
    
    def get_raw(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        data = dict(obs=self.obs_buf.copy(), act=self.act_buf.copy(), ret=self.ret_buf.copy(), cret=self.cret_buf.copy(),
                    adv=self.adv_buf.copy(), cadv=self.cadv_buf.copy(), logp=self.logp_buf.copy(), 
                    rew=self.rew_buf.copy(), crew=self.crew_buf.copy(), val=self.val_buf.copy(), cval=self.cval_buf.copy(), terminal=self.terminal_buf.copy())
        # return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}
        return data
    
    
    def load_raw(self, data):
        """Load raw numpy dict into this buffer (assumes matching shapes)."""
        self.obs_buf = data["obs"]
        self.act_buf = data["act"]
        self.ret_buf = data["ret"]
        self.cret_buf = data["cret"]
        self.adv_buf = data["adv"]
        self.cadv_buf = data["cadv"]
        self.logp_buf = data["logp"]
        self.rew_buf = data["rew"]
        self.crew_buf = data["crew"]
        self.val_buf = data["val"]
        self.cval_buf = data["cval"]
        self.terminal_buf = data["terminal"]
        self.ptr = self.max_size
        self.path_start_idx = 0


    def get_torch(self):
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in self.get_raw().items()}
    


class CVaR_Train_Agent:
    def __init__(self, env_name, num_procs=5, seed=0,steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
                vf_lr=1e-3, penalty_lr=4.5e-4,train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000, 
                target_kl=0.01, logger_kwargs=dict(), save_freq=1, alpha=0.2, cost_limit=30,lambda_init=1.0, device='cpu'):
        self.device = torch.device(device if device else "cpu")
        # Instantiate environment
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.seed = seed
        self.estimation_risk_level = 0.2
        self.num_procs = num_procs
        self.steps_per_epoch = steps_per_epoch  
        self.epochs = epochs
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.pi_lr = pi_lr
        self.vf_lr = vf_lr
        self.penalty_lr = penalty_lr * train_pi_iters * 0.5
        self.nu_lr = penalty_lr * train_pi_iters
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.lam = lam
        self.cost_limit = float(cost_limit)
        self.nu = torch.tensor(self.cost_limit, requires_grad=True, device=self.device).float()
        self.max_ep_len = max_ep_len
        self.target_kl = target_kl
        self.save_freq = save_freq
        self.alpha = alpha
        self.lambda_init = lambda_init
        self.penalty_param = torch.tensor(self.lambda_init, requires_grad=True, device=self.device).float()
        # penalty = softplus(penalty_param)
        
        
        # Set up logger and save configuration
        self.logger_kwargs = logger_kwargs
        self.logger = EpochLogger(**self.logger_kwargs)
        # self.logger.save_config(locals())
        frame = inspect.currentframe() 
        args, _, _, values = inspect.getargvalues(frame) 
        config = {k: values[k] for k in args if k != "self"} 
        self.logger.save_config(config)


    def compute_truncate_error_bound(self, avg_length):
        return self.gamma**avg_length * self.cost_max / (1 - self.gamma)
    
    def clopper_pearson_upper(self, k, N):
        """
        Clopper-Pearson upper confidence bound
        p_U = BetaInv(1-delta; k+1, N-k)
        """
        if k == N:
            return 1.0
        return beta.ppf(1 - self.estimation_risk_level, k + 1, N - k)

    
    def slipt_data(self, data, N_traj):
        # Create masks for trajectories with nuk
        terminals = data.get('terminal', None)
        if terminals is None:
            # Fallback: no terminal info, treat as single trajectory.
            return data
        terminals = terminals.to(dtype=torch.bool)
        num_steps = terminals.shape[0]

        traj_starts = [0]
        traj_ends = []
        for i in range(num_steps):
            if terminals[i]:
                traj_ends.append(i)
                if i + 1 < num_steps:
                    traj_starts.append(i + 1)

        # Include the last trajectory even if it doesn't end with a terminal state
        if len(traj_ends) == 0 or traj_ends[-1] != num_steps - 1:
            traj_ends.append(num_steps - 1)

        # Use trajectory cumulative costs for CVaR and probability estimation.
        traj_costs = []
        for i, start_idx in enumerate(traj_starts):
            end_idx = traj_ends[i] + 1
            traj_costs.append(data['crew'][start_idx:end_idx].sum().item())
        order = np.argsort(traj_costs)[::-1]
        # Find the closest cost below cost_limit in descending order
        sorted_costs = np.asarray(traj_costs)[order]
        
        below_mask = sorted_costs < self.cost_limit
        exceed_pos = int(np.sum(~below_mask))  # How many trajectories are above the cost limit
        traj_count = max(len(sorted_costs), 1)
        prob_estimate = self.clopper_pearson_upper(exceed_pos, traj_count)

        # Trajectory-level condition: J_c(tau) >= nu.
        nuk = max(0.0, float(self.nu.detach().item()))
        exceed_traj_mask = np.asarray(traj_costs) >= nuk

        # Convert trajectory mask -> step mask, then index all step-level tensors.
        exceed_step_mask = torch.zeros(num_steps, dtype=torch.bool, device=terminals.device)
        for i, start_idx in enumerate(traj_starts):
            end_idx = traj_ends[i] + 1
            if exceed_traj_mask[i]:
                exceed_step_mask[start_idx:end_idx] = True

        exceed_data = {k: v[exceed_step_mask] for k, v in data.items()}
        traj_costs_t = torch.as_tensor(traj_costs, dtype=torch.float32, device=terminals.device)

        return prob_estimate, exceed_data, exceed_pos, traj_costs_t


    # Set up function for computing PPO policy loss
    def compute_loss_pi(self, data, exceed_data, exceed_pos, N_traj, traj_costs):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        exceed_obs, exceed_act, exceed_cadv, exceed_logp_old = exceed_data['obs'], exceed_data['act'], exceed_data['cadv'], exceed_data['logp']
        cur_cost = data['cur_cost']
        penalty_param = data['cur_penalty']
        # Policy loss
        pi, logp = self.ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)

        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
        loss_rpi = (torch.min(ratio * adv, clip_adv)).mean()

        # clip_cadv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * cadv
        # loss_cpi = (torch.min(ratio * cadv, clip_cadv)).mean()
        
        # loss_cpi = ratio*cadv
        # loss_cpi = loss_cpi.mean()

        _, exceed_logp = self.ac.pi(exceed_obs, exceed_act) 
        ratio_cadv = torch.exp(exceed_logp - exceed_logp_old)
        if exceed_pos == 0:
            loss_cpi = torch.zeros((), device=obs.device)
        else:
            tail_term = (ratio_cadv * exceed_cadv).mean()  # multiply the indicator
            loss_cpi = (exceed_pos / N_traj) * tail_term / (1 - self.alpha)

        # loss_cpi = 1/(1-self.alpha) * (exceed_pos / N_traj) * (ratio_cadv*exceed_cadv).mean()   # multiply the indicator
        
        
        p = softplus(penalty_param)
        penalty_item = p.item()
    
        pi_objective = loss_rpi - penalty_item*loss_cpi
        pi_objective = pi_objective/(1+penalty_item)
        loss_pi = -pi_objective

        nu_proj = torch.clamp(self.nu, min=0.0)
        if traj_costs.numel() == 0:
            cvar_est = nu_proj
        else:
            tail_excess = torch.relu(traj_costs - nu_proj)
            cvar_est = nu_proj + tail_excess.mean() / max(1 - self.alpha, 1e-8)
        cost_deviation = cvar_est - self.cost_limit

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, cost_deviation, pi_info

    # Set up function for computing value loss
    def compute_loss_v(self, data, exceed_data):
        obs, ret, cret = data['obs'], data['ret'], data['cret']
        exceed_obs, exceed_cret = exceed_data['obs'], exceed_data['cret']
        loss_v = ((self.ac.v(obs) - ret)**2).mean()
        # loss_vc = ((self.ac.vc(obs) - cret)**2).mean()
        if exceed_obs.shape[0] == 0:
            loss_vc = torch.zeros((), device=obs.device)
        else:
            loss_vc = ((self.ac.vc(exceed_obs) - exceed_cret)**2).mean()
        return loss_v, loss_vc
    
    def update(self, N_traj):
        cur_cost = self.logger.get_stats('EpCost')[0]
        data = self.buf.get_torch()

        # Upate penalty parameter
        for k, v in data.items():
            data[k] = v.to(self.device)
        probEST, exceed_data, exceed_pos, traj_costs = self.slipt_data(data, N_traj)
        data['cur_cost'] = cur_cost
        data['cur_penalty'] = self.penalty_param
        pi_l_old, cost_dev, pi_info_old = self.compute_loss_pi(data, exceed_data, exceed_pos, N_traj, traj_costs)
        #print(penalty_param)
        loss_penalty = -self.penalty_param*cost_dev
        self.penalty_optimizer.zero_grad()
        loss_penalty.backward()
        self.penalty_optimizer.step()
        data['cur_penalty'] = self.penalty_param
        
        # Update nu parameter
        penalty_item = softplus(self.penalty_param.detach())
        nu_proj = torch.clamp(self.nu, min=0.0)
        if traj_costs.numel() == 0:
            cvar_term = nu_proj - self.cost_limit
        else:
            cvar_term = nu_proj + torch.relu(traj_costs - nu_proj).mean() / (1 - self.alpha) - self.cost_limit
        # cvar_term = (1 - 1/(1-self.alpha) * (exceed_pos / N_traj))   # No gradient
        loss_nu = penalty_item * cvar_term
        self.nu_optimizer.zero_grad()
        loss_nu.backward()
        self.nu_optimizer.step()
        with torch.no_grad():
            self.nu.clamp_(min=0.0)

        # Train agent - actor
        pi_l_old = pi_l_old.item()
        v_l_old, cv_l_old = self.compute_loss_v(data, exceed_data)
        v_l_old, cv_l_old = v_l_old.item(), cv_l_old.item() 

        # Train policy with multiple steps of gradient descent
        train_pi_iters=80
        for i in range(train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, _,pi_info = self.compute_loss_pi(data, exceed_data, exceed_pos, N_traj, traj_costs)
            kl = pi_info['kl']
            if kl > 1.2 * self.target_kl:
                self.logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break

            loss_pi.backward()
            self.pi_optimizer.step()

        self.logger.store(StopIter=i)

        # Value function learning
        train_v_iters=80
        for i in range(train_v_iters):
            loss_v, loss_vc = self.compute_loss_v(data, exceed_data)
            self.vf_optimizer.zero_grad()
            loss_v.backward()
            self.vf_optimizer.step()        
            if loss_vc.requires_grad:
                self.cvf_optimizer.zero_grad()
                loss_vc.backward()
                self.cvf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        self.logger.store(LossPi=pi_l_old, LossV=v_l_old,
                    KL=kl, Entropy=ent, ClipFrac=cf,
                    DeltaLossPi=(loss_pi.item() - pi_l_old),
                    DeltaLossV=(loss_v.item() - v_l_old))
        
        return probEST
        

    def _flatten_obs(self, o):
        return np.asarray(o, dtype=np.float32).reshape(-1)


    def cvar(self, actor_critic=core.MLPActorCritic, ac_kwargs=dict()):

        # Set up logger and save configuration
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Instantiate environment

        obs_dim = int(np.prod(self.env.observation_space.shape))
        act_dim = int(np.prod(self.env.action_space.shape))

        # Create actor-critic module
        self.ac = actor_critic(self.env.observation_space, self.env.action_space, **ac_kwargs).to(self.device)   

        # Count variables
        var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.v])
        self.logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

        # Set up optimizers for policy and value function
        # pi_lr = 3e-4
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.pi_lr)
        
        # penalty_lr = 5e-2
        self.penalty_optimizer = Adam([self.penalty_param], lr=self.penalty_lr)
        self.nu_optimizer = Adam([self.nu], lr=self.nu_lr)
        # vf_lr = 1e-3
        self.vf_optimizer = Adam(self.ac.v.parameters(), lr=self.vf_lr)
        self.cvf_optimizer = Adam(self.ac.vc.parameters(),lr=self.vf_lr)
        # Set up model saving
        self.logger.setup_pytorch_saver(self.ac)

        # Set up experience buffer
        local_steps = int(self.steps_per_epoch / self.num_procs)
        self.buf = PPOBuffer(obs_dim, act_dim, self.steps_per_epoch, self.gamma, self.lam)

        # Prepare for interaction with environment
        start_time = time.time()
        # o, ep_ret,ep_cret, ep_len = self.env.reset(), 0, 0, 0

        # Main loop: collect experience in env and update/log each epoch
        for epoch in range(self.epochs):
            state_dict_cpu = {k: v.detach().cpu() for k, v in self.ac.state_dict().items()}
            with ctx.Pool(processes=self.num_procs) as pool:
                results = pool.starmap(
                    _rollout_worker,
                    [(wid, self.env_name, actor_critic, ac_kwargs,
                    self.seed, local_steps, self.gamma, self.lam, self.max_ep_len, state_dict_cpu)
                    for wid in range(self.num_procs)]
                )
            merged = {}
            for k in results[0]["batch"].keys():
                merged[k] = np.concatenate([b["batch"][k] for b in results], axis=0)

            adv = merged["adv"]
            merged["adv"] = (adv - adv.mean()) / (adv.std() + 1e-8)
            cadv = merged["cadv"]
            merged["cadv"] = (cadv - cadv.mean()) / (cadv.std() + 1e-8)

            self.buf.load_raw(merged)

            # --- merge episode stats ---
            all_ep_ret  = np.concatenate([r["ep_rets"]  for r in results], axis=0) if any(len(r["ep_rets"]) for r in results) else np.array([], dtype=np.float32)
            all_ep_cost = np.concatenate([r["ep_costs"] for r in results], axis=0) if any(len(r["ep_costs"]) for r in results) else np.array([], dtype=np.float32)
            all_ep_len  = np.concatenate([r["ep_lens"]  for r in results], axis=0) if any(len(r["ep_lens"]) for r in results) else np.array([], dtype=np.float32)

            for r in all_ep_ret:
                self.logger.store(EpRet=float(r))
            for c in all_ep_cost:
                self.logger.store(EpCost=float(c))
            for l in all_ep_len:
                self.logger.store(EpLen=float(l))

            N_traj = sum([r["N_traj"] for r in results])
            
            # Save model
            if (epoch % self.save_freq == 0) or (epoch == self.epochs-1):
                self.logger.save_state({'env': self.env}, None)

            # Perform PPO update!
            probEST = self.update(N_traj)

            # Log info about epoch
            self.logger.log_tabular('Epoch', epoch)
            self.logger.log_tabular('EpRet', with_min_and_max=True)
            self.logger.log_tabular('EpCost', with_min_and_max=True)
            self.logger.log_tabular('EpLen', average_only=True)
            self.logger.log_tabular('Lambda', self.penalty_param.item())
            self.logger.log_tabular('Nu', self.nu.item())
            self.logger.log_tabular('ProbEst', probEST)
            # self.logger.log_tabular('VVals', with_min_and_max=True)
            self.logger.log_tabular('TotalEnvInteracts', (epoch+1)*self.steps_per_epoch)
            # self.logger.log_tabular('LossPi', average_only=True)
            # self.logger.log_tabular('LossV', average_only=True)
            # self.logger.log_tabular('DeltaLossPi', average_only=True)
            # self.logger.log_tabular('DeltaLossV', average_only=True)
            # self.logger.log_tabular('Entropy', average_only=True)
            # self.logger.log_tabular('KL', average_only=True)
            # self.logger.log_tabular('ClipFrac', average_only=True)
            # self.logger.log_tabular('StopIter', average_only=True)
            self.logger.log_tabular('Time', time.time()-start_time)
            self.logger.dump_tabular()
