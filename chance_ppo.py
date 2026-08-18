import numpy as np
import copy
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import safety_gym
from gym.wrappers.time_limit import TimeLimit
from scipy.stats import beta
import gym
import time
import inspect
import os.path as osp
import  core
from utils.logx import EpochLogger
from utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from torch.nn.functional import softplus
torch.autograd.set_detect_anomaly(True)

import multiprocessing as mp
ctx = mp.get_context("spawn")

# multi-process rollout worker
def _rollout_worker(wid, env_name, actor_critic, ac_kwargs,
                    seed, steps, gamma, lam, cost_limit, max_ep_len, state_dict_cpu, geometric_sample=False):
    # torch.manual_seed(seed + 1000 * wid)
    # np.random.seed(seed + 1000 * wid)

    def sample_traj_max_len():
        # T_Q ~ Geometric on {0,1,...}, P(T_Q=t)=(1-gamma)gamma^t
        # T_max = T_Q + 1
        p = 1.0 - gamma
        if p <= 0.0:
            return max_ep_len
        tq = np.random.geometric(p) - 1
        return int(tq) + 1
        # return min(max_ep_len, int(tq) + 1)

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
    # Auxiliary variable
    y = cost_limit
    w = 1.0
    traj_max_ep_len = sample_traj_max_len()
    terminal_flag, tau = False, None
    N_traj = 0

    for t in range(steps):
        aug_o = np.concatenate([o, np.array([y]), np.array([w])])
        a, v, vc, vg, logp = ac.step(torch.as_tensor(o, dtype=torch.float32, device=device), 
                                 torch.as_tensor(aug_o, dtype=torch.float32, device=device))
        next_o, r, d, info = env.step(a)
        c = info["cost"]

        next_y = (y - c)/gamma

        g = w if (y >0 and next_y <= 0) else 0.0
        next_w = w/gamma 

        ep_ret += r
        ep_cret += c
        ep_discount_cret += (gamma**t)*c
        ep_len += 1


        if geometric_sample:
            timeout = (ep_len == traj_max_ep_len)
        else:
            timeout = (ep_len == max_ep_len)
        terminal = d or timeout

        if terminal: 
            terminal_flag = True
        # save and log
        buf.store(o, y, aug_o, a, r, c, g, v, vc, vg, logp, terminal_flag)
        o = next_o
        y = next_y
        w = next_w

        if terminal: #or epoch_ended:
            N_traj += 1 
            ep_rets.append(ep_ret)
            ep_costs.append(ep_cret)
            ep_discount_costs.append(ep_discount_cret)
            ep_lens.append(ep_len)
            # if epoch_ended and not(terminal):
            #     print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
            if timeout:  #or epoch_ended:
                _, v, vc, vg, _ = ac.step(torch.as_tensor(o, dtype=torch.float32, device=device),
                                      torch.as_tensor(aug_o, dtype=torch.float32, device=device))
            else:
                v, vc, vg = 0, 0, 0
            buf.finish_path(last_val=v, last_cval=vc, last_gval=vg)
            o, ep_ret, ep_cret, ep_discount_cret, ep_len = env.reset(), 0.0, 0.0, 0.0, 0
            traj_max_ep_len = sample_traj_max_len()
            terminal_flag, tau = False, None

    return {"batch": buf.get_raw(), "ep_rets": np.asarray(ep_rets, dtype=np.float32), 
            "ep_costs": np.asarray(ep_costs, dtype=np.float32), 
            "ep_discount_costs": np.asarray(ep_discount_costs, dtype=np.float32), 
            "ep_lens": np.asarray(ep_lens, dtype=np.int32), "N_traj": N_traj}

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.97):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.aux_buf = np.zeros(core.combined_shape(size, 1), dtype=np.float32)
        self.aug_obs_buf = np.zeros(core.combined_shape(size, obs_dim+2), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.terminal_buf = np.zeros(size, dtype=np.bool_)
        
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.cadv_buf = np.zeros(size, dtype=np.float32)
        self.gadv_buf = np.zeros(size, dtype=np.float32)

        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.crew_buf = np.zeros(size, dtype=np.float32)
        self.grew_buf = np.zeros(size, dtype=np.float32)

        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.cret_buf = np.zeros(size, dtype=np.float32)
        self.gret_buf = np.zeros(size, dtype=np.float32)

        self.val_buf = np.zeros(size, dtype=np.float32)
        self.cval_buf = np.zeros(size, dtype=np.float32)
        self.gval_buf = np.zeros(size, dtype=np.float32)
        
        self.logp_buf = np.zeros(size, dtype=np.float32)

        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

        #buf.store(   o, a, r, c, v,vc, logp)
    def store(self, obs, aux, aug_obs, act, rew, crew, grew, val, cval, gval, logp, terminal):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.aux_buf[self.ptr] = aux
        self.aug_obs_buf[self.ptr] = aug_obs
        self.act_buf[self.ptr] = act

        self.rew_buf[self.ptr] = rew
        self.crew_buf[self.ptr] = crew
        self.grew_buf[self.ptr] = grew

        self.val_buf[self.ptr] = val
        self.cval_buf[self.ptr] = cval
        self.gval_buf[self.ptr] = gval

        self.logp_buf[self.ptr] = logp

        self.terminal_buf[self.ptr] = terminal
        self.ptr += 1

    def finish_path(self, last_val=0, last_cval=0, last_gval=0):
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
        grews = np.append(self.grew_buf[path_slice], last_gval)

        vals = np.append(self.val_buf[path_slice], last_val)
        cvals = np.append(self.cval_buf[path_slice], last_cval)   # correspongding to reward c
        gvals = np.append(self.gval_buf[path_slice], last_gval)   # correspongding to reward g

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        cdeltas = crews[:-1] + self.gamma * cvals[1:] - cvals[:-1]
        gdeltas = grews[:-1] + self.gamma * gvals[1:] - gvals[:-1]

        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        self.cadv_buf[path_slice] = core.discount_cumsum(cdeltas, self.gamma * self.lam)
        self.gadv_buf[path_slice] = core.discount_cumsum(gdeltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        self.cret_buf[path_slice] = core.discount_cumsum(crews, self.gamma)[:-1]
        self.gret_buf[path_slice] = core.discount_cumsum(grews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr


    def get_raw(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        data = dict(obs=self.obs_buf.copy(), aux=self.aux_buf.copy(), aug_obs=self.aug_obs_buf.copy(), act=self.act_buf.copy(), 
                    ret=self.ret_buf.copy(), cret=self.cret_buf.copy(), gret=self.gret_buf.copy(),
                    adv=self.adv_buf.copy(), cadv=self.cadv_buf.copy(), gadv=self.gadv_buf.copy(), logp=self.logp_buf.copy(), 
                    rew=self.rew_buf.copy(), crew=self.crew_buf.copy(), grew=self.grew_buf.copy(), 
                    val=self.val_buf.copy(), cval=self.cval_buf.copy(), gval=self.gval_buf.copy(), terminal=self.terminal_buf.copy())
        # return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}
        return data
    
    
    def load_raw(self, data):
        """Load raw numpy dict into this buffer (assumes matching shapes)."""
        self.obs_buf = data["obs"]
        self.aux_buf = data["aux"]
        self.aug_obs_buf = data["aug_obs"]
        self.act_buf = data["act"]
        self.ret_buf = data["ret"]
        self.cret_buf = data["cret"]
        self.gret_buf = data["gret"]
        self.adv_buf = data["adv"]
        self.cadv_buf = data["cadv"]
        self.gadv_buf = data["gadv"]
        self.logp_buf = data["logp"]
        self.rew_buf = data["rew"]
        self.crew_buf = data["crew"]
        self.grew_buf = data["grew"]
        self.val_buf = data["val"]
        self.cval_buf = data["cval"]
        self.gval_buf = data["gval"]
        self.terminal_buf = data["terminal"]
        self.ptr = self.max_size
        self.path_start_idx = 0

    
    def get_torch(self):
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in self.get_raw().items()}
    


class ChancePPO_Train_Agent:
    def __init__(self, env_name, num_procs=5, seed=0,steps_per_epoch=4000, epochs=50, multistep_penalty=False, discout_cost=False, kl_bound=False, geometric_sample=True,
                gamma=0.99, beta=0.1,clip_ratio=0.2, pi_lr=3e-4, vf_lr=1e-3, penalty_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000, 
                target_kl=0.01, logger_kwargs=dict(), save_freq=1, alpha = 0.2, cost_limit=30, lambda_init=1.0, device='cpu'):
        self.device = torch.device(device if torch.device else "cpu")
        # Instantiate environment
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.cost_max = 1
        self.estimation_risk_level = 0.2 #beta in the paper  1-beta = 0.8
        self.seed = seed
        self.num_procs = num_procs
        self.steps_per_epoch = steps_per_epoch  
        self.epochs = epochs
        self.multistep_penalty = multistep_penalty
        self.discout_cost = discout_cost
        self.kl_bound = kl_bound
        self.geometric_sample = geometric_sample
        self.gamma = gamma
        self.beta = beta
        self.clip_ratio = clip_ratio
        self.pi_lr = pi_lr # 3e-4
        self.vf_lr = vf_lr
        self.penalty_lr = penalty_lr  # 5e-4 if self.multistep_penalty else 1e-3
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.lam = lam
        self.max_ep_len = max_ep_len
        self.target_kl = target_kl
        self.save_freq = save_freq
        self.alpha = alpha
        self.cost_limit = cost_limit 
        self.lambda_init = lambda_init
        self.penalty_param = torch.tensor(self.lambda_init, requires_grad=True, device=self.device).float()
        # penalty = softplus(penalty_param)
        self.device = device
        
        # Set up logger and save configuration
        self.logger_kwargs = logger_kwargs
        self.logger = EpochLogger(**self.logger_kwargs)
        # self.logger.save_config(locals())
        frame = inspect.currentframe() 
        args, _, _, values = inspect.getargvalues(frame) 
        config = {k: values[k] for k in args if k != "self"} 
        self.logger.save_config(config)


    # Set up function for computing PPO policy loss
    def compute_loss_pi(self, data):
        obs, act, adv, cadv, gadv, logp_old = data['obs'], data['act'], data['adv'], data['cadv'], data['gadv'], data['logp']
        penalty_param = data['cur_penalty']
        # Policy loss
        pi, logp = self.ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)

        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
        loss_rpi = (torch.min(ratio * adv, clip_adv)).mean()
        # print(ratio.shape, adv.shape, cadv_s.shape)

        # PPO_lg version
        loss_cpi = ratio*cadv
        loss_cpi = loss_cpi.mean()

        loss_gpi = ratio*gadv
        loss_gpi = loss_gpi.mean()
        
        p = softplus(penalty_param)
        penalty_item = p.item()
    
        pi_objective = loss_rpi - penalty_item*loss_cpi   # average of reward - penalty * worst-cost 
        pi_objective = pi_objective/(1+penalty_item)
        loss_pi = - pi_objective   # Maximize the objective function

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        cpi_correct = ((ratio * cadv).mean() - cadv.mean()).item()
        gpi_correct = ((ratio * gadv).mean() - gadv.mean()).item()

        return loss_pi, cpi_correct, gpi_correct, pi_info

    # Set up function for computing value loss
    def compute_loss_v_vc(self, data):
        obs, aug_obs, ret, cret, gret = data['obs'], data['aug_obs'], data['ret'], data['cret'], data['gret']
        return ((self.ac.v(obs) - ret)**2).mean(), ((self.ac.vc(obs) - cret)**2).mean(), ((self.ac.vg(aug_obs) - gret)**2).mean()
    
    
    def slipt_data(self, data, N_traj):
        cost_limit = self.cost_limit
        terminals = data.get('terminal', None)
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


        # Use discounted cumulative costs to dertermin the probability estimation
        if self.discout_cost:
            traj_costs = []
            for i, start_idx in enumerate(traj_starts):
                end_idx = traj_ends[i] + 1
                traj_crew = data['crew'][start_idx:end_idx]
                t = torch.arange(traj_crew.shape[0], device=traj_crew.device, dtype=traj_crew.dtype)
                traj_costs.append((traj_crew * (self.gamma ** t)).sum().item())
        # Use cumulative costs to dertermin the probability estimation
        else:
            traj_costs = []
            for i, start_idx in enumerate(traj_starts):
                end_idx = traj_ends[i] + 1
                traj_costs.append(data['crew'][start_idx:end_idx].sum().item())


        order = np.argsort(traj_costs)[::-1]
        # Find the closest cost below cost_limit in descending order
        sorted_costs = np.asarray(traj_costs)[order]
        if self.discout_cost:
            avg_length = num_steps / N_traj
            truncate_error = self.compute_truncate_error_bound(avg_length)
            below_mask = sorted_costs < (cost_limit - truncate_error)
        else:
            below_mask = sorted_costs < cost_limit
        exceed_pos = np.sum(~below_mask) # How many trajectories are above the cost limit

        # Guarantee Pr[P(Jc​(π)≥d)≤ prob_estimate] ≥ 1-risk_level
        prob_estimate = self.clopper_pearson_upper(exceed_pos, N_traj)

        return prob_estimate, exceed_pos/N_traj
    
    ################################################################################
    """ TODO: Estimate the gardient of advantages b = E[∇logπk(a|s)Ac] """
    def flat_grad(self, grads):
        return torch.cat([g.reshape(-1) for g in grads if g is not None])

    def compute_gradient_A(self, policy_old, obs, act, adv_c):
        # pi = policy_old._distribution(obs)
        # logp = policy_old._log_prob_from_distribution(pi, act)
        pi, logp = policy_old(obs, act)
        if logp.dim() > 1:
            logp = logp.sum(axis=-1)  # Should already be summed
        loss_c = (logp * adv_c.detach()).mean()
        grads = torch.autograd.grad(loss_c, policy_old.parameters(), create_graph=False, retain_graph=False)
        b = self.flat_grad(grads).detach()
        return b
    ################################################################################

    ################################################################################
    """ Compute the upper bound of Bernoulli KL divergence for probability estimation """
    def bernoulli_kl(self, q, p, eps=1e-12):
        q = np.clip(q, eps, 1.0 - eps)
        p = np.clip(p, eps, 1.0 - eps)
        return q * np.log(q / p) + (1.0 - q) * np.log((1.0 - q) / (1.0 - p))

    def bernoulli_kl_upper(self, p, K, iters=40):
        p = float(np.clip(p, 0.0, 1.0))
        K = float(max(0.0, K))
        if p >= 1.0 or K <= 0.0:
            return p
        if p <= 0.0:
            # kl(q||0)=+inf for any q>0, so only q=0 feasible
            return 0.0
        if self.bernoulli_kl(1.0 - 1e-12, p) <= K:
                return 1.0
        lo, hi = p, 1.0 - 1e-12
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if self.bernoulli_kl(mid, p) <= K:
                lo = mid
            else:
                hi = mid
        return min(1.0, lo)
    ################################################################################

    ################################################################################
    def compute_delta(self, expectation, prob_estimate):
        return max(expectation/prob_estimate, self.cost_limit)
    
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
    
    def compute_dev(self, cur_cost, delta=0.0, policy_old=None, policy=None, data=None):
        cost_deviation = (cur_cost - delta*self.alpha)
        if self.multistep_penalty:
            if policy_old is None or policy is None or data is None:
                raise ValueError("policy_old, policy, and dataset must be provided when multistep_penalty is True.")
            obs, act, adv_c = data['obs'], data['act'], data['cadv']
            gradient = self.compute_gradient_A(policy_old, obs, act, adv_c)
            theta = torch.cat([p.view(-1) for p in policy.parameters()])
            theta_old = torch.cat([p.view(-1) for p in policy_old.parameters()])
            cost_deviation += torch.dot(gradient, theta - theta_old)
        return cost_deviation
    ################################################################################


    def update(self, N_traj, epoch=None):

        # cur_cost = self.logger.get_stats('EpCost')[0] #[mean, std]
        # Let gamma=1 but sample the length of trajectory from geometric distribution
        cur_cost = self.logger.get_stats('EpDiscountCost')[0] if self.discout_cost else self.logger.get_stats('EpCost')[0]


        data = self.buf.get_torch()
        for k, v in data.items():
            data[k] = v.to(self.device)
        data['cur_cost'] = cur_cost
        data['cur_penalty'] = self.penalty_param

        prob_estimate, prob_emprical = self.slipt_data(data, N_traj)
        # Prob EMA
        if epoch == 0:
            self.p_bar = prob_estimate
        else:
            self.p_bar = (1 - self.beta) * self.p_bar + self.beta * prob_estimate


        data['cur_penalty'] = self.penalty_param
        pi_l_old, _, _, pi_info_old = self.compute_loss_pi(data=data)
        pi_l_old = pi_l_old.item()
        v_l_old, cv_l_old, vg_l_old = self.compute_loss_v_vc(data)
        v_l_old, cv_l_old, vg_l_old = v_l_old.item(), cv_l_old.item(), vg_l_old.item()

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, cpi_correct, gpi_correct, pi_info = self.compute_loss_pi(data=data)
            kl = mpi_avg(pi_info['kl']) #pi_info['kl'] # mpi_avg(pi_info['kl'])
            if kl > 1.2 * self.target_kl:
                self.logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
            loss_pi.backward()
            self.pi_optimizer.step()
            self.pi_scheduler.step()
            """TODO: compute gradient of Ac and update penalty parameter"""
            prob_estimate_i = self.p_bar 
            delta = self.compute_delta(cur_cost + cpi_correct, self.p_bar + gpi_correct)  # prob_estimate)
            # print(f"Epoch: {i}, delta: {delta:.4f}, cur_cost: {cur_cost:.4f}, p_bar: {self.p_bar:.4f}, A_c: {cpi_correct:.4f}, A_g: {gpi_correct:.4f}")
            cost_dev = self.compute_dev(data['cur_cost'], delta=delta, policy_old=self.ac_old.pi, policy=self.ac.pi, data=data)
            loss_penalty = -self.penalty_param*cost_dev
            loss_penalty = -self.penalty_param*cost_dev
            self.penalty_optimizer.zero_grad()
            loss_penalty.backward()
            mpi_avg_grads(self.penalty_param)
            self.penalty_optimizer.step()
            self.penalty_scheduler.step()

        self.logger.store(StopIter=i)

        # Value function learning
        for i in range(self.train_v_iters):
            loss_v, loss_vc, loss_vg = self.compute_loss_v_vc(data)
            
            self.vf_optimizer.zero_grad()
            loss_v.backward()
            self.vf_optimizer.step()
            self.vf_scheduler.step()

            self.cvf_optimizer.zero_grad()
            loss_vc.backward()
            self.cvf_optimizer.step()
            self.cvf_scheduler.step()

            self.gvf_optimizer.zero_grad()
            loss_vg.backward()
            self.gvf_optimizer.step()
            self.gvf_scheduler.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        self.logger.store(LossPi=pi_l_old, LossV=v_l_old,
                    KL=kl, Entropy=ent, ClipFrac=cf,
                    DeltaLossPi=(loss_pi.item() - pi_l_old),
                    DeltaLossV=(loss_v.item() - v_l_old),
                    prob_est = prob_estimate,
                    prob_emprical = prob_emprical)
        return prob_estimate, prob_emprical

    def _checkpoint_path(self):
        return osp.join(self.logger.output_dir, "checkpoint.pt")

    def _legacy_model_path(self):
        return osp.join(self.logger.output_dir, "pyt_save", "model.pt")

    def _progress_start_epoch(self):
        progress_path = osp.join(self.logger.output_dir, "progress.txt")
        if not osp.exists(progress_path):
            return 0
        with open(progress_path, "r") as f:
            lines = [line.strip().split("\t") for line in f if line.strip()]
        if len(lines) <= 1 or "Epoch" not in lines[0]:
            return 0
        epoch_idx = lines[0].index("Epoch")
        epochs = []
        for row in lines[1:]:
            if len(row) > epoch_idx and row[epoch_idx] != "":
                epochs.append(int(float(row[epoch_idx])))
        return max(epochs) + 1 if epochs else 0

    def _save_checkpoint(self, epoch):
        checkpoint = {
            "epoch": epoch,
            "ac_state_dict": self.ac.state_dict(),
            "penalty_param": self.penalty_param.detach().cpu(),
            "p_bar": self.p_bar,
            "pi_optimizer": self.pi_optimizer.state_dict(),
            "penalty_optimizer": self.penalty_optimizer.state_dict(),
            "vf_optimizer": self.vf_optimizer.state_dict(),
            "cvf_optimizer": self.cvf_optimizer.state_dict(),
            "gvf_optimizer": self.gvf_optimizer.state_dict(),
            "pi_scheduler": self.pi_scheduler.state_dict(),
            "penalty_scheduler": self.penalty_scheduler.state_dict(),
            "vf_scheduler": self.vf_scheduler.state_dict(),
            "cvf_scheduler": self.cvf_scheduler.state_dict(),
            "gvf_scheduler": self.gvf_scheduler.state_dict(),
        }
        torch.save(checkpoint, self._checkpoint_path())

    def _load_resume_state(self):
        checkpoint_path = self._checkpoint_path()
        if osp.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            except TypeError:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.ac.load_state_dict(checkpoint["ac_state_dict"])
            self.penalty_param.data.copy_(checkpoint["penalty_param"].to(self.device))
            self.p_bar = checkpoint["p_bar"]
            self.pi_optimizer.load_state_dict(checkpoint["pi_optimizer"])
            self.penalty_optimizer.load_state_dict(checkpoint["penalty_optimizer"])
            self.vf_optimizer.load_state_dict(checkpoint["vf_optimizer"])
            self.cvf_optimizer.load_state_dict(checkpoint["cvf_optimizer"])
            self.gvf_optimizer.load_state_dict(checkpoint["gvf_optimizer"])
            self.pi_scheduler.load_state_dict(checkpoint["pi_scheduler"])
            self.penalty_scheduler.load_state_dict(checkpoint["penalty_scheduler"])
            self.vf_scheduler.load_state_dict(checkpoint["vf_scheduler"])
            self.cvf_scheduler.load_state_dict(checkpoint["cvf_scheduler"])
            self.gvf_scheduler.load_state_dict(checkpoint["gvf_scheduler"])
            start_epoch = int(checkpoint["epoch"]) + 1
            self.logger.log("Resuming chance PPO from checkpoint epoch %d." % checkpoint["epoch"])
            return start_epoch

        legacy_model_path = self._legacy_model_path()
        if osp.exists(legacy_model_path):
            try:
                model = torch.load(legacy_model_path, map_location=self.device, weights_only=False)
            except TypeError:
                model = torch.load(legacy_model_path, map_location=self.device)
            self.ac.load_state_dict(model.state_dict())
            self.logger.log("Loaded chance PPO actor-critic from model.pt. Optimizers start fresh.", color="yellow")
            return self._progress_start_epoch()
        else:
            self.logger.log("Resume requested but no checkpoint/model.pt was found. Starting from scratch.", color="yellow")
        return 0

    def chance_safety_constraint_ppo_bound(self, actor_critic=core.MLPActorCritic_auxiliary, ac_kwargs=dict(), resume=False):
        # Special function to avoid certain slowdowns from PyTorch + MPI combo.
        # setup_pytorch_for_mpi()
        print("worst-case PPO with alpha =", self.alpha)

        # Random seed
        self.seed += 10000 * proc_id()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        obs_dim = int(np.prod(self.env.observation_space.shape))
        act_dim = int(np.prod(self.env.action_space.shape))

        # Create actor-critic module
        self.ac = actor_critic(self.env.observation_space, self.env.action_space, **ac_kwargs).to(self.device)
        self.ac_old = copy.deepcopy(self.ac).to(self.device)

        # Count variables
        var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.v, self.ac.vc, self.ac.vg])
        self.logger.log('\nNumber of parameters: \t pi: %d, \t v: %d, \t vc: %d, \t vg: %d\n' % var_counts)

        # Set up optimizers for policy and value function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.pi_lr)
        self.penalty_optimizer = Adam([self.penalty_param], lr=self.penalty_lr)
        self.vf_optimizer = Adam(self.ac.v.parameters(), lr=self.vf_lr)
        self.cvf_optimizer = Adam(self.ac.vc.parameters(), lr=self.vf_lr)
        self.gvf_optimizer = Adam(self.ac.vg.parameters(), lr=self.vf_lr)

        epoch_decay = int(1e4) #int(5e4) #int(1e4) #1000
        self.pi_scheduler = StepLR(self.pi_optimizer, step_size=epoch_decay, gamma=0.9)
        self.vf_scheduler = StepLR(self.vf_optimizer, step_size=epoch_decay, gamma=0.9)
        self.cvf_scheduler = StepLR(self.cvf_optimizer, step_size=epoch_decay, gamma=0.9)
        self.gvf_scheduler = StepLR(self.gvf_optimizer, step_size=epoch_decay, gamma=0.9)
        self.penalty_scheduler = StepLR(self.penalty_optimizer, step_size=epoch_decay, gamma=0.9)

        # Set up model saving
        self.logger.setup_pytorch_saver(self.ac)
        start_epoch = self._load_resume_state() if resume else 0

        # Set up experience buffer
        local_steps = int(self.steps_per_epoch / self.num_procs)
        self.buf = PPOBuffer(obs_dim, act_dim, self.steps_per_epoch, self.gamma, self.lam)

        # Prepare for interaction with environment
        start_time = time.time()
        # o, ep_ret,ep_cret, ep_len = self.env.reset(), 0, 0, 0


        # Main loop: collect experience in env and update/log each epoch
        for epoch in range(start_epoch, self.epochs):
            self.ac_old.load_state_dict(self.ac.state_dict()) # Update old policy parameters -> theta_{old} = theta_{i-1}
            state_dict_cpu = {k: v.detach().cpu() for k, v in self.ac.state_dict().items()}
            with ctx.Pool(processes=self.num_procs) as pool:
                results = pool.starmap(
                    _rollout_worker,
                    [(wid, self.env_name, actor_critic, ac_kwargs,
                    self.seed, local_steps, self.gamma, self.lam, self.cost_limit, self.max_ep_len, state_dict_cpu, self.geometric_sample)
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
            all_ep_discount_cost = np.concatenate([r["ep_discount_costs"] for r in results], axis=0) if any(len(r["ep_discount_costs"]) for r in results) else np.array([], dtype=np.float32)
            all_ep_len  = np.concatenate([r["ep_lens"]  for r in results], axis=0) if any(len(r["ep_lens"]) for r in results) else np.array([], dtype=np.float32)

            for r in all_ep_ret:
                self.logger.store(EpRet=float(r))
            for c in all_ep_cost:
                self.logger.store(EpCost=float(c))
            for dc in all_ep_discount_cost:
                self.logger.store(EpDiscountCost=float(dc))
            for l in all_ep_len:
                self.logger.store(EpLen=float(l))

            N_traj = sum([r["N_traj"] for r in results])

            # Save model
            if (epoch % self.save_freq == 0) or (epoch == self.epochs-1):
                self.logger.save_state({'env': self.env}, None)
     
            # Perform update
            prob_est, prob_emprical = self.update(N_traj, epoch=epoch)

            # Log info about epoch
            self.logger.log_tabular('Epoch', epoch)
            self.logger.log_tabular('EpRet', with_min_and_max=False)
            self.logger.log_tabular('EpCost',with_min_and_max=False)
            if self.discout_cost:
                self.logger.log_tabular('EpDiscountCost',with_min_and_max=False)
            self.logger.log_tabular('EpLen', average_only=True)
            self.logger.log_tabular('Lambda', softplus(self.penalty_param).item())
            self.logger.log_tabular('ProbEst', prob_est)
            self.logger.log_tabular('ProbEmp', prob_emprical)
            self.logger.log_tabular('ProbEMA', self.p_bar)
            self.logger.log_tabular('TotalEnvInteracts', (epoch+1)*self.steps_per_epoch)
             # self.logger.log_tabular('VVals', with_min_and_max=True)
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
            self._save_checkpoint(epoch)
