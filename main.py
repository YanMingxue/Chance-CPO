#!/usr/bin/env python3
"""
Name: main.py
Author: Mingxue Yan
Date: 02/02/2026
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_VISIBLE_DEVICES"] = "-1"

import gym
import torch
from ppo import PPO_Train_Agent  
from chance_ppo import ChancePPO_Train_Agent
import core
from utils.run_utils import setup_logger_kwargs

def main(args):
    if args.resume and args.algo not in ('ppo', 'chance_ppo'):
        raise NotImplementedError("--resume is currently implemented for ppo and chance_ppo only.")

    if args.algo == 'ppo':
        exp_name = f"{args.env}/{args.algo}_{args.cost_limit}/seed{args.seed}" 
    elif args.algo ==('chance_ppo'):
        exp_name = f"{args.env}/{args.algo}_alpha{args.alpha}_{args.cost_limit}/seed{args.seed}" 
    else:
        raise NotImplementedError
    
    logger_kwargs = setup_logger_kwargs(exp_name, args.seed)
    logger_kwargs["append"] = args.resume
    num_steps = 1e8 # Can use 5e7 for the first 2 envs
    steps_per_epoch = 30000
    epochs = int(num_steps / steps_per_epoch)
    if args.algo == 'chance_ppo':
        chance_PPO = ChancePPO_Train_Agent(args.env, num_procs=args.num_procs, gamma=args.gamma, beta=args.beta,
                                clip_ratio=args.clip_ratio, pi_lr=args.pi_lr, vf_lr=args.vf_lr, penalty_lr=args.penalty_lr, seed=args.seed, 
                                steps_per_epoch=steps_per_epoch, epochs=epochs, target_kl=args.target_kl,
                                multistep_penalty=True, discout_cost=False, kl_bound=False, geometric_sample=True, logger_kwargs=logger_kwargs, 
                                alpha=args.alpha, cost_limit=args.cost_limit, lambda_init=args.lambda_init, device=args.device)
        chance_PPO.chance_safety_constraint_ppo_bound(actor_critic=core.MLPActorCritic_auxiliary,
                                                ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
                                                resume=args.resume)
    elif args.algo == 'ppo':  
        PPO = PPO_Train_Agent(args.env, num_procs=args.num_procs, gamma=args.gamma, 
                               clip_ratio=args.clip_ratio, pi_lr=args.pi_lr, vf_lr=args.vf_lr, penalty_lr=args.penalty_lr, seed=args.seed, 
                               steps_per_epoch=steps_per_epoch, target_kl=args.target_kl,
                               epochs=epochs,logger_kwargs=logger_kwargs,
                               cost_limit=args.cost_limit, lambda_init=args.lambda_init, device=args.device)
        PPO.ppo(actor_critic=core.MLPActorCritic, ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
                resume=args.resume)
    # elif args.algo == 'cvar':
    #     cvar = CVaR_Train_Agent(args.env, num_procs=args.num_procs, gamma=args.gamma, 
    #                            clip_ratio=args.clip_ratio, pi_lr=args.pi_lr, vf_lr=args.vf_lr, penalty_lr=args.penalty_lr, seed=args.seed, 
    #                            steps_per_epoch=steps_per_epoch, target_kl=args.target_kl,
    #                            epochs=epochs,logger_kwargs=logger_kwargs, alpha=args.alpha,
    #                            cost_limit=args.cost_limit, lambda_init=args.lambda_init, device=args.device)
    #     cvar.cvar(actor_critic=core.MLPActorCritic, ac_kwargs=dict(hidden_sizes=[args.hid]*args.l))
    else:
        raise NotImplementedError



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    
    # -------------------------------------------------#
    ''' Point Goal1 - v0'''

    parser.add_argument('--env', type=str, default='Safexp-PointGoal1-v0')
    parser.add_argument('--cost_limit', type=float, default=25)
    parser.add_argument('--exp_num', type=int, default='1')
    parser.add_argument('--alpha', type=float, default= 0.3) 
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--penalty_lr', type=float, default= 8e-5) 
    parser.add_argument('--lambda_init', type=float, default=1.0)
    parser.add_argument('--algo', type=str, default='chance_ppo', help="algorithm to use: ppo, chance_ppo")


    # -------------------------------------------------#
    ''' Car Goal1 - v0'''

    # parser.add_argument('--env', type=str, default='Safexp-CarGoal1-v0')
    # parser.add_argument('--exp_num', type=int, default='1')
    # parser.add_argument('--alpha', type=float, default= 0.1) 
    # parser.add_argument('--cost_limit', type=float, default=25)
    # parser.add_argument('--seed', '-s', type=int, default=0)
    # parser.add_argument('--penalty_lr', type=float, default= 8e-5) 
    # parser.add_argument('--lambda_init', type=float, default=1.0)
    # parser.add_argument('--algo', type=str, default='chance_ppo', help="algorithm to use: ppo, chance_ppo")

    # -------------------------------------------------#
    ''' Doggo Goal1 - v0'''

    # parser.add_argument('--env', type=str, default='Safexp-DoggoGoal1-v0')
    # parser.add_argument('--exp_num', type=int, default='1')
    # parser.add_argument('--alpha', type=float, default= 0.2) 
    # parser.add_argument('--cost_limit', type=float, default=25)
    # parser.add_argument('--seed', '-s', type=int, default=3)
    # parser.add_argument('--lambda_init', type=float, default=1.0) #0.6# chance ppo)
    # parser.add_argument('--penalty_lr', type=float, default= 2e-4) 
    # parser.add_argument('--algo', type=str, default='ppo', help="algorithm to use: ppo, chance_ppo")

    # parser.add_argument('--env', type=str, default='Safexp-DoggoGoal1-v0')
    # parser.add_argument('--exp_num', type=int, default='1')
    # parser.add_argument('--alpha', type=float, default= 0.2) 
    # parser.add_argument('--cost_limit', type=float, default=25)
    # parser.add_argument('--seed', '-s', type=int, default=0)
    # parser.add_argument('--lambda_init', type=float, default=0.6) #0.6 chance ppo # 1.0 ppo)
    # parser.add_argument('--penalty_lr', type=float, default= 1e-4) #2e-4) 
    # parser.add_argument('--algo', type=str, default='chance_ppo', help="algorithm to use: ppo, chance_ppo")

    # -------------------------------------------------#

    
    parser.add_argument("--device", type=str, default=None, help="device to use: cpu or cuda")
    parser.add_argument('--hid', type=int, default=128)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--num_procs', type=int, default=5)
    parser.add_argument('--gamma', type=float, default= 0.999) # 0.99)
    parser.add_argument('--beta', type=float, default= 0.2, help="EMA parameter")
    parser.add_argument('--clip_ratio', type=float, default=0.15)
    parser.add_argument('--pi_lr', type=float, default= 1e-4)
    parser.add_argument('--vf_lr', type=float, default=3e-4)
    parser.add_argument('--target_kl', type=float, default= 0.01) 
    parser.add_argument('--resume', action='store_true', help='resume PPO/chance PPO from data checkpoint if available')
    args = parser.parse_args()

    if args.device is None: 
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", args.device)

    main(args)
