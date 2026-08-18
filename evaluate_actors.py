#!/usr/bin/env python3
"""
Evaluate saved PPO and chance PPO actors without training.

The script scans saved PyTorch models under data/, runs evaluation episodes in
their configured environments, and mirrors the data/ directory layout under
data_evaluation/.
"""
import argparse
import csv
import json
import os
import os.path as osp
from pathlib import Path

import gym
import numpy as np
import safety_gym  # noqa: F401  # registers Safety Gym environments
import torch

import core  # noqa: F401  # required when unpickling saved actor-critic objects


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4, sort_keys=True)


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def model_run_dir(model_path):
    path = Path(model_path)
    if path.parent.name != "pyt_save":
        raise ValueError(f"Unexpected model path: {model_path}")
    return path.parent.parent


def infer_method(run_dir):
    parts = run_dir.parts
    for part in reversed(parts):
        if part.startswith("chance_ppo"):
            return "chance_ppo"
        if part.startswith("ppo"):
            return "ppo"
    return "unknown"


def find_runs(data_dir, envs=None, methods=("ppo", "chance_ppo")):
    data_dir = Path(data_dir)
    runs = []
    for model_path in sorted(data_dir.glob("**/pyt_save/model.pt")):
        run_dir = model_run_dir(model_path)
        config_path = run_dir / "config.json"
        if not config_path.exists():
            continue

        method = infer_method(run_dir)
        if method not in methods:
            continue

        config = load_json(config_path)
        env_name = config.get("env_name")
        if envs and env_name not in envs:
            continue

        runs.append(
            {
                "run_dir": run_dir,
                "model_path": model_path,
                "config_path": config_path,
                "config": config,
                "env_name": env_name,
                "method": method,
            }
        )
    return runs


def load_model(model_path, device):
    try:
        model = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()
    return model


def select_action(model, obs, action_space, deterministic, device):
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    with torch.no_grad():
        pi = model.pi._distribution(obs_t)
        if deterministic:
            if hasattr(pi, "probs"):
                action_t = torch.argmax(pi.probs)
            elif hasattr(pi, "mean"):
                action_t = pi.mean
            else:
                action_t = pi.sample()
        else:
            action_t = pi.sample()

    action = action_t.detach().cpu().numpy()
    if hasattr(action_space, "low") and hasattr(action_space, "high"):
        action = np.clip(action, action_space.low, action_space.high)
    return action


def seed_env(env, seed):
    if seed is None:
        return
    try:
        env.seed(seed)
    except AttributeError:
        pass
    try:
        env.action_space.seed(seed)
    except AttributeError:
        pass


def reset_env(env):
    result = env.reset()
    if isinstance(result, tuple):
        return result[0]
    return result


def step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = terminated or truncated
        return obs, reward, done, info
    return result


def evaluate_run(run, episodes, deterministic, seed, device):
    config = run["config"]
    env_name = run["env_name"]
    cost_limit = float(config.get("cost_limit", 25))
    max_ep_len = int(config.get("max_ep_len", 1000))

    model = load_model(run["model_path"], device)
    env = gym.make(env_name)
    seed_env(env, seed)

    rows = []
    missing_cost_steps = 0
    step_errors = 0

    for episode in range(episodes):
        obs = reset_env(env)
        ep_reward = 0.0
        ep_cost = 0.0
        ep_len = 0
        done = False
        error = ""

        while not done and ep_len < max_ep_len:
            action = select_action(model, obs, env.action_space, deterministic, device)
            try:
                obs, reward, done, info = step_env(env, action)
            except Exception as exc:
                step_errors += 1
                error = f"{type(exc).__name__}: {exc}"
                done = True
                break

            if "cost" not in info:
                missing_cost_steps += 1
            cost = float(info.get("cost", 0.0))

            ep_reward += float(reward)
            ep_cost += cost
            ep_len += 1

        rows.append(
            {
                "episode": episode,
                "reward": ep_reward,
                "cost": ep_cost,
                "length": ep_len,
                "violation": int(ep_cost >= cost_limit),
                "error": error,
            }
        )

    env.close()

    rewards = np.asarray([row["reward"] for row in rows], dtype=np.float64)
    costs = np.asarray([row["cost"] for row in rows], dtype=np.float64)
    violations = np.asarray([row["violation"] for row in rows], dtype=np.float64)
    lengths = np.asarray([row["length"] for row in rows], dtype=np.float64)

    summary = {
        "env_name": env_name,
        "method": run["method"],
        "source_run_dir": str(run["run_dir"]),
        "source_model": str(run["model_path"]),
        "episodes": int(episodes),
        "deterministic": bool(deterministic),
        "seed": seed,
        "cost_limit": cost_limit,
        "violation_probability": float(violations.mean()),
        "average_reward": float(rewards.mean()),
        "std_reward": float(rewards.std()),
        "average_cost": float(costs.mean()),
        "std_cost": float(costs.std()),
        "average_length": float(lengths.mean()),
        "missing_cost_steps": int(missing_cost_steps),
        "step_errors": int(step_errors),
    }
    return rows, summary


def output_dir_for_run(run_dir, data_dir, output_dir):
    rel = Path(run_dir).relative_to(Path(data_dir))
    return Path(output_dir) / rel


def main(args):
    methods = ("ppo", "chance_ppo") if args.method == "all" else (args.method,)
    device = torch.device(args.device)

    runs = find_runs(args.data_dir, envs=args.env, methods=methods)
    if args.limit is not None:
        runs = runs[: args.limit]
    if not runs:
        raise RuntimeError("No matching saved PPO or chance PPO runs were found.")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    all_summaries = []

    for idx, run in enumerate(runs, start=1):
        print(f"[{idx}/{len(runs)}] Evaluating {run['run_dir']}")
        rows, summary = evaluate_run(
            run=run,
            episodes=args.episodes,
            deterministic=args.deterministic,
            seed=args.seed,
            device=device,
        )

        out_dir = output_dir_for_run(run["run_dir"], args.data_dir, args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_csv(
            out_dir / "episodes.csv",
            rows,
            fieldnames=["episode", "reward", "cost", "length", "violation", "error"],
        )
        save_json(out_dir / "summary.json", summary)
        all_summaries.append(summary)

    summary_fields = [
        "env_name",
        "method",
        "source_run_dir",
        "source_model",
        "episodes",
        "deterministic",
        "seed",
        "cost_limit",
        "violation_probability",
        "average_reward",
        "std_reward",
        "average_cost",
        "std_cost",
        "average_length",
        "missing_cost_steps",
        "step_errors",
    ]
    write_csv(Path(args.output_dir) / "summary.csv", all_summaries, summary_fields)

    env_names = sorted({summary["env_name"] for summary in all_summaries})
    for env_name in env_names:
        env_rows = [summary for summary in all_summaries if summary["env_name"] == env_name]
        env_dir = Path(args.output_dir) / env_name
        env_dir.mkdir(parents=True, exist_ok=True)
        write_csv(env_dir / "summary.csv", env_rows, summary_fields)

    print(f"Saved evaluation records to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="data_evaluation")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--env", action="append", default=None, help="Environment name to evaluate. Can be repeated.")
    parser.add_argument("--method", choices=["all", "ppo", "chance_ppo"], default="all")
    parser.add_argument("--deterministic", action="store_true", help="Use mean/argmax actions instead of sampling.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N matching runs.")
    main(parser.parse_args())
