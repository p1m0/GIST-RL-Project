import argparse
import time

import numpy as np
import gymnasium as gym

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from torch import tensor
import torch.nn as nn


def sample_trajectory(env: gym.Env, policy, max_length: int, render: bool = False):
    """Sample a rollout in the environment from a policy."""
    ob = env.reset()[0]
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:

        ac: np.ndarray = policy.predict(ob)[0]
        next_ob, rew, done, _, info = env.step(ac)
        
        steps += 1
        rollout_done = done or steps >= max_length

        # record result of taking that action
        obs.append(ob)
        acs.append(ac)
        rewards.append(rew)
        next_obs.append(next_ob)
        # terminals.append(done)
        terminals.append(done and not info.get("TimeLimit.truncated", False))

        ob = next_ob

        # end the rollout if the rollout ended
        if rollout_done:
            break

    episode_statistics = {"l": steps, "r": np.sum(rewards)}
    if "episode" in info:
        episode_statistics.update(info["episode"])

    # env.close()

    return {
        "observation": np.array(obs, dtype=np.float32),
        "image_obs": np.array(image_obs, dtype=np.uint8),
        "reward": np.array(rewards, dtype=np.float32),
        "action": np.array(acs, dtype=np.float32),
        "next_observation": np.array(next_obs, dtype=np.float32),
        "terminal": np.array(terminals, dtype=np.float32),
        "episode_statistics": episode_statistics,
    }

def sample_n_trajectories(
    env: gym.Env, policy, ntraj: int, max_length: int, render: bool = False
):
    """Collect ntraj rollouts."""
    trajs = []
    for _ in range(ntraj):
        # collect rollout
        traj = sample_trajectory(env, policy, max_length, render)
        trajs.append(traj)
    return trajs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--n_iter", "-n", type=int, default=200)

    parser.add_argument("--use_baseline", action="store_true")
    parser.add_argument("--baseline_learning_rate", "-blr", type=float, default=5e-3)
    parser.add_argument("--baseline_gradient_steps", "-bgs", type=int, default=5)
    parser.add_argument("--use_reward_to_go", "-rtg", action="store_true")
    parser.add_argument("--gae_lambda", type=float, default=0.99)
    parser.add_argument("--normalize_advantages", "-na", action="store_true")
    parser.add_argument(
        "--batch_size", "-b", type=int, default=1000
    )  # steps collected per train iteration
    parser.add_argument(
        "--eval_batch_size", "-eb", type=int, default=400
    )  # steps collected per eval iteration

    parser.add_argument("--discount", type=float, default=1.0)
    parser.add_argument("--learning_rate", "-lr", type=float, default=5e-3)
    parser.add_argument("--n_layers", "-l", type=int, default=3)
    parser.add_argument("--layer_size", "-s", type=int, default=256)

    parser.add_argument(
        "--ep_len", type=int
    )  # students shouldn't change this away from env's default
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    parser.add_argument("--video_log_freq", type=int, default=-1)
    parser.add_argument("--scalar_log_freq", type=int, default=10)
    parser.add_argument('--save_params', action='store_true')

    args = parser.parse_args()

    log_dir = f"data/SB_{args.exp_name}_{args.env_name}_{time.strftime('%d-%m-%Y_%H-%M-%S')}/"

    """
    --discount 0.96 -n 1000 -l 2 -s 128 -b 5000 -lr 0.003 \
    --baseline_gradient_steps 10 \
    -na -rtg --use_baseline --gae_lambda 0.97 \
    --exp_name HalfCheetah
    """

    vec_env = make_vec_env(args.env_name, n_envs=1)
    policy_kwargs = {
        # "net_arch": [args.layer_size] * args.n_layers,
        "net_arch": [args.layer_size] * args.n_layers,
        "activation_fn": nn.Tanh,
    }

    model = A2C("MlpPolicy", vec_env, verbose=1,
                tensorboard_log=log_dir,
                learning_rate=args.learning_rate,
                n_steps=args.batch_size,
                normalize_advantage=args.normalize_advantages,
                use_rms_prop=False,
                gae_lambda=args.gae_lambda,
                gamma=args.discount,
                policy_kwargs=policy_kwargs)
    
    model.learn(total_timesteps=5000000, log_interval=args.scalar_log_freq)

    eval_env = gym.make(args.env_name, render_mode="rgb_array")

    trajectories = sample_n_trajectories(
        eval_env,
        policy=model,
        ntraj=100,
        max_length=eval_env.spec.max_episode_steps,
    )

    returns = [t["episode_statistics"]["r"] for t in trajectories]
    ep_lens = [t["episode_statistics"]["l"] for t in trajectories]

    print(f"Evaluation over {len(trajectories)} trajectories:")
    print(f"Mean return: {np.mean(returns):.2f}, Mean episode length: {np.mean(ep_lens):.2f}")
    print(f"Std return: {np.std(returns):.2f}, Max return: {np.max(returns):.2f}, Min return: {np.min(returns):.2f}")

    model.save(log_dir + "sb_a2c_model")

if __name__ == "__main__":
    main()