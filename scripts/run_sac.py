import os
import time
import yaml
import argparse

from agents.sac_agent import SoftActorCritic
from infrastructure.replay_buffer import ReplayBuffer

import gymnasium as gym
from gymnasium import wrappers
import numpy as np
import torch
from infrastructure import pytorch_util as ptu
import tqdm

from infrastructure import utils
from infrastructure.logger import Logger


def run_training_loop(args: argparse.Namespace):
    # set random seeds
    logger = Logger(args.logdir)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # make the gym environment
    env = gym.make(args.env_name, render_mode='rgb_array')
    eval_env = gym.make(args.env_name, render_mode='rgb_array')
    render_env = gym.make(args.env_name, render_mode='rgb_array')

    ep_len = env.spec.max_episode_steps
    batch_size = args.batch_size

    ob_shape = env.observation_space.shape
    ac_dim = env.action_space.shape[0]

    # simulation timestep, will be used for video saving
    if "model" in dir(env):
        fps = 1 / env.model.opt.timestep
    else:
        fps = env.env.metadata["render_fps"]

    # initialize agent
    agent = SoftActorCritic(
        ob_shape,
        ac_dim,
        n_layers=args.n_layers,
        hidden_size=args.hidden_size,
        discount=0.99,
        soft_target_update_rate=0.005,
    )

    replay_buffer = ReplayBuffer()

    observation = env.reset()[0]

    for step in tqdm.trange(args.total_steps, dynamic_ncols=True):
        if step < args.random_steps:
            action = env.action_space.sample()
        else:
            action = agent.get_action(observation)

        # Step the environment and add the data to the replay buffer
        next_observation, reward, done, _, info = env.step(action)
        replay_buffer.insert(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            done=done and not info.get("TimeLimit.truncated", False),
        )

        if done:
            # logger.log_scalar(info["episode"]["r"], "train_return", step)
            # logger.log_scalar(info["episode"]["l"], "train_ep_len", step)
            observation = env.reset()[0]
        else:
            observation = next_observation

        # Train the agent
        if step >= args.random_steps:
            batch = replay_buffer.sample(batch_size)
            update_info = agent.update(batch['observations'], batch['actions'], batch['rewards'],
                                       batch['next_observations'], batch['dones'], step)

            # Logging
            update_info["actor_lr"] = agent.actor_lr_scheduler.get_last_lr()[0]
            update_info["critic_lr"] = agent.critic_lr_scheduler.get_last_lr()[0]

            if step % args.log_interval == 0:
                for k, v in update_info.items():
                    logger.log_scalar(v, k, step)
                    logger.log_scalars
                logger.flush()

        # Run evaluation
        if step % args.eval_interval == 0:
            trajectories = utils.sample_n_trajectories(
                eval_env,
                policy=agent,
                ntraj=args.num_eval_trajectories,
                max_length=ep_len,
            )
            returns = [t["episode_statistics"]["r"] for t in trajectories]
            ep_lens = [t["episode_statistics"]["l"] for t in trajectories]

            logger.log_scalar(np.mean(returns), "eval_return", step)
            logger.log_scalar(np.mean(ep_lens), "eval_ep_len", step)

            if len(returns) > 1:
                logger.log_scalar(np.std(returns), "eval/return_std", step)
                logger.log_scalar(np.max(returns), "eval/return_max", step)
                logger.log_scalar(np.min(returns), "eval/return_min", step)
                logger.log_scalar(np.std(ep_lens), "eval/ep_len_std", step)
                logger.log_scalar(np.max(ep_lens), "eval/ep_len_max", step)
                logger.log_scalar(np.min(ep_lens), "eval/ep_len_min", step)

            if args.num_render_trajectories > 0:
                video_trajectories = utils.sample_n_trajectories(
                    render_env,
                    agent,
                    args.num_render_trajectories,
                    ep_len,
                    render=True,
                )

                logger.log_paths_as_videos(
                    video_trajectories,
                    step,
                    fps=fps,
                    max_videos_to_save=args.num_render_trajectories,
                    video_title="eval_rollouts",
                )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--total_steps", type=int, default=1000000)
    parser.add_argument("--random_steps", type=int, default=10000)

    parser.add_argument("--eval_interval", "-ei", type=int, default=5000)
    parser.add_argument("--num_eval_trajectories", "-neval", type=int, default=10)
    parser.add_argument("--num_render_trajectories", "-nvid", type=int, default=0)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-g", default=0)
    parser.add_argument("--log_interval", type=int, default=1000)

    args = parser.parse_args()

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data")

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = (
        args.exp_name
        + "_"
        + args.env_name
        + "_"
        + time.strftime("%d-%m-%Y_%H-%M-%S")
    )
    logdir = os.path.join(data_path, logdir)
    args.logdir = logdir
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    run_training_loop(args)


if __name__ == "__main__":
    main()
