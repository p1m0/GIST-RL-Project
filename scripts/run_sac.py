from calendar import c
import os
import time
import argparse

from agents.sac_agent import SoftActorCritic
from infrastructure.replay_buffer import ReplayBuffer

import gymnasium as gym
import numpy as np
import torch
import tqdm

from gymnasium.wrappers import RescaleAction, ClipAction, RecordEpisodeStatistics, NormalizeObservation

from infrastructure import utils
from infrastructure.logger import Logger
from infrastructure import pytorch_util as ptu


def run_training_loop(args: argparse.Namespace):
    # set random seeds
    ptu.init_gpu(not args.no_gpu, args.which_gpu)

    logger = Logger(args.logdir)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # make the gym environment
    env = RecordEpisodeStatistics(
            ClipAction(
                RescaleAction(
                    NormalizeObservation(
                        gym.make(args.env_name, render_mode=None),
                        epsilon=1e-8,
                    ),
                    min_action=-1.0, 
                    max_action=1.0,
                )
            )
        )
    
    eval_env = RecordEpisodeStatistics(
            ClipAction(
                RescaleAction(
                    NormalizeObservation(
                        gym.make(args.env_name, render_mode=None),
                        epsilon=1e-8,
                    ),
                    min_action=-1.0, 
                    max_action=1.0,
                )
            )
        )

    ep_len = env.spec.max_episode_steps
    batch_size = args.batch_size

    ob_shape = env.observation_space.shape
    ac_dim = env.action_space.shape[0]

    # initialize agent
    agent = SoftActorCritic(
        ob_shape,
        ac_dim,
        learning_rate=args.learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        alpha_learning_rate=args.alpha_learning_rate,
        n_layers=args.n_layers,
        hidden_size=args.hidden_size,
        discount=args.discount,
        temperature=args.temperature,
        soft_target_update_rate=args.soft_target_update_rate,
        activation=args.activation,
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
                                       batch['next_observations'], batch['dones'])

            # Logging
            update_info["actor_lr"] = agent.actor_lr_scheduler.get_last_lr()[0]
            update_info["critic_lr"] = agent.critic_lr_scheduler.get_last_lr()[0]

            if step % args.log_interval == 0:
                for k, v in update_info.items():
                    logger.log_scalar(v, k, step)
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

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--critic_learning_rate", type=float, default=5e-5)
    parser.add_argument("--alpha_learning_rate", type=float, default=2e-5)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--soft_target_update_rate", type=float, default=0.005)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument("--total_steps", type=int, default=1000000)
    parser.add_argument("--random_steps", type=int, default=1000)

    parser.add_argument("--eval_interval", "-ei", type=int, default=1000)
    parser.add_argument("--num_eval_trajectories", "-neval", type=int, default=10)
    parser.add_argument("--num_render_trajectories", "-nvid", type=int, default=0)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-g", default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument('--save_params', action='store_true')

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
