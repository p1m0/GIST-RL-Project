import os
import time
import torch
import tqdm

import gymnasium as gym
import numpy as np

from agents.a2c_agent import PGAgent

from infrastructure import pytorch_util as ptu
from infrastructure import utils
from infrastructure.logger import Logger

MAX_NVIDEO = 2

def run_training_loop(args):
    logger = Logger(args.logdir)

    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # make the gym environment
    env = gym.make(args.env_name, render_mode='rgb_array')

    max_ep_len = args.ep_len or env.spec.max_episode_steps

    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]

    # simulation timestep, will be used for video saving
    if hasattr(env, "model"):
        fps = 1 / env.model.opt.timestep
    else:
        fps = env.env.metadata["render_fps"]

    # initialize agent
    agent = PGAgent(
        ob_dim,
        ac_dim,
        discrete=False,
        n_layers=args.n_layers,
        layer_size=args.layer_size,
        gamma=args.discount,
        learning_rate=args.learning_rate,
        use_baseline=args.use_baseline,
        use_reward_to_go=args.use_reward_to_go,
        normalize_advantages=args.normalize_advantages,
        baseline_learning_rate=args.baseline_learning_rate,
        baseline_gradient_steps=args.baseline_gradient_steps,
        gae_lambda=args.gae_lambda,
    )

    total_envsteps = 0
    start_time = time.time()

    # for itr in range(args.n_iter):
    for itr in tqdm.trange(int(args.total_timesteps / args.batch_size), dynamic_ncols=True):
        trajs, envsteps_this_batch = utils.sample_trajectories(
            env,
            agent.actor,
            args.batch_size,
            max_ep_len,
        )

        total_envsteps += envsteps_this_batch

        # trajs should be a list of dictionaries of NumPy arrays, where each dictionary corresponds to a trajectory.
        # this line converts this into a single dictionary of lists of NumPy arrays.
        trajs_dict = {k: [traj[k] for traj in trajs] for k in trajs[0]}

        train_info: dict = agent.update(
            trajs_dict["observation"],
            trajs_dict["action"],
            trajs_dict["reward"],
            trajs_dict["terminal"],
        )

        if itr % args.scalar_log_freq == 0:
            env.reset()
            trajectories = utils.sample_n_trajectories(
                env,
                policy=agent.actor,
                ntraj=10,
                max_length=max_ep_len,
            )
            returns = [t["episode_statistics"]["r"] for t in trajectories]
            ep_lens = [t["episode_statistics"]["l"] for t in trajectories]

            logger.log_scalar(np.mean(returns), "eval_return", total_envsteps)
            logger.log_scalar(np.mean(ep_lens), "eval_ep_len", total_envsteps)

            if len(returns) > 1:
                logger.log_scalar(np.std(returns), "eval/return_std", total_envsteps)
                logger.log_scalar(np.max(returns), "eval/return_max", total_envsteps)
                logger.log_scalar(np.min(returns), "eval/return_min", total_envsteps)
                logger.log_scalar(np.std(ep_lens), "eval/ep_len_std", total_envsteps)
                logger.log_scalar(np.max(ep_lens), "eval/ep_len_max", total_envsteps)
                logger.log_scalar(np.min(ep_lens), "eval/ep_len_min", total_envsteps)
            # save eval metrics


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    # parser.add_argument("--n_iter", "-n", type=int, default=200)
    parser.add_argument("--total_timesteps", type=int, default=5000000)

    parser.add_argument("--use_reward_to_go", "-rtg", action="store_true")
    parser.add_argument("--use_baseline", action="store_true")
    parser.add_argument("--baseline_learning_rate", "-blr", type=float, default=5e-3)
    parser.add_argument("--baseline_gradient_steps", "-bgs", type=int, default=5)
    parser.add_argument("--gae_lambda", type=float, default=0.99)
    parser.add_argument("--normalize_advantages", "-na", action="store_true")
    parser.add_argument(
        "--batch_size", "-b", type=int, default=1000
    )  # steps collected per train iteration
    parser.add_argument(
        "--eval_batch_size", "-eb", type=int, default=400
    )  # steps collected per eval iteration

    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--learning_rate", "-lr", type=float, default=5e-3)
    parser.add_argument("--n_layers", "-l", type=int, default=2)
    parser.add_argument("--layer_size", "-s", type=int, default=256)

    parser.add_argument(
        "--ep_len", type=int
    )  # students shouldn't change this away from env's default
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    parser.add_argument("--video_log_freq", type=int, default=-1)
    parser.add_argument("--scalar_log_freq", type=int, default=10)

    args = parser.parse_args()

    # create directory for logging

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
