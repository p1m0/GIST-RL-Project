import argparse
import time

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from torch import tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--n_iter", "-n", type=int, default=200)

    parser.add_argument("--use_reward_to_go", "-rtg", action="store_true")
    parser.add_argument("--use_baseline", action="store_true")
    parser.add_argument("--baseline_learning_rate", "-blr", type=float, default=5e-3)
    parser.add_argument("--baseline_gradient_steps", "-bgs", type=int, default=5)
    parser.add_argument("--gae_lambda", type=float, default=None)
    parser.add_argument("--normalize_advantages", "-na", action="store_true")
    parser.add_argument(
        "--batch_size", "-b", type=int, default=1000
    )  # steps collected per train iteration
    parser.add_argument(
        "--eval_batch_size", "-eb", type=int, default=400
    )  # steps collected per eval iteration

    parser.add_argument("--discount", type=float, default=1.0)
    parser.add_argument("--learning_rate", "-lr", type=float, default=5e-3)
    parser.add_argument("--n_layers", "-l", type=int, default=2)
    parser.add_argument("--layer_size", "-s", type=int, default=64)

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

    log_dir = f"data/SB_A2C_{args.exp_name}_{args.env_name}_{time.strftime("%d-%m-%Y_%H-%M-%S")}/"

    vec_env = make_vec_env(args.env_name, n_envs=4)

    model = A2C("MlpPolicy", vec_env, verbose=1, tensorboard_log=log_dir)
    model.learn(total_timesteps=args.n_iter, log_interval=args.scalar_log_freq)

    if args.save_params:
        model.save(log_dir + "a2c_model.pth")

if __name__ == "__main__":
    main()