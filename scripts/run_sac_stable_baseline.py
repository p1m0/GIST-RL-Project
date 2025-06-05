import argparse
import time

from cv2 import log
import gymnasium as gym

from stable_baselines3 import SAC


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
    parser.add_argument("--log_interval", type=int, default=4)
    parser.add_argument('--save_params', action='store_true')

    args = parser.parse_args()

    env = gym.make(args.env_name, render_mode="rgb_array")

    log_dir = f"data/SB_SAC_{args.exp_name}_{args.env_name}_{time.strftime('%d-%m-%Y_%H-%M-%S')}/"

    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    model.learn(total_timesteps=args.total_steps, log_interval=args.log_interval)
    
    if args.save_params:
        model.save(log_dir + "a2c_model.pth")

if __name__ == "__main__":
    main()