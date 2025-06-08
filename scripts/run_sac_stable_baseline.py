import argparse
import time

import numpy as np
import gymnasium as gym

from cv2 import log
from gymnasium.wrappers import RescaleAction, ClipAction, RecordEpisodeStatistics, NormalizeObservation
from stable_baselines3 import SAC


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
    parser.add_argument("--exp_name", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--soft_target_update_rate", type=float, default=0.005)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument("--total_steps", type=int, default=1000000)
    parser.add_argument("--random_steps", type=int, default=1000)

    parser.add_argument("--eval_interval", "-ei", type=int, default=1000)
    parser.add_argument("--num_eval_trajectories", "-neval", type=int, default=100)
    parser.add_argument("--num_render_trajectories", "-nvid", type=int, default=0)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-g", default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument('--save_params', action='store_true')

    args = parser.parse_args()

    env = gym.make(args.env_name, render_mode=None)
    # base_env = gym.make(args.env_name, render_mode=None)
    # env = RecordEpisodeStatistics(
    #         ClipAction(
    #             RescaleAction(
    #                 NormalizeObservation(
    #                     base_env,
    #                     epsilon=1e-8,
    #                 ),
    #                 min_action=-1.0, 
    #                 max_action=1.0,
    #             )
    #         )
    #     )

    log_dir = f"data/SB_{args.exp_name}_{args.env_name}_{time.strftime('%d-%m-%Y_%H-%M-%S')}/"

    policy_kwargs = dict(
        net_arch=dict(
            pi=[args.hidden_size] * args.n_layers,  # Policy network
            qf=[args.hidden_size] * args.n_layers   # Q-function network
        )
    )

    model = SAC("MlpPolicy", env,
                verbose=1,
                gamma=args.discount,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                tau=args.soft_target_update_rate,
                ent_coef=args.temperature,
                tensorboard_log=log_dir,
                policy_kwargs=policy_kwargs)
    
    # model = SAC("MlpPolicy", env,
    #             verbose=2,
    #             tensorboard_log=log_dir,
    #         )
    print(f"Model created in environment {args.env_name}, logging to {log_dir}")

    model.learn(total_timesteps=args.total_steps, log_interval=args.log_interval)

    print(f"Saving model parameters to {log_dir}")
    model.save(log_dir + 'sac_model')

    eval_env = gym.make(args.env_name, render_mode="rgb_array")

    trajectories = sample_n_trajectories(
        eval_env,
        policy=model,
        ntraj=args.num_eval_trajectories,
        max_length=eval_env.spec.max_episode_steps,
    )

    returns = [t["episode_statistics"]["r"] for t in trajectories]
    ep_lens = [t["episode_statistics"]["l"] for t in trajectories]

    print(f"Evaluation over {len(trajectories)} trajectories:")
    print(f"Mean return: {np.mean(returns):.2f}, Mean episode length: {np.mean(ep_lens):.2f}")
    print(f"Std return: {np.std(returns):.2f}, Max return: {np.max(returns):.2f}, Min return: {np.min(returns):.2f}")


if __name__ == "__main__":
    main()