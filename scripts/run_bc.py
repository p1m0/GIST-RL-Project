"""
Runs behavior cloning and DAgger for homework 1

Functions to edit:
    1. run_training_loop
"""

import argparse
import minari
import os
import time
import torch

import numpy as np
import gymnasium as gym

from infrastructure import utils
from infrastructure.logger import Logger
from infrastructure.replay_buffer import ReplayBuffer
from networks.bc_policy import MLPPolicyBC


# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
MJ_ENV_NAMES = {
    'HalfCheetah-v5': 'halfcheetah/expert-v0',
    'Ant-v5': 'ant/expert-v0',
    'Humanoid-v5': 'humanoid/expert-v0'
}


def run_training_loop(params):
    """
    Runs training with the specified parameters
    (behavior cloning or dagger)

    Args:
        params: experiment parameters
    """

    #############
    ## INIT
    #############
    
    # Get params, create logger, create TF session
    logger = Logger(params['logdir'])

    # Set random seeds
    seed = params['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set logger attributes
    log_video = True

    #############
    ## DATA COLLECTION
    #############

    print("\nCollecting data to be used for training...")

    replay_buffer = ReplayBuffer(params['max_replay_buffer_size'])
    
    dataset = minari.load_dataset('mujoco/' + MJ_ENV_NAMES[params['env_name']], download=True)
    for episode in dataset.iterate_episodes():
        # Add each episode to the replay buffer
        for idx in range(len(episode)):
            replay_buffer.insert(
                observation=episode.observations[idx],
                action=episode.actions[idx],
                reward=0, # we don't need reward for BC
                next_observation=episode.observations[idx], # we don't need next_observation for BC
                done=False # we don't need done for BC
            )
    print(f"Collected {len(replay_buffer)} transitions from the dataset.")

    #############
    ## ENV
    #############

    env = gym.make(params['env_name'], render_mode='rgb_array') #render_mode='rgb_array' or 'human'
    env.reset(seed=seed)

    params['ep_len'] = env.spec.max_episode_steps

    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]

    #############
    ## AGENT
    #############

    actor = MLPPolicyBC(
        ac_dim,
        ob_dim,
        params['n_layers'],
        params['hidden_size'],
        learning_rate=params['learning_rate'],
    )

    #######################
    ## TRAINING LOOP
    #######################

    # train agent (using sampled data from replay buffer)
    print('\nTraining agent using sampled data from replay buffer...')
    training_logs = []
    for itr in range(params['num_iter']):
        batch = replay_buffer.sample(params['train_batch_size'])
        ob_batch = batch['observations']
        ac_batch = batch['actions']
        
        train_log = actor.update(ob_batch, ac_batch)
        training_logs.append(train_log)
        if itr % params['scalar_log_freq'] == 0:
            eval_env = gym.make(params['env_name'], render_mode="rgb_array")
            trajectories = utils.sample_n_trajectories(
                eval_env,
                policy=actor,
                ntraj=100,
                max_length=eval_env.spec.max_episode_steps,
            )
            mean_return = np.mean([t["episode_statistics"]["r"] for t in trajectories])

            print('Iter: {} - {} : {}'.format(itr, 'eval_return', mean_return))
            logger.log_scalar(mean_return, 'eval_return', itr)

    if params['save_params']:
        print('\nSaving agent params')
        actor.save('{}/policy.pt'.format(params['logdir']))

    eval_env = gym.make(params['env_name'], render_mode="rgb_array")

    trajectories = utils.sample_n_trajectories(
        eval_env,
        policy=actor,
        ntraj=100,
        max_length=eval_env.spec.max_episode_steps,
    )

    returns = [t["episode_statistics"]["r"] for t in trajectories]
    ep_lens = [t["episode_statistics"]["l"] for t in trajectories]

    print(f"Evaluation over {len(trajectories)} trajectories:")
    print(f"Mean return: {np.mean(returns):.2f}, Mean episode length: {np.mean(ep_lens):.2f}")
    print(f"Std return: {np.std(returns):.2f}, Max return: {np.max(returns):.2f}, Min return: {np.min(returns):.2f}")
    print(f"Replay buffer size: {len(replay_buffer)}")

            

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--expert_data', '-ed', type=str, required=True) #relative to where you're running this script from
    parser.add_argument('--env_name', '-env', type=str, help=f'choices: {", ".join(MJ_ENV_NAMES.keys())}', required=True)
    parser.add_argument('--exp_name', '-exp', type=str, default='pick an experiment name', required=True)

    parser.add_argument('--num_iter', type=int, default=10000)
    parser.add_argument('--train_batch_size', type=int, default=256)

    parser.add_argument('--n_layers', type=int, default=3)  # depth, of policy to be learned
    parser.add_argument('--hidden_size', type=int, default=256)  # width of each layer, of policy to be learned
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-4)  # LR for supervised learning

    parser.add_argument('--max_replay_buffer_size', type=int, default=10000000)
    parser.add_argument('--save_params', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    # convert args to dictionary
    params = vars(args)

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    # directory for logging
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    ###################
    ### RUN TRAINING
    ###################

    run_training_loop(params)


if __name__ == "__main__":
    main()
