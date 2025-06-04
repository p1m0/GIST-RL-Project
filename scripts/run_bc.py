"""
Runs behavior cloning and DAgger for homework 1

Functions to edit:
    1. run_training_loop
"""

import argparse
import minari
import pickle
import os
import time
import torch

import numpy as np
import gymnasium as gym
import pandas as pd

from infrastructure import pytorch_util as ptu
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

    #############
    ## ENV
    #############

    env = gym.make(params['env_name'], render_mode='rgb_array') #render_mode='rgb_array' or 'human'
    env.reset(seed=seed)

    params['ep_len'] = params['ep_len'] or env.spec.max_episode_steps

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
        params['size'],
        learning_rate=params['learning_rate'],
    )

    #######################
    ## TRAINING LOOP
    #######################

    # train agent (using sampled data from replay buffer)
    print('\nTraining agent using sampled data from replay buffer...')
    training_logs = []
    for _ in range(params['num_agent_train_steps_per_iter']):
        batch = replay_buffer.sample(params['train_batch_size'])
        ob_batch = batch['observations']
        ac_batch = batch['actions']
        
        train_log = actor.update(ob_batch, ac_batch)
        training_logs.append(train_log)

    print('\nBeginning logging procedure...')
    if log_video:
        print('\nCollecting video rollouts eval')
        eval_video_paths = utils.sample_n_trajectories(
            env, actor, MAX_NVIDEO, params['ep_len'], True)

        logger.log_paths_as_videos(
            eval_video_paths, 1,
            fps=15,
            max_videos_to_save=MAX_NVIDEO,
            video_title='eval_rollouts')

    print("\nCollecting data for eval...")
    eval_paths, _ = utils.sample_trajectories(
        env, actor, params['eval_batch_size'], params['ep_len'])

    logs = utils.compute_metrics(eval_paths, eval_paths)

    # perform the logging
    for key, value in logs.items():
        print('{} : {}'.format(key, value))
        logger.log_scalar(value, key, 0)
    print('Done logging...\n\n')

    if params['save_params']:
        print('\nSaving agent params')
        actor.save('{}/policy.pt'.format(params['logdir']))
            

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--expert_data', '-ed', type=str, required=True) #relative to where you're running this script from
    parser.add_argument('--env_name', '-env', type=str, help=f'choices: {", ".join(MJ_ENV_NAMES.keys())}', required=True)
    parser.add_argument('--exp_name', '-exp', type=str, default='pick an experiment name', required=True)
    parser.add_argument('--ep_len', type=int)

    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1000)  # number of gradient steps for training policy (per iter in n_iter)

    parser.add_argument('--batch_size', type=int, default=1000)  # training data collected (in the env) during each iteration
    parser.add_argument('--eval_batch_size', type=int,
                        default=1000)  # eval data collected (in the env) for logging metrics
    parser.add_argument('--train_batch_size', type=int,
                        default=100)  # number of sampled data points to be used per gradient/train step

    parser.add_argument('--n_layers', type=int, default=2)  # depth, of policy to be learned
    parser.add_argument('--size', type=int, default=64)  # width of each layer, of policy to be learned
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)  # LR for supervised learning

    parser.add_argument('--video_log_freq', type=int, default=5)
    parser.add_argument('--scalar_log_freq', type=int, default=1)
    parser.add_argument('--max_replay_buffer_size', type=int, default=1000000)
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
