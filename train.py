import numpy as np
import gym
import os, sys
from arguments import get_args
from rl_modules.ddpg_agent import ddpg_agent
import random
import torch
from mpi4py import MPI
from xarm_env.load_xarm7 import ReachXarm
from xarm_env.pick_and_place import PickAndPlaceXarm

"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)

"""
def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'action': env.action_space.shape[0],
            'action_max': env.action_space.high[0],
            }
    params['max_timesteps'] = env._max_episode_steps
    return params

def get_env(task):
    if task == 'FetchReach-v1':
        return ReachXarm(xml_path='./assets/fetch/reach_xarm_with_gripper.xml')
    elif task == 'FetchPickAndPlace-v1':
        return PickAndPlaceXarm(xml_path='./assets/fetch/pick_and_place_xarm.xml')
def launch(args):
    # create the ddpg_agent
    #env = gym.make(args.env_name)
    # print(env._max_episode_steps)
    # exit()
    env = get_env(args.env_name)
    # set random seeds for reproduce
    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    # get the environment parameters
    env_params = get_env_params(env)
    # create the ddpg agent to interact with the environment 
    ddpg_trainer = ddpg_agent(args, env, env_params)
    ddpg_trainer.learn()

if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    # get the params
    args = get_args()
    launch(args)
