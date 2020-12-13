from rl_modules.ddpg_agent import model_factory
import torch
import os
from datetime import datetime
import numpy as np
from rl_modules.replay_buffer import replay_buffer, new_replay_buffer
from rl_modules.image_only_replay_buffer import image_replay_buffer, state_replay_buffer
from rl_modules.models import actor, critic, asym_goal_outside_image, sym_image, sym_image_critic
from mpi_utils.normalizer import normalizer
from her_modules.her import her_sampler, her_sampler_new
from mujoco_py.modder import TextureModder, MaterialModder, CameraModder, LightModder
import cv2
import itertools
import matplotlib.pyplot as plt
from rl_modules.cheap_model import cheap_cnn
from mujoco_py import load_model_from_path, MjSim, MjViewer, MjRenderContextOffscreen
from mujoco_py.generated import const
from rl_modules.utils import plot_grad_flow
from torch import autograd
import time
import torch.nn as nn
from mpi4py import MPI
from mpi_utils.mpi_utils import sync_networks, sync_grads
from rl_modules.utils import timeit
from rl_modules.trajectory import Trajectory
from rl_modules.base import Agent
import random
import gym
from rl_modules.utils import Benchmark
import numpy as np
import torch
import torch.nn.functional as F


"""
the replay buffer here is basically from the openai baselines code

"""
class replay_buffer:
    def __init__(self, env_params, buffer_size):
        self.env_params = env_params
        self.T = env_params['max_timesteps']
        self.size = int(buffer_size)


        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        # create the buffer to store info
        self.buffers = {'obs_states': np.empty([self.size, self.env_params['obs']]),
                        'obs_img': np.empty([self.size, 100, 100, 3], dtype=np.uint8),
                        'g_states': np.empty([self.size, self.env_params['goal']]),
                        }
    
    # store the episode
    def store_episode(self, observation):
        obs_state, obs_img, goal = observation["observation"], observation["observation_image"], observation["desired_goal"]
     
        batch_size = 1

        idxs = self._get_storage_idx(inc=batch_size)
        # store the informations
        self.buffers['obs_img'][idxs] = obs_img
        self.buffers['g_states'][idxs] = goal
        self.buffers['obs_states'][idxs] = obs_state
        self.n_transitions_stored += self.T * batch_size
    
    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffers = {}
        for key in self.buffers.keys():
            temp_buffers[key] = self.buffers[key][:self.current_size]
        random_idxs = np.random.choice(self.current_size, batch_size, replace=False)

        return temp_buffers["obs_states"][random_idxs], temp_buffers["g_states"][random_idxs], temp_buffers["obs_img"][random_idxs]

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size+inc)
        if inc == 1:
            idx = idx[0]
        return idx



def show_video(img):
    cv2.imshow('frame', cv2.resize(img, (200,200)))
    cv2.waitKey()

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

def _select_actions(args, env_params, pi):
    action = pi.cpu().detach().numpy().squeeze()
    # add the gaussian
    action += args.noise_eps * env_params['action_max'] * np.random.randn(*action.shape)
    action = np.clip(action, -env_params['action_max'], env_params['action_max'])
    # random actions...
    random_actions = np.random.uniform(low=-env_params['action_max'], high=env_params['action_max'], \
                                        size=env_params['action'])
    # choose if use the random actions
    action += np.random.binomial(1, args.random_eps, 1)[0] * (random_actions - action)
    return action


def eval_agent(env, net, args):
    total_success_rate = []
    for _ in range(5):
        per_success_rate = []
        observation = env.reset()
        for _ in range(50):
            obs_img = env.render(mode="rgb_array", height=100, width=100)
            #show_video(obs_img)
            obs_img = obs_img[np.newaxis, :].copy()
            obs_img = torch.tensor(obs_img, dtype=torch.float32)
            obs_img = obs_img.permute(0, 3, 1, 2)

            if args.cuda:
                obs_img = obs_img.cuda()

            with torch.no_grad():
                pi = net(obs_img)
                actions = pi.detach().cpu().numpy().squeeze()

            observation_new, _, _, info = env.step(actions)
            obs_img = env.render(mode="rgb_array", height=100, width=100)

            per_success_rate.append(info['is_success'])
        total_success_rate.append(per_success_rate)
    total_success_rate = np.array(total_success_rate)
    local_success_rate = np.mean(total_success_rate[:, -1])
    return local_success_rate

def load_teacher_student(args):

    env = gym.make('FetchPush-v1')
    viewer = MjRenderContextOffscreen(env.sim)
    viewer.cam.distance = 1.2 # this will be randomized baby: domain randomization FTW
    viewer.cam.azimuth = 180 # this will be randomized baby: domain Randomization FTW
    viewer.cam.elevation = -25 # this will be randomized baby: domain Randomization FTW
    env.env._viewers['rgb_array'] = viewer

    env_params = get_env_params(env)
    env_params["load_saved"] = False

    # load teacher
    # teacher_path = 'saved_models/sym_state/FetchPush-v1/model.pt'
    teacher_path = 'sym_server_weights/sym_state/FetchPush-v1/model.pt'
    obj = torch.load(teacher_path, map_location=lambda storage, loc: storage)

    # init teacher
    teacher_network, _, _, _ = model_factory('sym_state', env_params)
    teacher_network.load_state_dict(obj['actor_net'])

    # init student
    student_network, _, _, _ = model_factory('asym_goal_in_image', env_params)

    if args.cuda:
        student_network.cuda()
        teacher_network.cuda()

    student_optim = torch.optim.Adam(student_network.parameters(), lr=args.lr_actor)

    def _preproc_inputs_image(obs_img):
        obs_img = torch.tensor(obs_img, dtype=torch.float32)
        obs_img = obs_img.permute(0, 3, 1, 2)
        
        if args.cuda:
            obs_img = obs_img.cuda()
        return obs_img
    
    def _preproc_inputs(obs, g):
        obs_norm = np.clip((obs - obj['o_mean'])/obj['o_std'], -args.clip_range, args.clip_range)
        g_norm = np.clip((g - obj['g_mean'])/obj['g_std'], -args.clip_range, args.clip_range)
        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, g_norm], axis=1)
        inputs = torch.tensor(inputs, dtype=torch.float32)
        if args.cuda:
            inputs = inputs.cuda()
        return inputs

    steps = 0
    episodes = 0
    buffer = replay_buffer(env_params, 1e5)
    batch_size = 512
    while True:
        observation = env.reset()
        observation["observation_image"] = env.render(mode="rgb_array", height=100, width=100)

        for i in range(50):
            with torch.no_grad():
                # teacher 
                teacher_input_tensor = _preproc_inputs(observation["observation"].copy()[np.newaxis, :], observation["desired_goal"].copy()[np.newaxis, :])
                pi_teacher = teacher_network(teacher_input_tensor)

            student_input_tensor = _preproc_inputs_image(observation["observation_image"].copy()[np.newaxis, :])
            pi_student = student_network(student_input_tensor)


            if random.uniform(0,1) > 0.4:
                action = _select_actions(args, env_params, pi_student)
            else:
                action = _select_actions(args, env_params, pi_teacher)
            
            observation, _, _, info = env.step(action)
            observation["observation_image"] = env.render(mode="rgb_array", height=100, width=100)

            # store data in the buffer
            buffer.store_episode(observation)

            # update function
            if steps > batch_size:
                obs_state, des_goal, obs_img = buffer.sample(batch_size)
                with torch.no_grad():
                    teacher_input_tensor = _preproc_inputs(obs_state, des_goal)
                    pi_teacher = teacher_network(teacher_input_tensor)
                student_input_tensor = _preproc_inputs_image(obs_img)
                pi_student = student_network(student_input_tensor)
                
                # minimize loss
                student_loss = F.mse_loss(pi_student, pi_teacher)

                # step function
                student_optim.zero_grad()
                student_loss.backward()
                student_optim.step()
            
            steps += 1

        
        if episodes % 100 == 0:
            succ_rate = eval_agent(env, student_network, args)
            # save latest model
            # saving
            save_path = "saved_models/distill/image_only/"

            os.makedirs(save_path)

            torch.save({
                "actor_net": student_network.state_dict(),
                "meta_data": "image only, no normalization needed"
            }, save_path + "model.pt")


            print(f"Succ rate is {succ_rate}")

        episodes += 1


from arguments import get_args

args = get_args()
load_teacher_student(args)
