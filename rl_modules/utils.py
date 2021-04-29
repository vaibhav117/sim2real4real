import matplotlib.pyplot as plt
import time 
import cv2
import numpy as np
from mpi4py import MPI
from mujoco_py import MjRenderContextOffscreen
from mujoco_py.modder import TextureModder
import torch
import torch.nn.functional as F

def load_viewer(sim, device_id=MPI.COMM_WORLD.Get_rank()):
    viewer = MjRenderContextOffscreen(sim, device_id=device_id)
    viewer.cam.distance = 1.2 # this will be randomized baby: domain randomization FTW
    viewer.cam.azimuth = 180 # this will be randomized baby: domain Randomization FTW
    viewer.cam.elevation = -25 # this will be randomized baby: domain Randomization FTW
    viewer.cam.lookat[2] = 0.5 # IMPORTANT FOR ALIGNMENT IN SIM2REAL !!
    return viewer

def load_viewer_to_env(env, device_id=MPI.COMM_WORLD.Get_rank()):
    viewer = MjRenderContextOffscreen(env.sim, device_id=-1)
    viewer.cam.distance = 1.2 # this will be randomized baby: domain randomization FTW
    viewer.cam.azimuth = 180 # this will be randomized baby: domain Randomization FTW
    viewer.cam.elevation = -25 # this will be randomized baby: domain Randomization FTW
    viewer.cam.lookat[2] = 0.5 # IMPORTANT FOR ALIGNMENT IN SIM2REAL !!
    env.env._viewers["rgb_array"] = viewer
    return env

def normalize_depth(img):
    near = 0.021
    far = 2.14
    img = near / (1 - img * (1 - near / far))
    return img*15.5

def use_real_depths_and_crop(rgb, depth, vis=False):
    # TODO: add rgb normalization as well
    depth = normalize_depth(depth)
    depth = (depth - 0.021) / (2.14 - 0.021)
    
    depth = cv2.resize(depth[10:80, 10:90], (100,100))
    rgb = cv2.resize(rgb[10:80, 10:90, :], (100,100))
    
    if vis:
        from depth_tricks import create_point_cloud
        create_point_cloud(rgb, depth, vis=True)

    return rgb, depth[:, :, np.newaxis]

def use_real_depths_and_crop_np(rgb, depth, vis=False):
    # TODO: add rgb normalization as well
    depth = normalize_depth(depth)
    depth = (depth - 0.021) / (2.14 - 0.021)
    
    depth = depth[:, 10:80, 10:90]
    rgb = rgb[:, 10:80, 10:90]
    
    if vis:
        from depth_tricks import create_point_cloud
        create_point_cloud(rgb, depth, vis=True)

    return rgb, depth[:, :,:, np.newaxis]


def scripted_action(obs, picked_object):
    if not picked_object:
        if abs(obs["observation"][6]) > 0.001 or abs(obs["observation"][7] - 0.02) > 0.001:
            # print("X")
            action = np.asarray([obs["observation"][6], obs["observation"][7] - 0.02, 0]) * 50
            b = np.asarray((-1)).reshape((1))
            action = np.concatenate((action, b), axis=0)
            return action, False

        # if abs(obs["observation"][7] - 0.02) > 0.001:
        #     print("Y")
        #     y_act = obs["observation"][7] - 0.02  
        #     action = np.asarray([0, y_act, 0]) * 10
        #     b = np.asarray((-1)).reshape((1))
        #     action = np.concatenate((action, b), axis=0)
        #     return action, False
        
        if abs(obs["observation"][8]) > 0.001:
            # print("Z")
            action = np.asarray([0, 0, obs["observation"][8]]) * 50
            b = np.asarray((-1)).reshape((1))
            action = np.concatenate((action, b), axis=0)
            return action, False

    if abs(obs["observation"][7]) > 0.017:
        # print("close gripper")
        action = np.asarray([0, 0, 0, 1])
        return action, True

    # print("go towards goal")
    desired_goal = obs["desired_goal"]  # place of goal
    achieved_goal = obs["achieved_goal"] # actual goal

    action = desired_goal - achieved_goal
    scaler = 50
    action = np.asarray([action[0]*scaler, action[1]*scaler, action[2]*scaler, 1])

    return action, True




class Benchmark:

    def __init__(self):
        self.counter = {}
        self.total_time = {}
    
    def add(self, k, t):
        if k in self.counter:
            self.counter[k] += 1
            self.total_time[k] += t
        else:
            self.counter[k] = 1
            self.total_time[k] = t
    
    def plot(self):
        times = []
        keys = []
        freq = []
        for k, num_times in self.counter.items():
            total_time = self.total_time[k]
            times.append(total_time / num_times)
            keys.append(k)
            freq.append(num_times)

        #print(keys)
        fig, axs = plt.subplots(1,2)

        axs[0].plot(range(len(keys)), times)
        axs[0].set_xlabel("keys")
        axs[0].set_ylabel("function time")
        axs[1].plot(range(len(keys)), freq)
        axs[1].set_xlabel("keys")
        axs[1].set_ylabel("freq of functions")
        plt.xticks(range(len(keys)), keys, rotation=90)
        plt.savefig('speed_plot.png')
        plt.clf()
    
    def __call__(self, fn):
        # *args and **kwargs are to support positional and named arguments of fn
        def get_time(*args, **kwargs): 
            start = time.time() 
            output = fn(*args, **kwargs)
            time_taken = time.time() - start
            self.add(fn.__name__, time_taken)
            return output  # make sure that the decorator returns the output of fn
        return get_time 

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    print(layers)
    plt.show()

def timeit(fn): 
    # *args and **kwargs are to support positional and named arguments of fn
    def get_time(*args, **kwargs): 
        start = time.time() 
        output = fn(*args, **kwargs)
        time_taken = time.time() - start
        print(f"Time taken in {fn.__name__}: {time_taken:.7f}")

        return output  # make sure that the decorator returns the output of fn
    return get_time 


def show_video(img):
    cv2.imshow('frame', cv2.resize(img, (200,200)))
    cv2.waitKey(0)

def get_texture_modder(env):
    modder = TextureModder(env.sim)
    return modder

def randomize_textures(modder, env):
    for name in env.sim.model.geom_names:
        print(name)
        if name != 'object0': 
            modder.rand_all(name)

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


# pre_process the inputs
def _preproc_inputs_state(obs, args, is_np):
    obj = obs["obj"]
    if is_np:
        obs_state = obs["observation"][np.newaxis, :]
        g = obs["desired_goal"][np.newaxis, :]
    else:
        obs_state = obs["observation"].numpy()
        g = obs["desired_goal"].numpy()

    obs_norm = np.clip((obs_state - obj['o_mean'])/obj['o_std'], -args.clip_range, args.clip_range)
    g_norm = np.clip((g - obj['g_mean'])/obj['g_std'], -args.clip_range, args.clip_range)
    # concatenate the stuffs

    inputs = np.concatenate([obs_norm, g_norm], axis=1)
    if is_np:
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
    else:
        inputs = torch.tensor(inputs, dtype=torch.float32)

    return inputs

def _preproc_inputs_image_goal(obs, args, is_np):
    """
    One function to preprocess them all.

    """
    if is_np:
        obs_img = obs["rgb"][np.newaxis, :].copy()
        g = obs["desired_goal"][np.newaxis, :].copy()
        depth = obs["dep"][np.newaxis, :].copy()
        obj = obs["obj"]
        g = g[np.newaxis, :]
    else:
        obs_img = obs["rgb"].numpy().copy()
        g = obs["desired_goal"].numpy().copy()
        depth = obs["dep"].numpy().copy()
        obj = obs["obj"]

    
    if args.depth:
        # add depth observation
        if is_np:
            obs_img = obs_img.squeeze(0)
            obs_img = obs_img.astype(np.float32)
            print(obs_img.shape, depth.shape)
            # obs_img = obs_img / 255 # normalize image data between 0 and 1
            obs_img, depth = use_real_depths_and_crop(obs_img, depth)
            obs_img = np.concatenate((obs_img, depth), axis=2)
            obs_img = torch.tensor(obs_img, dtype=torch.float32).unsqueeze(0)
            obs_img = obs_img.permute(0, 3, 1, 2)
        else:
            obs_img = obs_img.astype(np.float32)
            # obs_img = obs_img / 255 # normalize image data between 0 and 1
            obs_im, depth = use_real_depths_and_crop_np(obs_img, depth)
            obs_im = torch.tensor(obs_im, dtype=torch.float32).permute(0, 3, 1, 2)
            depth = torch.tensor(depth, dtype=torch.float32).permute(0, 3, 1, 2)
            obs_im = F.interpolate(obs_im, size=(100,100))
            depth = F.interpolate(depth, (100,100))
            obs_img = torch.cat((obs_im, depth), axis=1)
    else:
        obs_img = torch.tensor(obs_img, dtype=torch.float32)
        obs_img = obs_img.permute(0, 3, 1, 2)
    

    g = np.clip((g - obj['g_mean'])/obj['g_std'], -args.clip_range, args.clip_range)
    g_norm = torch.tensor(g, dtype=torch.float32)
    state_based_input = _preproc_inputs_state(obs, args, is_np)
    if args.cuda:
        obs_img = obs_img.cuda(MPI.COMM_WORLD.Get_rank())
        g_norm = g_norm.cuda(MPI.COMM_WORLD.Get_rank())
        state_based_input = state_based_input.cuda(MPI.COMM_WORLD.Get_rank())
    
    return obs_img, g_norm, state_based_input


def display_state(obs):
    """
    Display state
    """
    cv2.imshow("frame", obs["rgb"])
    cv2.waitKey(1)


