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

def get_viewer(env):
    return env.env._viewers["rgb_array"]

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

    x = abs(obs["observation"][6])
    y = abs(obs["observation"][7] - 0.04)
    z = abs(obs["observation"][8])
    
    if picked_object and (x > 0.1 or y > 0.1 or z > 0.1):
        picked_object = False


    if not picked_object: #3 # TODO: make a function that does this robustly
        # if robot is above the object then first align
        if x > 0.001 or y > 0.001:
            
            action = np.asarray([obs["observation"][6], obs["observation"][7] - 0.04, 0]) * 50
            b = np.asarray((-1)).reshape((1))
            action = np.concatenate((action, b), axis=0)
            return action, False
        
        # if robot is aligned, then go down
        if abs(obs["observation"][8]) > 0.001:
            # print("three")
            # print("Z")
            
            action = np.asarray([0, 0, obs["observation"][8]]) * 50
            b = np.asarray((-1)).reshape((1))
            action = np.concatenate((action, b), axis=0)
            return action, False

    left_gripper = obs['observation'][:3]
    right_gripper = obs['observation'][-3:]
    a = abs(right_gripper[1] - left_gripper[1])


    if abs(obs["observation"][7]) > 0.017 and a > 0.050:
        # print(f"close gripper {obs['observation'][-5:]}")
        # print(obs["observation"][])
        action = np.asarray([0, 0, 0, 1])
        return action, True

    # print("go towards goal")
    desired_goal = obs["desired_goal"]  # place of goal
    achieved_goal = obs["achieved_goal"] # actual goal

    action = desired_goal - achieved_goal
    scaler = 50
    action = np.asarray([action[0]*scaler, action[1]*scaler, action[2]*scaler, 1])

    return action, True


def scripted_action_new(obs, picked_object):
    x = obs["observation"][7]
    y = obs["observation"][8]
    z = obs["observation"][9] - 0.005   # The offset makes the arm grip the object better, as it aim to grab the point a little below the mid point
    
    if picked_object and (abs(x) > 0.1 or abs(y) > 0.1 or abs(z) > 0.1):
        picked_object = False

    if not picked_object: #3 # TODO: make a function that does this robustly
        # if robot is above the object then first align
        if abs(x) > 0.001 or abs(y) > 0.001:
            action = np.asarray([x, y, 0]) * 50
            b = np.asarray((1)).reshape((1))
            grip_changed = np.asarray((0)).reshape((1))
            action = np.concatenate((action, b, grip_changed), axis=0)
            return action, False
        
        # if robot is aligned, then go down
        if abs(z) > 0.001:            
            action = np.asarray([0, 0, z]) * 50
            b = np.asarray((-1)).reshape((1))
            grip_changed = np.asarray((0)).reshape((1))
            action = np.concatenate((action, b, grip_changed), axis=0)
            return action, False

    left_gripper = obs['observation'][:3]
    right_gripper = obs['observation'][-3:]
    a = abs(right_gripper[1] - left_gripper[1])
    if abs(z) > 0.0007 and a > 0.050:
        action = np.asarray([0, 0, 0, 1, 1])
        return action, True
    
    # print("go towards goal")
    desired_goal = obs["desired_goal"]  # place of goal
    achieved_goal = obs["achieved_goal"] # actual goal

    action = desired_goal - achieved_goal
    scaler = 10
    action = np.asarray([action[0]*scaler, action[1]*scaler, action[2]*scaler, 1, 0])
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


def out_of_bounds(obs):
    '''
    Return true if object is out of bounds from robot reach
    '''
    #if x pos is < 0.9  < 1.8 then bad, 0.05 < y < 1.3
    object_pos = obs['observation'][3:6]
    x = object_pos[0]
    y = object_pos[1]

    gripper_x = obs['observation'][0]
    gripper_y = obs['observation'][1]
    gripper_z = obs['observation'][2]

    # print(gripper_x, gripper_y, gripper_z)

    if x < 0.9 or x >= 1.75:
        return True
    
    if y < 0.05 or y > 1.25:
        return True

    if gripper_x < 0.9 or gripper_x >= 1.75:
        return True
    
    if gripper_y < 0.05 or gripper_y > 1.25:
        return True
    
    if gripper_z > 0.5:
        return True 
    
    return False

# pre_process the inputs
def _preproc_inputs_state(obs, args, is_np):
    obj = obs["obj"]
    if is_np:
        obs_state = obs["observation"][np.newaxis, :]
        g = obs["desired_goal"][np.newaxis, :]
    else:
        obs_state = obs["observation"].numpy()
        g = obs["desired_goal"].numpy()

    # obs_norm = np.clip((obs_state - obj['o_mean'])/obj['o_std'], -args.clip_range, args.clip_range)
    g_norm = np.clip((g - obj['g_mean'])/obj['g_std'], -args.clip_range, args.clip_range)
    # concatenate the stuffs

    obs_norm = obs_state
    #g_norm = g

    inputs = np.concatenate([obs_norm, g_norm], axis=1)
    if is_np:
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
    else:
        inputs = torch.tensor(inputs, dtype=torch.float32)

    return inputs

def is_np_or_not(obs, is_np):
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

    return obs_img, g, depth, obj

def _preproc_image(obs, args, is_np):
    obs_img, _, depth, _ = is_np_or_not(obs, is_np)

    if args.depth:
        obs_img = obs_img.squeeze(0)
        obs_img = obs_img.astype(np.float32)
        # obs_img = obs_img / 255 # normalize image data between 0 and 1
        obs_img, depth = use_real_depths_and_crop(obs_img, depth)
        obs_img = np.concatenate((obs_img, depth), axis=2)
        obs_img = torch.tensor(obs_img, dtype=torch.float32).unsqueeze(0)
        obs_img = obs_img.permute(0, 3, 1, 2)

    if args.cuda:
        obs_img = obs_img.cuda(MPI.COMM_WORLD.Get_rank())

    return obs_img


def _preproc_inputs_image_goal(obs, args, is_np):
    """
    One function to preprocess them all.

    """
    obs_img, g, depth, obj = is_np_or_not(obs, is_np)
    
    if args.depth:
        # add depth observation
        if is_np:
            obs_img = obs_img.squeeze(0)
            obs_img = obs_img.astype(np.float32)
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


