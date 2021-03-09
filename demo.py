import torch
from rl_modules.models import actor, critic, asym_goal_outside_image, sym_image, sym_image_critic
from arguments import get_args
import gym
from mujoco_py import load_model_from_path, MjSim, MjViewer, MjRenderContextOffscreen
import numpy as np
from mujoco_py.generated import const
from rl_modules.ddpg_agent import randomize_textures
from mujoco_py.modder import TextureModder, MaterialModder, CameraModder, LightModder
from rl_modules.ddpg_agent import model_factory
from rl_modules.ddpg_agent import show_video
import cv2
import numpy as np
import matplotlib.pyplot as plt
from xarm_env.load_xarm7 import ReachXarm
# import open3d as o3d
from open3d import *
from depth_tricks import create_point_cloud

import numpy as np
import math



def save_trajectory(rollouts, filename='trajs/traj1.pkl'):
    print(f" Saving rollout to {filename}")
    torch.save(rollouts, filename)

def depth2pcd(depth):
    def remap(x, in_range_l, in_range_r, out_range_l, out_range_r):
        return (x - in_range_l) / (in_range_r - in_range_l) * (out_range_r - out_range_l) + out_range_l
    # depth = remap(depth, depth.min(), depth.max(), 0, 1)
    # print(depth)
    scalingFactor = 1
    fovy = 60
    aspect = depth.shape[1] / depth.shape[0]
    # fovx = 2 * math.atan(math.tan(fovy * 0.5 * math.pi / 360) * aspect)
    width = depth.shape[1]
    height = depth.shape[0]
    fovx = 2 * math.atan(width * 0.5 / (height * 0.5 / math.tan(fovy * math.pi / 360 / 2))) / math.pi * 360
    fx = width / 2 / (math.tan(fovx * math.pi / 360 / 2))
    fy = height / 2 / (math.tan(fovy * math.pi / 360 / 2))
    points = []

    for v in range(0, height, 10):
        for u in range(0, width, 10):
            Z = depth[v][u] / scalingFactor
            if Z == 0:
                continue
            X = (u - width / 2) * Z / fx
            Y = (v - height / 2) * Z / fy
            points.append([X, Y, Z])
            
    return np.array(points)

#video=cv2.VideoWriter('video.mp4',-1,1,(100,100))
def randomize_camera(viewer):
    viewer.cam.distance = 1.2 + np.random.uniform(-0.05, 0.5)
    viewer.cam.azimuth = 180 + np.random.uniform(-2, 2)
    viewer.cam.elevation = -25 + np.random.uniform(1, 3)


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


def _eval_agent(args, paths, env, image_based=True, cuda=False):

        # load model
        sim = env.sim
        viewer = MjRenderContextOffscreen(sim)
        viewer.cam.distance = 1.2
        # viewer.cam.distance = 1.5
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -25
        # viewer.cam.elevation = 0
        # viewer.cam.lookat[0] = 1.5
        # viewer.cam.lookat[1] = 0.75
        # viewer.cam.lookat[2] = 1.0
        # 1 0.75 0.45
        env.env._viewers['rgb_array'] = viewer


        # env params
        env_params = get_env_params(env)
        env_params["model_path"] = paths[args.env_name]['xarm'][args.task] # TODO: fix bad practice
        env_params["load_saved"] = args.loadsaved
        env_params["depth"] = False
        
        loaded_model, _, _, _ = model_factory(args.task, env_params)

        model_path = paths[args.env_name]['xarm'][args.task] + '/model.pt'
        if args.task != 'sym_state':
            obj = torch.load(model_path, map_location=lambda storage, loc: storage)
            loaded_model.load_state_dict(obj['actor_net'])
        else:
            o_mean, o_std, g_mean, g_std, actor_state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
            loaded_model.load_state_dict(actor_state_dict)
            obj = {}
            obj["g_mean"] = g_mean
            obj["g_std"] = g_std

        if cuda:
            loaded_model.cuda()

        total_success_rate = []
        rollouts = []

        def _preproc_inputs_image_goal(obs_img, g):
            obs_img = torch.tensor(obs_img, dtype=torch.float32)
            obs_img = obs_img.permute(0, 3, 1, 2)
            g = np.clip((g - obj['g_mean'])/obj['g_std'], -args.clip_range, args.clip_range)
            g_norm = torch.tensor( g, dtype=torch.float32)
            if args.cuda:
                obs_img = obs_img.cuda(MPI.COMM_WORLD.Get_rank())
                g_norm = g_norm.cuda(MPI.COMM_WORLD.Get_rank())
            return obs_img, g_norm
        
        def _prepoc_image(obs_img):
            obs_img = torch.tensor(obs_img, dtype=torch.float32)
            obs_img = obs_img.permute(0, 3, 1, 2)
            if args.cuda:
                obs_img = obs_img.cuda(MPI.COMM_WORLD.Get_rank())
            return obs_img

        def get_policy(obs_img, g):
            if args.task == "sym_state":
                raise NotImplementedError
            if args.task == "asym_goal_outside_image":
                o_tensor, g_tensor = _preproc_inputs_image_goal(obs_img, g)
                pi = loaded_model(o_tensor, g_tensor)
                print(f"Goal tensor: {g_tensor} Actions: {pi}")
                return pi
            if args.task == "asym_goal_in_image":
                pi = loaded_model(_prepoc_image(obs_img))
                return pi
            if args.task == "sym_image":
                o_tensor, _ = _preproc_inputs_image_goal(obs_img, g)
                pi = loaded_model(o_tensor)
                return pi

        for _ in range(args.n_test_rollouts):
            per_success_rate = []
            observation = env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            rollout = []
            modder = TextureModder(env.sim)
            if args.randomize:
                randomize_textures(modder, env.sim)
                randomize_camera(viewer)

            for _ in range(env._max_episode_steps):
                if args.randomize:
                    # randomize_camera(viewer)
                    randomize_textures(modder, env.sim)
                obs_img, depth_image = env.render(mode="rgb_array", height=100, width=100, depth=True)

                show_video(obs_img)
                # env.render()
                # obs_img = cv2.cvtColor(obs_img, cv2.COLOR_BGR2RGB)
                # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(dep_im, alpha=0.03), cv2.COLORMAP_JET
                if args.depth:
                    create_point_cloud(env, dep_img=depth_image, col_img=obs_img)
                
                with torch.no_grad():
                    pi = get_policy(obs_img.copy()[np.newaxis, :], g[np.newaxis, :])
                    actions = pi.detach().cpu().numpy().squeeze()
                # print(f"Actions {actions}")
                observation_new, _, _, info = env.step(actions)


                rollout.append({
                    'obs_img': obs_img,
                    'depth_img': depth_image,
                    'actions': actions
                })

                obs = observation_new['observation']
                obs_img, depth_image= env.render(mode="rgb_array", height=100, width=100, depth=True)
                g = observation_new['desired_goal']
                per_success_rate.append(info['is_success'])

            total_success_rate.append(per_success_rate)
            rollouts.append(rollout)
            
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])

        if args.record:
            # save trajectory
            save_trajectory(rollouts)
        print(local_success_rate)



# process the inputs
def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs



if __name__ == '__main__':
    paths = {
        'FetchReach-v1': {
            'old_robo': {
                'sym_state': './all_weigths/FetchReach-v1/',
                'asym_goal_outside_image': './randomized_server_weights/asym_goal_outside_image/FetchReach-v1',
                'asym_goal_in_image': 'sym_server_weights/distill/',
                'sym_image': ''
            },
            'xarm': {
                'asym_goal_in_image': './sym_server_weights/saved_models/asym_goal_in_image/FetchReach-v1',
                'asym_goal_outside_image': './sym_server_weights/saved_models/asym_goal_outside_image/FetchReach-v1'
            }
        },
        'FetchPush-v1': {
            'sym_state': '',
            'asym_goal_in_image': 'sym_server_weights/saved_models/distill/image_only/',
            'asym_goal_outside_image': './sym_server_weights/asym_goal_outside_image_distill/FetchPush-v1/',
        },
        'FetchSlide-v1': {

        },
        'FetchPickAndPlace-v1': {

        }
    }
    args = get_args()
    args.env_name = 'FetchReach-v1'
    args.task = 'asym_goal_in_image'
    # args.task = 'sym_state'
    # env = gym.make(args.env_name)
    env = ReachXarm(xml_path='./assets/fetch/reach_xarm_with_gripper.xml')
    _eval_agent(args, paths, env)
  
