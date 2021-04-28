import torch
from rl_modules.models import actor, critic, asym_goal_outside_image, sym_image, sym_image_critic
from arguments import get_args
import gym
from mujoco_py import load_model_from_path, MjSim, MjViewer, MjRenderContextOffscreen
import numpy as np
from mujoco_py.generated import const
from rl_modules.ddpg_agent import randomize_textures, randomize_camera
from mujoco_py.modder import TextureModder, MaterialModder, CameraModder, LightModder
from rl_modules.ddpg_agent import model_factory
from rl_modules.ddpg_agent import show_video
import cv2
import numpy as np
import matplotlib.pyplot as plt
from xarm_env.load_xarm7 import ReachXarm
from xarm_env.pick_and_place import PickAndPlaceXarm
# import open3d as o3d
from open3d import *
from depth_tricks import create_point_cloud, create_point_cloud2
import open3d as o3d
import numpy as np
import math
import datetime
from pathlib import Path
import os
import glob
from pcd_utils import get_real_pcd_from_recording, display_interactive_point_clouds
from rl_modules.utils import use_real_depths_and_crop
from verify_sim2real import go_through_all_possible_randomizations


def save_trajectory(rollouts, filename='trajs/traj1.pkl'):
    print(f" Saving rollout to {filename}")
    torch.save(rollouts, filename)

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
        viewer.cam.distance = 1.20
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -11
        viewer.cam.lookat[2] = 0.5
        env.env._viewers['rgb_array'] = viewer
        modder = TextureModder(env.sim)

        # pcds2 = go_through_all_possible_randomizations(env, viewer, modder)
        # env.reset()
        # env params
        env_params = get_env_params(env)
        env_params["model_path"] = paths[args.env_name]['xarm'][args.task] # TODO: fix bad practice
        env_params["load_saved"] = args.loadsaved
        env_params["depth"] = args.depth
        
        loaded_model, _, _, _ = model_factory(args.task, env_params)

        def load_plot(file_path):
            obj = torch.load(file_path, map_location=torch.device('cpu'))
            rew_asym = obj["reward_plots"]
            # two = len(obj['losses']) / len(obj['reward_plots'])
            plt.plot(np.arange(len(rew_asym)), rew_asym, color='red')
            # plt.plot(np.arange(len(obj['actor_losses'])), obj['actor_losses'], color='blue')
            plt.show()

        # loading best model for Fetch Reach
        model_path = paths[args.env_name]['xarm'][args.task] + '/model.pt'
        model_path = 'curr_bc_model.pt'

        load_plot(model_path)

        if True:
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

        def _preproc_inputs_image_goal(obs_img, g, depth=None):
            if args.depth:
                # add depth observation
                obs_img = obs_img.squeeze(0)
                obs_img = obs_img.astype(np.float32)
                # obs_img = obs_img / 255 # normalize image data between 0 and 1
                obs_img, depth = use_real_depths_and_crop(obs_img, depth)
                obs_img = np.concatenate((obs_img, depth), axis=2)
                obs_img = torch.tensor(obs_img, dtype=torch.float32).unsqueeze(0)
                obs_img = obs_img.permute(0, 3, 1, 2)
            else:
                obs_img = torch.tensor(obs_img, dtype=torch.float32)
                obs_img = obs_img.permute(0, 3, 1, 2)
            
            print(f"Mean {obj['g_mean']} | STD: {obj['g_std']} | clip range {-args.clip_range}:{args.clip_range}")
            print(f"Goal now {g}")

            g = np.clip((g - obj['g_mean'])/obj['g_std'], -args.clip_range, args.clip_range)
            print(f"Goal after norm {g}")
            g_norm = torch.tensor(g, dtype=torch.float32)
            # g_norm = torch.zeros((1, 3))
            if args.cuda:
                obs_img = obs_img.cuda(MPI.COMM_WORLD.Get_rank())
                g_norm = g_norm.cuda(MPI.COMM_WORLD.Get_rank())
            print(f"Goal is {g_norm} | unnormalised {g}")
            return obs_img, g_norm
        
        def _prepoc_image(obs_img):
            obs_img = torch.tensor(obs_img, dtype=torch.float32)
            obs_img = obs_img.permute(0, 3, 1, 2)
            if args.cuda:
                obs_img = obs_img.cuda(MPI.COMM_WORLD.Get_rank())
            return obs_img

        # pre_process the inputs
        def _preproc_inputs_state(obs, g):
            print(obs.shape, obj['o_mean'].shape)
            obs_norm = np.clip((obs - obj['o_mean'])/obj['o_std'], -args.clip_range, args.clip_range).reshape(1,-1)
            g_norm = np.clip((g - obj['g_mean'])/obj['g_std'], -args.clip_range, args.clip_range)
            # concatenate the stuffs
            print(obs_norm.shape, g_norm.shape)
            inputs = np.concatenate([obs_norm, g_norm], axis=1)
            inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
            return inputs

        def get_policy(obs_img, g, obs=None, depth=None):
            if args.task == "sym_state":
                inputs = _preproc_inputs_state(obs, g)
                pi = loaded_model(inputs)
                return pi
            if args.task == "asym_goal_outside_image":
                o_tensor, g_tensor = _preproc_inputs_image_goal(obs_img, g, depth)
                # g_tensor = torch.tensor(np.asarray([0.2, 0.2, 0.2])).view(1, -1).to(torch.float32)
                pi = loaded_model(o_tensor, g_tensor)
                return pi
            if args.task == "asym_goal_in_image":
                pi = loaded_model(_prepoc_image(obs_img))
                return pi
            if args.task == "sym_image":
                o_tensor, _ = _preproc_inputs_image_goal(obs_img, g)
                pi = loaded_model(o_tensor)
                return pi
        
        def create_folder_and_save(obj, folder_name='rollout_records'):            
            Path(folder_name).mkdir(parents=True, exist_ok=True)
            timestamp =  str(datetime.datetime.now())
            path_name = os.path.join(folder_name, timestamp)

            torch.save(obj, path_name)
            print(f"Trajectory saved to {path_name}")

        for _ in range(args.n_test_rollouts):
            per_success_rate = []
            observation = env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            rollout = []
            all_info = []
            pcds = []
            
            # if args.randomize:
            #     randomize_textures(modder, env.sim)
            #     # randomize_camera(viewer)
            
            max_steps = env._max_episode_steps
            max_steps = 100
            hard_coded_goal = np.asarray([1.63, 0.51, 0.33])
            for _ in range(max_steps):
                viewer.cam.distance = 1.20
                viewer.cam.azimuth = 180
                viewer.cam.elevation = -11

                if args.randomize:
                    # randomize_camera(viewer)
                    randomize_textures(modder, env.sim)
                # env.render()
                
                obs_img, depth_image = env.render(mode="rgb_array", height=100, width=100, depth=True)
                # print(obs_img.dtype, depth_image.dtype, obs_img.mean())
                # viewer.render(100, 100)
                # data, dep = viewer.read_pixels(100, 100, depth=True)
                # obs_img, depth_image = data[::-1, :, :], dep[::-1, :]
                save_obs_img, save_depth_image = use_real_depths_and_crop(obs_img, depth_image)
                

                pcd = create_point_cloud(save_obs_img, save_depth_image, fovy=45)
                pcds.append(("none", pcd))
                
                # env.render()
                # obs_img = cv2.cvtColor(obs_img, cv2.COLOR_BGR2RGB)
                # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(dep_im, alpha=0.03), cv2.COLORMAP_JET
                if args.depth:
                    # create_point_cloud(env, dep_img=depth_image, col_img=obs_img)
                    # g = hard_coded_goal
                    pi = get_policy(obs_img.copy()[np.newaxis, :], g[np.newaxis, :], depth=depth_image[:, :, np.newaxis])
                    actions = pi.detach().cpu().numpy().squeeze()
                    print(f"Actions is {actions}")
                else:
                    with torch.no_grad():
                        if args.task != 'sym_state':
                            pi = get_policy(obs_img.copy()[np.newaxis, :], g[np.newaxis, :])
                        else:
                            pi = get_policy(obs_img=obs_img.copy()[np.newaxis, :], g=g[np.newaxis, :], obs=observation["observation"])
                        actions = pi.detach().cpu().numpy().squeeze()
                observation_new, _, _, info = env.step(actions)


                rollout.append({
                    'obs_img': save_obs_img,
                    'depth_img': save_depth_image,
                    'actions': actions,
                })

                show_video(save_obs_img)

                obs = observation_new['observation']
               
                g = observation_new['desired_goal']
                observation = observation_new
                per_success_rate.append(info['is_success'])

            # hide under a flag
            if args.record:
                create_folder_and_save({'traj': rollout, 'goal': g})

            # from pcd_utils import display_interactive_point_cloud

            # display_interactive_point_cloud(pcds)
           
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
            'xarm': {
                'asym_goal_outside_image': './sym_server_weights/saved_models/asym_goal_outside_image/FetchPickAndPlace-v1',
                'sym_state': './sym_server_weights/saved_models/sym_state/FetchPickAndPlace-v1',
                'asym_goal_outside_image_distill': './sym_server_weights/saved_models/asym_goal_outside_image_distill/FetchPickAndPlace-v1',
            }
        }
    }
    args = get_args()
    args.env_name = 'FetchPickAndPlace-v1'
    args.task = 'asym_goal_outside_image'
    # args.task = 'sym_state'
    # env = gym.make(args.env_name)
    if args.env_name =='FetchReach-v1':
        env = ReachXarm(xml_path='./assets/fetch/reach_xarm_with_gripper.xml')
    else:
        env = PickAndPlaceXarm(xml_path='./assets/fetch/pick_and_place_xarm.xml')
    _eval_agent(args, paths, env)
  
