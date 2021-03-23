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

def get_real_pcd(filepath='xarm_env/obs_dump.pkl'):
    rolls = torch.load(filepath)

    pcds1 = []
    pcd2s = []

    for r in rolls:
        sim_img_obs = r['sim_img_obs']
        sim_dep_obs = r['sim_dep_obs']
        robo_dep_obs = r['depth_obs'].astype(np.float32)
        robo_img_obs = r['img_obs']
        orig_depth = r['orig_robo_depth'].astype(np.float32) / 1000
        
        orig_sim_dep = sim_dep_obs.copy()
        orig_real_dep = orig_depth.copy()

        # show_video(orig_depth)
        # show_video(robo_img_obs)

        pcd1 = create_point_cloud2(sim_dep_obs, sim_img_obs)
        pcd2 = create_point_cloud2(orig_depth, robo_img_obs)

        return pcd2

        pcds1.append(pcd1)
        pcd2s.append(pcd2)

    vis.add_geometry(pcd2s[0])


def get_most_recent_file(folder='./real_recordings/'):
    list_of_files = glob.glob(os.path.join(folder, '*')) # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    print(latest_file)
    return latest_file


def retrieve_traj():
    filepath = get_most_recent_file()
    obj = torch.load(filepath)
    return obj

def get_real_pcd_from_recording():

    obj = retrieve_traj()
    goal = obj["goal"]
    pcds = []
    for obs in obj['traj']:
        depth_obs = obs["depth_obs"]
        img_obs = obs["img_obs"]
        action = obs["action"]

        pcd = create_point_cloud(img_obs, depth_obs)
        return pcd
        pcds.append({'pcd': pcd, 'goal': goal, 'action': action, 'depth_obs': depth_obs, 'img_obs': img_obs})

    visualize(pcds)


def get_real_depth(filepath='xarm_env/obs_dump.pkl'):
    rolls = torch.load(filepath)

    pcds1 = []
    pcd2s = []

    for r in rolls:
        sim_img_obs = r['sim_img_obs']
        sim_dep_obs = r['sim_dep_obs']
        robo_dep_obs = r['depth_obs'].astype(np.float32)
        robo_img_obs = r['img_obs']
        orig_depth = r['orig_robo_depth'].astype(np.float32) / 1000
        
        orig_sim_dep = sim_dep_obs.copy()
        orig_real_dep = orig_depth.copy()

        # show_video(orig_depth)
        # show_video(robo_img_obs)

        pcd1 = create_point_cloud2(sim_dep_obs, sim_img_obs)
        pcd2 = create_point_cloud2(orig_depth, robo_img_obs)

        return orig_depth, robo_img_obs

        pcds1.append(pcd1)
        pcd2s.append(pcd2)

    vis.add_geometry(pcd2s[0])



def go_through_all_possible_randomizations(env, viewer, modder, test_elevation=True, show_distances=False, test_angle=False):
    # viewer.cam.distance = 1.2 + np.random.uniform(-0.05, 0.5)
    # viewer.cam.azimuth = 180 + np.random.uniform(-2, 2)
    # viewer.cam.elevation = -25 + np.random.uniform(1, 3)

    def normalize_depth(img):
        near = 0.021
        far = 2.14
        img = near / (1 - img * (1 - near / far))
        return img*15.5

    width = 100
    height = 100
    env.reset()
    # env.render()

    randomize_textures(modder, env.sim)

    distances = np.arange(1.15, 2.9, 0.05)
    pcds = []
    if show_distances:
        for d in distances:
            print(d)
            viewer.cam.distance = d
            viewer.render(width, height)
            data, dep = viewer.read_pixels(width, height, depth=True)
            obs_img, depth_image = data[::-1, :, :], dep[::-1, :]
            depth_image = normalize_depth(depth_image)
            # show_video(obs_img)
            # show_video(depth_image)
            pcds.append(create_point_cloud(obs_img, depth_image))
        
        display_interactive_point_cloud(pcds)


    # test elevation
    elevation = -25 + np.random.uniform(1, 3)
    elevations = np.arange(-11, -9.5, 0.5)
    distances = np.arange(1.0, 1.25, 0.05)
    azimuths = np.arange(178, 184, 1.0)
    viewer.cam.lookat[2] = 0.5
    pcds = []
    viewer.cam.distance = 0.8
    if test_elevation:
        for ele in elevations:
            for a in azimuths:
                for d in distances:
                    viewer.cam.distance = 1.20
                    viewer.cam.elevation = -11
                    viewer.cam.azimuth = 180
                    
                    viewer.render(width, height)
                    data, dep = viewer.read_pixels(width, height, depth=True)
                    obs_img, depth_image = data[::-1, :, :], dep[::-1, :]
                    
                    depth_image = normalize_depth(depth_image)
                    depth_image = cv2.resize(depth_image[10:80, 10:90], (100,100))
                    obs_img = cv2.resize(obs_img[10:80, 10:90, :], (100,100))
                    real_depth, real_img = get_real_depth()
                    # fig, axs = plt.subplots(2, 2)
                    # print(depth_image.shape)
                    # axs[0, 0].imshow(real_depth)
                    # axs[0, 1].imshow(depth_image)
                    # axs[1, 0].imshow(real_img)
                    # axs[1, 1].imshow(obs_img)
                    # plt.show()
                    # # exit()

                    real_depth_grey = cv2.cvtColor(real_depth, cv2.COLOR_GRAY2BGR)
                    depth_image_grey = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)
                    obs_img_cv = cv2.cvtColor(obs_img, cv2.COLOR_RGB2BGR)
                    real_img_cv = cv2.cvtColor(real_img, cv2.COLOR_RGB2BGR)
                    mini = real_depth.min()
                    maxi = real_depth.max()
                    depth_image_grey = (depth_image - mini) / (maxi - mini)
                    real_depth_grey = (real_depth - mini) / (maxi - mini)
                    numpy_images = np.hstack((obs_img_cv, real_img_cv))
                    numpy_horizontal = np.hstack((real_depth_grey, depth_image_grey))
                    # cv2.imshow('all depths', numpy_horizontal)
                    # cv2.imshow('all images', numpy_images)
                    # cv2.waitKey(1)
                    description = f"Distance: {d}, Elevation: {ele}, Azimuth: {a}"
                    print(description)
                    pcds.append((description, create_point_cloud(obs_img, depth_image, fovy=45)))
        display_interactive_point_cloud(pcds)


    if test_angle:
        pcds = []
        viewer.cam.elevation = -10
        viewer.cam.distance = 0.8
        angles = np.arange(0, 1, 0.05)
        for a in angles:
            viewer.cam.lookat[2] = a
            viewer.render(width, height)
            data, dep = viewer.read_pixels(width, height, depth=True)
            obs_img, depth_image = data[::-1, :, :], dep[::-1, :]
            depth_image = normalize_depth(depth_image)
            pcds.append(create_point_cloud(obs_img, depth_image))
        display_interactive_point_cloud(pcds)
    

def normalize_depthz(img):
    near = 0.021
    far = 2.14
    img = near / (1 - img * (1 - near / far))
    return img*15.5

def display_interactive_point_cloud(pcds):
    # hide under a flag

    print("starting int")
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    i = 1
    # real_pcd = get_real_pcd()
    real_pcd = get_real_pcd_from_recording()

    vis.add_geometry(pcds[0][1])


    def update_pcd(vis):
        nonlocal i, pcds, real_pcd
        # global pcds
        i = (i+1) % len(pcds)
        vis.clear_geometries()
        vis.add_geometry(pcds[i][1])
        print(pcds[i][0])
        vis.add_geometry(real_pcd)

    vis.register_key_callback(ord("K"), update_pcd)
    vis.run()
    vis.destroy_window()

    print("ending int")


def save_trajectory(rollouts, filename='trajs/traj1.pkl'):
    print(f" Saving rollout to {filename}")
    torch.save(rollouts, filename)

def depth2pcd(depth):
    def remap(x, in_range_l, in_range_r, out_range_l, out_range_r):
        return (x - in_range_l) / (in_range_r - in_range_l) * (out_range_r - out_range_l) + out_range_l
    # depth = remap(depth, depth.min(), depth.max(), 0, 1)
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

        env.env._viewers['rgb_array'] = viewer
        modder = TextureModder(env.sim)

        go_through_all_possible_randomizations(env, viewer, modder)

        # env params
        env_params = get_env_params(env)
        env_params["model_path"] = paths[args.env_name]['xarm'][args.task] # TODO: fix bad practice
        env_params["load_saved"] = args.loadsaved
        env_params["depth"] = args.depth
        
        loaded_model, _, _, _ = model_factory(args.task, env_params)
        # obj = torch.load(env_params["model_path"]+ '/model.pt', map_location=torch.device('cpu'))
        # plt.plot(obj["reward_plots"])
        # plt.show()

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

        def use_real_depths_and_crop(rgb, depth):
            def normalize_depth(img):
                near = 0.021
                far = 2.14
                img = near / (1 - img * (1 - near / far))
                return img*15.5
            depth = normalize_depth(depth)
            depth = cv2.resize(depth[10:80, 10:90], (100,100))
            rgb = cv2.resize(rgb[10:80, 10:90, :], (100,100))

            # from depth_tricks import create_point_cloud
            # create_point_cloud(rgb, depth, vis=True)

            return rgb, depth[:, :, np.newaxis]

        def _preproc_inputs_image_goal(obs_img, g, depth=None):
            if args.depth:
                # add depth observation
                obs_img = obs_img.squeeze(0)
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

        def get_policy(obs_img, g, depth=None):
            if args.task == "sym_state":
                raise NotImplementedError
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
            
            if args.randomize:
                randomize_textures(modder, env.sim)
                # randomize_camera(viewer)
            max_steps = env._max_episode_steps
            max_steps = 10
            hard_coded_goal = np.asarray([1.63, 0.51, 0.33])
            for _ in range(max_steps):
                if args.randomize:
                    # randomize_camera(viewer)
                    randomize_textures(modder, env.sim)
                obs_img, depth_image = env.render(mode="rgb_array", height=100, width=100, depth=True)

                # show_video(obs_img)
                save_depth_image = normalize_depthz(depth_image)
                save_obs_img, save_depth_image = use_real_depths_and_crop(obs_img, depth_image)
                pcd = create_point_cloud(save_obs_img, save_depth_image)
                pcds.append(pcd)
                
                # env.render()
                # obs_img = cv2.cvtColor(obs_img, cv2.COLOR_BGR2RGB)
                # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(dep_im, alpha=0.03), cv2.COLORMAP_JET
                if args.depth:
                    # create_point_cloud(env, dep_img=depth_image, col_img=obs_img)
                    g = hard_coded_goal
                    pi = get_policy(obs_img.copy()[np.newaxis, :], g[np.newaxis, :], depth=depth_image[:, :, np.newaxis])
                    actions = pi.detach().cpu().numpy().squeeze()
                    print(f"Actions is {actions}")
                else:
                    with torch.no_grad():
                        pi = get_policy(obs_img.copy()[np.newaxis, :], g[np.newaxis, :])
                        actions = pi.detach().cpu().numpy().squeeze()
                observation_new, _, _, info = env.step(actions)


                rollout.append({
                    'obs_img': save_obs_img,
                    'depth_img': save_depth_image,
                    'actions': actions,
                })

                obs = observation_new['observation']
                obs_img, depth_image = env.render(mode="rgb_array", height=100, width=100, depth=True)
                g = observation_new['desired_goal']
                per_success_rate.append(info['is_success'])

            # hide under a flag
            if args.record:
                create_folder_and_save({'traj': rollout, 'goal': hard_coded_goal})

            vis = o3d.visualization.VisualizerWithKeyCallback()
            vis.create_window()
            i = 1
            vis.add_geometry(pcds[0])


            def update_pcd(vis):
                nonlocal i, pcds
                # global pcds
                i = (i+1) % len(pcds)
                vis.clear_geometries()
                vis.add_geometry(pcds[i])

            vis.register_key_callback(ord("K"), update_pcd)
            vis.run()
            vis.destroy_window()

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
    args.task = 'asym_goal_outside_image'
    # args.task = 'sym_state'
    # env = gym.make(args.env_name)
    env = ReachXarm(xml_path='./assets/fetch/reach_xarm_with_gripper.xml')
    _eval_agent(args, paths, env)
  
