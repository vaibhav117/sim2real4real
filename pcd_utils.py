import torch
import os
from depth_tricks import create_point_cloud
import numpy as np 
import open3d as o3d
import glob

def get_most_recent_file(folder='./real_recordings/'):
    list_of_files = glob.glob(os.path.join(folder, '*')) # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    print(latest_file)
    return latest_file


def retrieve_traj():
    filepath = get_most_recent_file()
    obj = torch.load(filepath)
    return obj


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


def get_real_pcd_from_recording(all=False):

    obj = retrieve_traj()
    goal = obj["goal"]
    pcds = []
    for obs in obj['traj']:
        depth_obs = obs["depth_obs"]
        img_obs = obs["img_obs"]
        action = obs["action"]
        depth_obs = (depth_obs - 0.021) / (2.14 - 0.021)
        pcd = create_point_cloud(img_obs, depth_obs)
        if not all:
            return pcd
        pcds.append({'pcd': pcd, 'goal': goal, 'action': action, 'depth_obs': depth_obs, 'img_obs': img_obs})

    # visualize(pcds)
    return pcds


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

        pcd1 = create_point_cloud(sim_img_obs, sim_dep_obs)
        pcd2 = create_point_cloud(robo_img_obs, orig_depth)

        return orig_depth, robo_img_obs

        pcds1.append(pcd1)
        pcd2s.append(pcd2)

    vis.add_geometry(pcd2s[0])


def display_interactive_point_clouds(pcds1, pcds2):
    # hide under a flag

    print("starting int")
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    i = 1
    # real_pcd = get_real_pcd()
    real_pcd = get_real_pcd_from_recording()

    vis.add_geometry(pcds1[0][1])
    vis.add_geometry(pcds2[0][1])


    def update_pcd(vis):
        nonlocal i, pcds1, pcds2, real_pcd
        # global pcds
        i = (i+1) % len(pcds1)
        vis.clear_geometries()
        vis.add_geometry(pcds1[i][1])
        vis.add_geometry(pcds2[i][1])
        print(pcds2[i][0])
        vis.add_geometry(real_pcd)

    vis.register_key_callback(ord("K"), update_pcd)
    vis.run()
    vis.destroy_window()

    print("ending int")


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
        # vis.add_geometry(real_pcd)

    vis.register_key_callback(ord("K"), update_pcd)
    vis.run()
    vis.destroy_window()

    print("ending int")
