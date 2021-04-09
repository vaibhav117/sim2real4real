from rl_modules.utils import use_real_depths_and_crop
from rl_modules.ddpg_agent import randomize_textures, randomize_camera
import numpy as np
import cv2
from pcd_utils import get_real_depth, display_interactive_point_cloud, get_real_pcd_from_recording, get_real_image_from_recording
from depth_tricks import create_point_cloud
import torch
from xarm_env.pick_and_place import PickAndPlaceXarm
from rl_modules.utils import load_viewer, get_texture_modder

def go_through_all_possible_randomizations(env, viewer, modder, test_elevation=True, show_distances=False, test_angle=False):
    # viewer.cam.distance = 1.2 + np.random.uniform(-0.05, 0.5)
    # viewer.cam.azimuth = 180 + np.random.uniform(-2, 2)
    # viewer.cam.elevation = -25 + np.random.uniform(1, 3)
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
    distances = np.arange(1.0, 1.45, 0.05)
    azimuths = np.arange(178, 184, 1.0)
    viewer.cam.lookat[2] = 0.5
    pcds = []
    if test_elevation:
        for ele in elevations:
            for a in azimuths:
                for d in distances:
                    viewer.cam.distance = 1.25
                    viewer.cam.elevation = -11
                    viewer.cam.azimuth = 180
                    # viewer.cam.distance = d
                    # viewer.cam.elevation = ele
                    # viewer.cam.azimuth = a

                    randomize_textures(modder, env.sim)
                    
                    obs_img, depth_image = env.render(mode="rgb_array", height=100, width=100, depth=True)
                   
                    obs_img, depth_image = use_real_depths_and_crop(obs_img, depth_image)

                    real_img, real_depth = get_real_image_from_recording()
                 

                    real_depth_grey = cv2.cvtColor(real_depth, cv2.COLOR_GRAY2BGR)
                    depth_image_grey = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)
                    obs_img_cv = cv2.cvtColor(obs_img, cv2.COLOR_RGB2BGR)
                    real_img_cv = cv2.cvtColor(real_img, cv2.COLOR_RGB2BGR)
                    mini = real_depth.min()
                    maxi = real_depth.max()
                    depth_image_grey = (depth_image - mini) / (maxi - mini)
                    real_depth_grey = (real_depth - mini) / (maxi - mini)
                    numpy_images = np.hstack((obs_img_cv, real_img_cv))
                    print(real_depth_grey.shape)
                    print(depth_image_grey.shape)
                    numpy_horizontal = np.hstack((depth_image_grey.squeeze(2), real_depth_grey))
                    cv2.imshow('all depths', numpy_horizontal)
                    cv2.imshow('all images', numpy_images)
                    cv2.waitKey(1)

                    description = "Distance: {d}, Elevation: {ele}, Azimuth: {a}"
                    print(description)
                    pcds.append((description, create_point_cloud(obs_img, depth_image, fovy=45)))
        # return pcds
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


def test_pick_and_place():
    env = PickAndPlaceXarm(xml_path='./assets/fetch/pick_and_place_xarm.xml')

    viewer = load_viewer(env.sim)
    env.env._viewers['rgb_array'] = viewer
    modder = get_texture_modder(env)
    go_through_all_possible_randomizations(env, viewer, modder)


# get_real_pcd_from_recording(all=True)
# test_pick_and_place()
