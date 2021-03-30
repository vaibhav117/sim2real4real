from rl_modules.utils import use_real_depths_and_crop
from rl_modules.ddpg_agent import randomize_textures, randomize_camera
import numpy as np
import cv2
from pcd_utils import get_real_depth, display_interactive_point_cloud
from depth_tricks import create_point_cloud
import torch

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
    return []
    pcds = []
    viewer.cam.distance = 0.8
    if test_elevation:
        for ele in elevations:
            for a in azimuths:
                for d in distances:
                    return []
                    viewer.cam.distance = 1.20
                    viewer.cam.elevation = -11
                    viewer.cam.azimuth = 180
                    # viewer.cam.distance = d
                    # viewer.cam.elevation = ele
                    # viewer.cam.azimuth = a

                    randomize_textures(modder, env.sim)
                    
                    obs_img, depth_image = env.render(mode="rgb_array", height=100, width=100, depth=True)
                    # viewer.render(width, height)
                    # data, dep = viewer.read_pixels(width, height, depth=True)
                    # obs_img, depth_image = data[::-1, :, :], dep[::-1, :]
                    
                    # depth_image = normalize_depth(depth_image)
                    # depth_image = cv2.resize(depth_image[10:80, 10:90], (100,100))
                    # obs_img = cv2.resize(obs_img[10:80, 10:90, :], (100,100))
                    obs_img, depth_image = use_real_depths_and_crop(obs_img, depth_image)

                    real_depth, real_img = get_real_depth()
                    # fig, axs = plt.subplots(2, 2)
                    # print(depth_image.shape)
                    # axs[0, 0].imshow(real_depth)
                    # axs[0, 1].imshow(depth_image)
                    # axs[1, 0].imshow(real_img)
                    # axs[1, 1].imshow(obs_img)
                    # plt.show()
                    # # exit()

                    # real_depth_grey = cv2.cvtColor(real_depth, cv2.COLOR_GRAY2BGR)
                    # depth_image_grey = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)
                    # obs_img_cv = cv2.cvtColor(obs_img, cv2.COLOR_RGB2BGR)
                    # real_img_cv = cv2.cvtColor(real_img, cv2.COLOR_RGB2BGR)
                    # mini = real_depth.min()
                    # maxi = real_depth.max()
                    # depth_image_grey = (depth_image - mini) / (maxi - mini)
                    # real_depth_grey = (real_depth - mini) / (maxi - mini)
                    # numpy_images = np.hstack((obs_img_cv, real_img_cv))
                    # numpy_horizontal = np.hstack((real_depth_grey, depth_image_grey))
                    # cv2.imshow('all depths', numpy_horizontal)
                    # cv2.imshow('all images', numpy_images)
                    # cv2.waitKey(1)
                    description = f"Distance: {d}, Elevation: {ele}, Azimuth: {a}"
                    print(description)
                    pcds.append((description, create_point_cloud(obs_img, depth_image, fovy=45)))
        return pcds
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
    