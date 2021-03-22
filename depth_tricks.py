import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np 
from rl_modules.ddpg_agent import show_video
import cv2


def create_point_cloud(col_img, dep_img, fovy=45, convert_img=True):
    if convert_img:
        col_img = cv2.cvtColor(col_img, cv2.COLOR_BGR2RGB)
    dep_img = dep_img.astype(np.float32)
   
    height = col_img.shape[0]
    width = col_img.shape[1]
    col_img = o3d.cpu.pybind.geometry.Image(col_img)
    dep_img = o3d.cpu.pybind.geometry.Image(dep_img)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(col_img, dep_img, convert_rgb_to_intensity=False)

    # plt.title('Redwood grayscale image')
    # plt.imshow(rgbd_image.color)
    # # print(rgbd_image.color.shape)
    # # show_video(rgbd_image.color)
    # plt.subplot(1, 2, 2)
    # plt.title('Redwood depth image')
    # plt.imshow(rgbd_image.depth)
    # plt.show()


    # calculate focal length
    f = (1./np.tan(np.deg2rad(fovy)/2)) * height / 2.0
    intrinsic = np.array(((f, 0, width / 2), (0, f, height / 2), (0, 0, 1)))
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=width, height=height, fx=f, fy=f, cx=width / 2, cy=height/2)
    # create point cloud and display
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # o3d.visualization.draw_geometries([pcd])

    return pcd


def create_point_cloud2(dep_img, col_img, near=0.02, far=2, fovy=45):
    height = col_img.shape[0]
    width = col_img.shape[1]
    # fovy = env.sim.model.cam_fovy[-1]

    # print(dep_img.dtype)
    col_img = o3d.cpu.pybind.geometry.Image(col_img)
    dep_img = o3d.cpu.pybind.geometry.Image(dep_img)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(col_img, dep_img, convert_rgb_to_intensity=False)

    # plt.title('Redwood grayscale image')
    # plt.imshow(rgbd_image.color)
    # plt.subplot(1, 2, 2)
    # plt.title('Redwood depth image')
    # plt.imshow(rgbd_image.depth)
    # plt.show()
    

    # calculate focal length
    f = (1./np.tan(np.deg2rad(fovy)/2)) * height / 2.0
    intrinsic = np.array(((f, 0, width / 2), (0, f, height / 2), (0, 0, 1)))
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=width, height=height, fx=f, fy=f, cx=width / 2, cy=height/2)

    # create point cloud and display
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return pcd

