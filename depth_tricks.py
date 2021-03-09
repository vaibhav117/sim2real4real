import open3d
import matplotlib.pyplot as plt

def create_point_cloud(env, dep_img, col_img):
    height = col_im.shape[0]
    width = col_im.shape[1]
    fovy = env.sim.model.cam_fovy[-1]

    # get z variables
    extent = env.sim.model.stat.extent
    near = env.sim.model.vis.map.znear * extent
    far = env.sim.model.vis.map.zfar * extent

    dep_img = near / (1 - dep_img * (1 - near / far))
    dep_img = dep_img.copy()

    col_img = o3d.cpu.pybind.geometry.Image(col_img)
    dep_img = o3d.cpu.pybind.geometry.Image(dep_img)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(col_img, dep_img, convert_rgb_to_intensity=False)

    plt.title('Redwood grayscale image')
    plt.imshow(rgbd_image.color)
    plt.subplot(1, 2, 2)
    plt.title('Redwood depth image')
    plt.imshow(rgbd_image.depth)
    plt.show()

    # calculate focal length
    f = (1./np.tan(np.deg2rad(fovy)/2)) * height / 2.0
    intrinsic = np.array(((f, 0, width / 2), (0, f, height / 2), (0, 0, 1)))
    camera_intrinsic = open3d.camera.PinholeCameraIntrinsic(width=200, height=200, fx=f, fy=f, cx=width / 2, cy=height/2)

    # create point cloud and display
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([pcd])

