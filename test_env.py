import gym
import matplotlib.pyplot as plt
import cv2
import numpy as np 
from gym.envs.robotics import rotations, robot_env, utils

from mujoco_py import MjRenderContextOffscreen


def env_setup(env, goal):
    initial_qpos = {
        'robot0:slide0': 0.4049,
        'robot0:slide1': 0.48,
        'robot0:slide2': 0.0,
    }

    for name, value in initial_qpos.items():
        env.sim.data.set_joint_qpos(name, value)
    utils.reset_mocap_welds(env.sim)
    ach_goal = env.sim.data.get_site_xpos('robot0:grip')
    env.sim.data.set_xpos('robot0:grip', goal)
    print(ach_goal)
    env.sim.forward()

    # qpos = env.sim.data.qpos

    # Move end effector into position.
    gripper_target = np.array([-0.498, 0.005, -0.431 + env.gripper_extra_height]) + env.sim.data.get_site_xpos('robot0:grip')
    gripper_rotation = np.array([1., 0., 1., 0.])
    env.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
    env.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
    for _ in range(10):
        env.sim.step()

    # Extract information for sampling goals.
    env.initial_gripper_xpos = env.sim.data.get_site_xpos('robot0:grip').copy()
    if env.has_object:
        env.height_offset = env.sim.data.get_site_xpos('object0')[2]


def reset_goal_fetch_reach(env, ach_goal):
    sites_offset = (env.env.sim.data.site_xpos - env.env.sim.model.site_pos).copy()
    site_id = env.env.sim.model.site_name2id('target0')
    env.env.sim.model.site_pos[site_id] = ach_goal - sites_offset[0]
    env.env.sim.forward()
    # env.env.sim.forward()

def reset_goal_fetch_push(env, desired_goal):
    # sites_offset = (env.env.sim.data.site_xpos - env.env.sim.model.site_pos).copy()
    # site_id = env.env.sim.model.site_name2id('target0')
    # env.env.sim.model.site_pos[site_id] = env.env.sim.data.get_site_xpos('robot0:grip').copy() - sites_offset[0]
    # env.env.sim.forward()

    object_xpos = env.env.initial_gripper_xpos[:2]
    while np.linalg.norm(object_xpos - env.env.initial_gripper_xpos[:2]) < 0.1:
        object_xpos = env.env.initial_gripper_xpos[:2] + env.env.np_random.uniform(-env.env.obj_range, env.env.obj_range, size=2)
    object_qpos = env.env.sim.data.get_joint_qpos('object0:joint')
    assert object_qpos.shape == (7,)
    # object_qpos[:2] = object_xpos
    # print(ach_goal)
    object_qpos[:2] = desired_goal[:2]
    env.env.sim.data.set_joint_qpos('object0:joint', object_qpos)
    env.env.sim.forward()
    # pass

def timeit(*args, **kwargs):
    avg_time = 0
    for i in range(kwargs["num_itrs"]):
        t = time.time()
        args(kwargs)
        e = time.time()
        avg_time = avg_time + (e - t)
    # print(f"Time taken is {avg_time / kwargs["num_itrs"]} secs")

# @timeit
def benchmark_gpu_rendering(env):
    # add rendering context
    viewer = MjRenderContextOffscreen(env.env.sim, True, -1, "cuda")
    viewer.cam.distance = 1.2
    viewer.cam.azimuth = 180
    viewer.cam.elevation = -25
    env.env._viewers['rgb_array'] = viewer


# benchmark_gpu_rendering(env)

def goal_realign(mode):

    if mode == 'fetch_reach':
        env = gym.make('FetchReach-v1')
    else:
        env = gym.make('FetchPush-v1')


    states = []
    observation = env.reset()
    # sites_offset = (env.env.sim.data.site_xpos - env.env.sim.model.site_pos).copy()
    # site_id = env.env.sim.model.site_name2id('target0')
    # env.env.sim.model.site_pos[site_id] = observation['desired_goal'] - sites_offset[0]
    # env.env.sim.forward()

    for x in range(100):
        # img_obs = env.render(mode="rgb_array", height=100, width=100)
        # env.render()
        ach_goal = observation["achieved_goal"]
        if mode == 'fetch_reach':
            # reset_goal_fetch_reach(env, ach_goal)
            pass
        else:
            # reset_goal_fetch_push(env, observation["desired_goal"])
            pass
        # env.env._get_viewer("rgb_array").render(100, 100)
        # data = env.env._get_viewer("rgb_array").read_pixels(100, 100, depth=False)

        # Visualize target.
        # print(ach_goal, observation["desired_goal"])
        sites_offset = (env.env.sim.data.site_xpos - env.env.sim.model.site_pos).copy()
        site_id = env.env.sim.model.site_name2id('target0')
        env.env.sim.model.site_pos[site_id] = observation['desired_goal'] - sites_offset[0]
        env.env.sim.forward()

        # env.env._get_viewer("human").render()

        # img = data[::-1, :, :]
        # cv2.imshow('frame', cv2.resize(img, (200,200)))
        # cv2.waitKey(0)
        states.append(env.env.sim.get_state())
        observation, _, _, _ = env.step(env.action_space.sample())
        print("hello")

    # now make red dot at the pint where box is at last step
    last_s = states[-1]
    env.env.sim.set_state(last_s)
    pos_of_box = env.env.sim.data.get_joint_qpos('object0:joint') # 7 dim
    # pos = pos_of_box[:3] - [0.02,0.02,0]
    pos = observation["achieved_goal"] - [0.02,0.02,0]
    
    for x in range(100):
        state = states[x]
        env.env.sim.set_state(state)
        # render red ball
        sites_offset = (env.env.sim.data.site_xpos - env.env.sim.model.site_pos).copy()
        site_id = env.env.sim.model.site_name2id('target0')
        # pos[:2] = env.env.np_random.uniform(-env.env.obj_range, env.env.obj_range, size=2)
        env.env.sim.model.site_pos[site_id] = pos - sites_offset[0]
        env.env.sim.forward()


        # env.render()
        env.env._get_viewer("human").render()


# goal_realign(mode='fetch_push')

def plot_shit():
    rews_sym_images = [0, 0, 0.1, 0, 0.1, 0.2, 0.1, 0.4, 0.1, 0.2, 0, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.4, 0.5, 0.4, 0.1, 0.3]

    rews_sym_states = [0, 0.5, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    rews_asym_states_images = [0, 0.3, 0.7, 0.8, 0.8,0.8,0.8,0.8, 0.8, 0.82, 0.86, 0.9, 0.87, 0.89, 0.9, 0.9, 0.9, 0.8, 0.9,0.9, 0.9, 0.9, 0.95]
    plt.plot(rews_sym_images, label="sym images", color="red")
    plt.plot(rews_sym_states, label="sym states", color="blue")
    plt.plot(rews_asym_states_images, label="asym images/states", color="green")
    plt.plot()
    plt.ylabel("rewards")
    plt.ylabel("epochs")
    plt.legend()
    plt.title("Fetch Reach")
    plt.show()

# plot_shit()
# observation = env.reset()
# print(observation["achieved_goal"])
# while True:
#     # observation = env.reset()
#     env.env._get_viewer("human").render()
#     env_setup(env.env, observation["desired_goal"])
#     env.env._get_viewer("human").render()

env = gym.make('FetchReach-v1')
print(env.action_space.sample())
