import gym
import matplotlib.pyplot as plt
import cv2
import numpy as np 
from gym.envs.robotics import rotations, robot_env, utils

env = gym.make('FetchReach-v1')


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

states = []
observation = env.reset()
def goal_realign():
    for x in range(1000):
        # img_obs = env.render(mode="rgb_array", height=100, width=100)
        # env.render()
        ach_goal = observation["achieved_goal"]
        reset_goal_fetch_reach(env, ach_goal)
        # env.env._get_viewer("rgb_array").render(100, 100)
        # data = env.env._get_viewer("rgb_array").read_pixels(100, 100, depth=False)
        env.env._get_viewer("human").render()
        # img = data[::-1, :, :]
        # cv2.imshow('frame', cv2.resize(img, (200,200)))
        # cv2.waitKey(0)
        states.append(env.env.sim.get_state())
        observation, _, _, _ = env.step(env.action_space.sample())
        print("hello")


    for x in range(1000):
        state = states[x]
        env.env.sim.set_state(state)

        env.render()

observation = env.reset()
print(observation["achieved_goal"])
while True:
    # observation = env.reset()
    env.env._get_viewer("human").render()
    env_setup(env.env, observation["desired_goal"])
    env.env._get_viewer("human").render()
    # print("hello")
    