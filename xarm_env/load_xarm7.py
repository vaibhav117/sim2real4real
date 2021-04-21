import torch
import gym
import os
import mujoco_py
from gym import utils
import time
import numpy as np

from gym.envs.robotics import rotations
from gym.envs.robotics import utils

import os
import copy
import numpy as np

import gym
from gym import error, spaces
from gym.utils import seeding

from mujoco_py.modder import TextureModder, MaterialModder, CameraModder, LightModder


def randomize_textures(sim, modder):
    for name in sim.model.geom_names:
        modder.rand_all(name)


try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

DEFAULT_SIZE = 500

def robot_get_obs(sim):
    """Returns all joint positions and velocities associated with
    a robot.
    """
    if sim.data.qpos is not None and sim.model.joint_names:
        names = [n for n in sim.model.joint_names]
        qposes = []
        qvels = []

        for name in names:
            if name == 'object0:joint':
                b = sim.data.get_joint_qvel(name)
                c = sim.data.get_joint_qpos(name)
                qvels.append(b.ravel())
                qposes.append(c.ravel())
            else:
                qvels.append(sim.data.get_joint_qvel(name).reshape(1))
                qposes.append(sim.data.get_joint_qpos(name).reshape(1))
        
        qposes = np.concatenate(qposes)
        qvels = np.concatenate(qvels)
        
        return qposes, qvels
        
    return np.zeros(0), np.zeros(0)

class RobotEnv(gym.GoalEnv):
    def __init__(self, model_path, initial_qpos, n_actions, n_substeps):

        model = mujoco_py.load_model_from_path(model_path)
        self.sim = mujoco_py.MjSim(model, nsubsteps=n_substeps)
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.seed()
        self._env_setup(initial_qpos=initial_qpos)
        self.initial_state = copy.deepcopy(self.sim.get_state())

        self.goal = self._sample_goal()
        obs = self._get_obs()
        self.action_space = spaces.Box(-1., 1., shape=(n_actions,), dtype='float32')
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    # Env methods
    # ----------------------------

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        # print(f"Reward: {reward}")
        return obs, reward, done, info

    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        super(RobotEnv, self).reset()
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal().copy()
        obs = self._get_obs()
        return obs

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE, depth=False):
        self._render_callback()
        if mode == 'rgb_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            if depth == False:
                data = self._get_viewer(mode).read_pixels(width, height, depth=depth)
            else:
                data, dep = self._get_viewer(mode).read_pixels(width, height, depth=depth)
                return data[::-1, :, :], dep[::-1, :]
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer(mode).render()

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'rgb_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)
            self._viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    # Extension methods
    # ----------------------------

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
        self.sim.set_state(self.initial_state)
        self.sim.forward()
        return True

    def _get_obs(self):
        """Returns the observation.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        raise NotImplementedError()

    def _sample_goal(self):
        """Samples a new goal and returns it.
        """
        raise NotImplementedError()

    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        pass

    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position,
        for example.
        """
        pass

    def _render_callback(self):
        """A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class FetchEnv(RobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold, initial_qpos, reward_type,
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self._max_episode_steps = 50 

        super(FetchEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos=initial_qpos)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        # print(f"Distance is {d}")
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    def get_gripper_pos(self):
        grip_pos = self.sim.data.get_site_xpos('ee')
        return grip_pos

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            # self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            # self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            ctrl_set_action(self.sim, 1)
            # for _ in range(10):
            #     self.sim.step()
            self.sim.forward()

    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]


        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        
        # gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl]) # wtf does this do ??
        
        # Don't know what to do about this.
        
        # assert gripper_ctrl.shape == (2,)
        # if self.block_gripper:
        #     gripper_ctrl = np.zeros_like(gripper_ctrl)


        # action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        if gripper_ctrl <= 0:
            # open gripper
            ctrl_set_action(self.sim, -1)
            # for i in range(50):
            # self.sim.step()
        elif gripper_ctrl > 0:
            # close gripper
            ctrl_set_action(self.sim, 1.0)
            # for i in range(50):
            # self.sim.step()


        one = np.asarray([0, 0, 0, 0])
        action_with_0_rotation = np.expand_dims(np.concatenate((pos_ctrl, one)), axis=0)
        mocap_set_action(self.sim, action_with_0_rotation)

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('ee')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        #grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = robot_get_obs(self.sim)
        if self.has_object:
            object_pos = self.sim.data.get_site_xpos('object0')
            # rotations
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
            # velocities
            object_velp = self.sim.data.get_site_xvelp('object0') * dt
            object_velr = self.sim.data.get_site_xvelr('object0') * dt
            # gripper state
            object_rel_pos = object_pos - grip_pos
            # object_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        
        
        # we need all gripper joint information, unlike old joint information that was only left and righ finger joint.
        gripper_state = robot_qpos[7:12]

        # TODO: no velocity information to be fed right now
        gripper_vel = robot_qvel[7:12] * dt  # change to a scalar if the gripper is made symmetric

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())
       

        # state for reach: grip_pos, gripper_state
        if not self.has_object:
            obs = np.concatenate([
                grip_pos, gripper_state
            ])
        else:
            obs = np.concatenate([
                grip_pos, object_pos.ravel(), object_rel_pos.ravel(), object_rot.ravel(), object_velp.ravel(), object_velr.ravel(), gripper_state,
            ])
            


        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _viewer_setup(self):

        # return
        body_id = self.sim.model.body_name2id('link7')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        # print(f"Sites Offset {sites_offset}")
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        self.sim.forward()
        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.01:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
                # object_xpos = self.initial_gripper_xpos[:2] + np.asarray([0, 0])
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)

            # with 0.5 probability, set starting state of the robot gripper on top of the object
            if np.random.uniform(0, 1) > 0.5:
                act = np.zeros((4,))

                pos_of_box = object_qpos[:3]

                init_gripper_position = self.initial_gripper_xpos

                act[:3] = pos_of_box - init_gripper_position
                act[1] -= 0.03
                # open gripper
                act[3] = -1

                self._set_action(act)

                for i in range(80):
                    self._set_action(act)
                    self.sim.step()
                    # self.render()

        self.sim.forward()
        return True

    def _sample_goal(self):
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
            # # identify bounds
            goal = self.initial_gripper_xpos[:3] + np.asarray([self.target_range, -self.target_range, 0.1])
            goal += self.target_offset
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
        # print(f"Goal is {goal}")
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
                
        reset_mocap_welds(self.sim) # Important
        reset_mocap2body_xpos(self.sim)
        ctrl_set_action(self.sim, 1)
        action = np.asarray([[0.3, 0, 0.15, 0, 0, 0, 0]])
        mocap_set_action(self.sim, action)
        self.sim.forward()

        for i in range(50):
            self.sim.step()

        # viewer = mujoco_py.MjViewer(self.sim)
        # while True:
        #     print("hey")
        #     viewer.render()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('ee').copy()
        # print(f"Initial gripper position {self.initial_gripper_xpos}")
        # exit()
        
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]
        
        self.sim.forward()


    def render(self, mode='human', width=500, height=500, depth=False):
        return super(FetchEnv, self).render(mode, width, height, depth=depth)


from gym import utils as utz


class XarmFetchReachEnv(FetchEnv, utz.EzPickle):
    def __init__(self, xml_path, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.4049,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
        }
        FetchEnv.__init__(
            self, xml_path, has_object=False, block_gripper=True, n_substeps=10,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utz.EzPickle.__init__(self)



def reset_mocap2body_xpos(sim):
    """Resets the position and orientation of the mocap bodies to the same
    values as the bodies they're welded to.
    """

    if (sim.model.eq_type is None or
        sim.model.eq_obj1id is None or
        sim.model.eq_obj2id is None):
        return
    for eq_type, obj1_id, obj2_id in zip(sim.model.eq_type,
                                         sim.model.eq_obj1id,
                                         sim.model.eq_obj2id):
        if eq_type != mujoco_py.const.EQ_WELD:
            continue

        
        mocap_id = sim.model.body_mocapid[obj1_id]
        if mocap_id != -1:
            # obj1 is the mocap, obj2 is the welded body
            body_idx = obj2_id
        else:
            # obj2 is the mocap, obj1 is the welded body
            mocap_id = sim.model.body_mocapid[obj2_id]
            body_idx = obj1_id
        

        assert (mocap_id != -1)
        sim.data.mocap_pos[mocap_id][:] = sim.data.body_xpos[body_idx]
        sim.data.mocap_quat[mocap_id][:] = sim.data.body_xquat[body_idx]


def mocap_set_action(sim, action):
    """The action controls the robot using mocaps. Specifically, bodies
    on the robot (for example the gripper wrist) is controlled with
    mocap bodies. In this case the action is the desired difference
    in position and orientation (quaternion), in world coordinates,
    of the of the target body. The mocap is positioned relative to
    the target body according to the delta, and the MuJoCo equality
    constraint optimizer tries to center the welded body on the mocap.
    """
    if sim.model.nmocap > 0:
        # action, _ = np.split(action, (sim.model.nmocap * 7, ))
        # action = action.reshape(sim.model.nmocap, 7)

        pos_delta = action[:, :3]
        quat_delta = action[:, 3:]

        reset_mocap2body_xpos(sim)
        # print(sim.data.mocap_pos, type(sim.data.mocap_pos))
        sim.data.mocap_pos[:] = sim.data.mocap_pos + pos_delta
        sim.data.mocap_quat[:] = sim.data.mocap_quat + quat_delta

        # sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        # sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)


def reset_mocap_welds(sim):
    """Resets the mocap welds that we use for actuation.
    """
    if sim.model.nmocap > 0 and sim.model.eq_data is not None:
        for i in range(sim.model.eq_data.shape[0]):
            if sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                sim.model.eq_data[i, :] = np.array(
                    [0., 0., 0., 1., 0., 0., 0.])
    sim.forward()


def ctrl_set_action(sim, action):
    """For torque actuators it copies the action into mujoco ctrl field.
    For position actuators it sets the target relative to the current qpos.
    """
    if sim.data.ctrl is not None:
        i = 0
        if sim.model.actuator_biastype[i] == 0:
            sim.data.ctrl[i] = action
        else:
            idx = sim.model.jnt_qposadr[sim.model.actuator_trnid[i, 0]]
            sim.data.ctrl[i] = sim.data.qpos[idx] + action

class ReachXarm:

    def __init__(self, xml_path, reward_type='sparse'):
        self.env = XarmFetchReachEnv(xml_path, reward_type)
        self.sim = self.env.sim

    def reset(self):
        return self.env.reset()
    
    def render(self, mode="human", width=500, height=500, depth=False):
        return self.env.render(mode, width, height, depth=depth)
    
    def step(self, action):
        return self.env.step(action)

    def compute_reward(self, achieved_goal, goal, info):
        return self.env.compute_reward(achieved_goal, goal, info)

    def seed(self, seed=None):
        return self.env.seed(seed)
    
    def close(self):
        return self.env.close()

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def dt(self):
        return self.env.dt

    @property
    def _max_episode_steps(self):
        return self.env._max_episode_steps
    

def check_distance():
    reach_combined_gripper = 'assets/fetch/reach_xarm_with_gripper.xml'


    env = XarmFetchReachEnv(xml_path=reach_combined_gripper)
    obs = env.reset()

    print(env.get_gripper_pos())
    s = time.time()
    for i in range(2):
        action = np.asarray([1, 0, 0, 1])
        env.step(action)
        env.render()
    e = time.time()
    print(f"{e - s} seconds")
    print(env.get_gripper_pos())


def test_like_a_mf():

    reach_env = './assets/fetch/reach.xml'
    xarm_path = './assets/fetch/robot_xarm_main.xml'
    gripper_model = './assets/final/xarm7_with_gripper.xml'
    reach_gripper_model = './assets/fetch/reach_gripper.xml'
    reach_combined = './assets/fetch/reach_combined.xml'
    reach_combined_gripper = 'assets/fetch/reach_xarm_with_gripper.xml'


    env = XarmFetchReachEnv(xml_path=reach_combined_gripper)
    obs = env.reset()

    modder = TextureModder(env.sim)

    while True:
        obs = env.reset()
        randomize_textures(env.sim, modder)
        for _ in range(50):
            action = env.action_space.sample()
            action = np.zeros_like(action)
            print(action.shape)
            # action = np.asarray([0.01, 0.02, 0.03, 1])
            # goal = obs["desired_goal"]
            # action[0:3] = goal
            obs, rew, done, _ = env.step(action)
            env.render()

        
# test_like_a_mf()
# check_distance()
