from rl_modules.utils import use_real_depths_and_crop
from xarm_env.load_xarm7 import FetchEnv
from gym import utils as utz
from rl_modules.utils import load_viewer, show_video, randomize_textures, get_texture_modder, load_viewer_to_env, use_real_depths_and_crop
import numpy as np

class XarmFetchPickPlaceEnv(FetchEnv, utz.EzPickle):
    def __init__(self, xml_path='assets/fetch/pick_and_place_xarm.xml', reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.4049,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
        }
        FetchEnv.__init__(
            self, xml_path, has_object=True, block_gripper=False, n_substeps=10,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.10, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utz.EzPickle.__init__(self)


class PickAndPlaceXarm:

    def __init__(self, xml_path, reward_type='sparse'):
        self.env = XarmFetchPickPlaceEnv(xml_path, reward_type)
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


def show_big_ball(env, pos):
    sites_offset = (env.sim.data.site_xpos - env.sim.model.site_pos).copy()
    # print(f"Sites Offset {sites_offset}")
    site_id = env.sim.model.site_name2id('target1')
    env.sim.model.site_pos[site_id] = pos - sites_offset[0]
    env.sim.forward()
    # env.sim.step()


def scripted_action(obs, picked_object):
    if not picked_object:
        if abs(obs["observation"][6]) > 0.001 or abs(obs["observation"][7] - 0.02) > 0.001:
            # print("X")
            action = np.asarray([obs["observation"][6], obs["observation"][7] - 0.02, 0]) * 50
            b = np.asarray((-1)).reshape((1))
            action = np.concatenate((action, b), axis=0)
            return action, False

        # if abs(obs["observation"][7] - 0.02) > 0.001:
        #     print("Y")
        #     y_act = obs["observation"][7] - 0.02  
        #     action = np.asarray([0, y_act, 0]) * 10
        #     b = np.asarray((-1)).reshape((1))
        #     action = np.concatenate((action, b), axis=0)
        #     return action, False
        
        if abs(obs["observation"][8]) > 0.001:
            # print("Z")
            action = np.asarray([0, 0, obs["observation"][8]]) * 50
            b = np.asarray((-1)).reshape((1))
            action = np.concatenate((action, b), axis=0)
            return action, False

    if abs(obs["observation"][7]) > 0.017:
        # print("close gripper")
        action = np.asarray([0, 0, 0, 1])
        return action, True

    # print("go towards goal")
    desired_goal = obs["desired_goal"]  # place of goal
    achieved_goal = obs["achieved_goal"] # actual goal

    action = desired_goal - achieved_goal
    scaler = 50
    action = np.asarray([action[0]*scaler, action[1]*scaler, action[2]*scaler, 1])

    return action, True



def test_scripted_policy():
    env = PickAndPlaceXarm(xml_path='./assets/fetch/pick_and_place_xarm.xml')

    env = load_viewer_to_env(env)
    modder = get_texture_modder(env)

    num_steps = 50
    while True:
        obs = env.reset()
        pick_object = False
        r = -1
        j = 0
        for i in range(60):
            j += 1
            rgb, depth = env.render(mode='rgb_array', depth=True, height=100, width=100)
            rgb, depth = use_real_depths_and_crop(rgb, depth)
            show_video(rgb)
            act, pick_object = scripted_action(obs, pick_object) 
            # env.render()
            obs,  r, _, infor = env.step(act)
        print(j)

# test_scripted_policy()