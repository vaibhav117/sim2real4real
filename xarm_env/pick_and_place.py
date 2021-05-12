from rl_modules.utils import use_real_depths_and_crop
from xarm_env.load_xarm7 import FetchEnv
from gym import utils as utz
from rl_modules.utils import load_viewer, show_video, randomize_textures, get_texture_modder, load_viewer_to_env, use_real_depths_and_crop, scripted_action
import numpy as np
import matplotlib.pyplot as plt

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

def out_of_bounds(obs):
    '''
    Return true if object is out of bounds from robot reach
    '''
    #if x pos is < 0.9  < 1.8 then bad, 0.05 < y < 1.3
    object_pos = obs['observation'][3:6]
    x = object_pos[0]
    y = object_pos[1]

    if x < 0.9 or x >= 1.75:
        return True
    
    if y < 0.05 or y > 1.25:
        return True
    
    return False

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
        is_succ = 0
        k = 0
        ended = True
        m = 0
        distances = []
        while (not is_succ) or k < 50:
            j += 1
            # rgb, depth = env.render(mode='rgb_array', depth=True, height=100, width=100)
            # rgb, depth = use_real_depths_and_crop(rgb, depth)
            # show_video(rgb)
            # print(obs["observation"][:3])
            
            act, pick_object = scripted_action(obs, pick_object)
            
            env.render()
            if ended or obs["observation"][0] < 1.1:
                ended = True
                act, pick_object = scripted_action(obs, pick_object) 
            else:
                act = np.asarray([-1, 0, 0, 0])

            obs,  r, _, infor = env.step(act)
            
            left_gripper = obs['observation'][:3]
            right_gripper = obs['observation'][-3:]
            distances.append(abs(right_gripper[1] - left_gripper[1]))
            # print(abs(right_gripper[1] - left_gripper[1]))
            if infor['is_success'] != 1:
                k = 0
            else:
                k += 1

            is_succ = infor['is_success']

            m += 1

            if out_of_bounds(obs):
                print("object out of bounds, ending episode...")
                break

            show_big_ball(env, obs['observation'][-3:])
            
            if m > 500:
                print("episode is too long, breaking...")
                break

        # plt.plot(np.arange(len(distances)), distances, color='red')
        # plt.show()
        print(is_succ)
# z = 0.7
# 0 < y < 1.2
# 1.1 < x < 1.7
test_scripted_policy()