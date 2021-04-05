from rl_modules.utils import use_real_depths_and_crop
from xarm_env.load_xarm7 import FetchEnv
from gym import utils as utz
from rl_modules.utils import load_viewer, show_video, randomize_textures, get_texture_modder
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

def test():
    env = XarmFetchPickPlaceEnv()

    viewer = load_viewer(env.sim)
    env._viewers['rgb_array'] = viewer

    modder = get_texture_modder(env)
    randomize_textures(modder, env)
    while True:
        print("Resetting..")
        obs = env.reset()

        import time
        for i in range(80):
            print(obs["observation"].shape)
            desired_goal = obs["desired_goal"]  # place of goal
            achieved_goal = obs["achieved_goal"] # actual goal

            # show_big_ball(env, achieved_goal)

            # action = obs["achieved_goal"] - desired_goal
            # obs["observation"][6:9] = object relative pose, so action we should take to go to object

            action = obs["observation"][6:9] - [0, 0.1, 0.1]

            b = np.asarray((-1)).reshape((1))
            print(b.shape)
            action = np.concatenate((action, b), axis=0)
            print(action.shape)
            env.render()
            obs,  r, _, infor = env.step(action)
            print(f"Reward is {r}")
            

        # continue


        # # for i in range(30):
        # for i in range(100):
        #     env.render()

        # open gripper
        # for i in range(10):


        #     desired_goal = obs["desired_goal"]  # place of goal
        #     achieved_goal = obs["achieved_goal"] # actual goal

        #     # show_big_ball(env, achieved_goal)
        #     # show_big_ball(env, desired_goal)

        #     env.render()
        #     action = np.asarray([0, 0, 0, -1])
        #     obs  ,_, _, infor = env.step(action)

        # # go slightly sideways
        # for i in range(12):

        #     desired_goal = obs["desired_goal"]  # place of goal
        #     achieved_goal = obs["achieved_goal"] # actual goal

        #     # show_big_ball(env, achieved_goal)
        #     # show_big_ball(env, desired_goal)

        #     env.render()
        #     action = np.asarray([0, -0.1, 0, -1])
            
        #     obs, _, _, infor = env.step(action)

        # # go down
        # for i in range(15):

        #     desired_goal = obs["desired_goal"]  # place of goal
        #     achieved_goal = obs["achieved_goal"] # actual goal

        #     # show_big_ball(env, achieved_goal)
        #     # show_big_ball(env, desired_goal)

        #     env.render()
        #     action = np.asarray([0, 0, -0.5, -1])
            
        #     obs, _, _, infor = env.step(action)
        

        # # close gripper
        for i in range(50):

            desired_goal = obs["desired_goal"]  # place of goal
            achieved_goal = obs["achieved_goal"] # actual goal

            # show_big_ball(env, achieved_goal)
            # show_big_ball(env, desired_goal)

            env.render()
            action = np.asarray([0, 0, 0, 1])
            obs, r, _, infor = env.step(action)

            print(f"Reward is {r}")

        
        ## go to goal
        for i in range(100):

            desired_goal = obs["desired_goal"]  # place of goal
            achieved_goal = obs["achieved_goal"] # actual goal

            # show_big_ball(env, achieved_goal)
            # show_big_ball(env, desired_goal)
            action = desired_goal - achieved_goal

            env.render()
            action = np.asarray([action[0]*10, action[1]*10, action[2]*10, 1])
            obs, r, _, infor = env.step(action)

            print(f"Reward is {r}")

        # for i in range(50):
        #     action = np.asarray([0, 0, 0, 0.1])
        #     env.render()
        
        # # go up
        # for i in range(40):

        #     desired_goal = obs["desired_goal"]  # place of goal
        #     achieved_goal = obs["achieved_goal"] # actual goal

        #     # show_big_ball(env, achieved_goal)

        #     env.render()
        #     action = np.asarray([0, 0, 0.5, 1])
            
        #     obs, _, _, infor = env.step(action)

        # while True:
        #     for i in range(40):

        #         desired_goal = obs["desired_goal"]  # place of goal
        #         achieved_goal = obs["achieved_goal"] # actual goal

        #         show_big_ball(env, achieved_goal)

        #         env.render()
        #         action = np.asarray([0, 0.5, 0,1])
                
        #         obs, _, _, infor = env.step(action)
            
        #     for i in range(40):
        #         desired_goal = obs["desired_goal"]  # place of goal
        #         achieved_goal = obs["achieved_goal"] # actual goal

        #         show_big_ball(env, achieved_goal)

        #         env.render()
        #         action = np.asarray([0, -0.5, 0, 1])
        #         obs, _, _, infor = env.step(action)
        #     break
        

# test()