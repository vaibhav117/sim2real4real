from xarm_env.load_xarm7 import FetchEnv
from gym import utils as utz

class XarmFetchPickPlaceEnv(FetchEnv, utz.EzPickle):
    def __init__(self, xml_path='assets/fetch/pick_and_place_xarm.xml', reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.4049,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
        }
        FetchEnv.__init__(
            self, xml_path, has_object=True, block_gripper=False, n_substeps=100,
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