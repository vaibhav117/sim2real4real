import numpy as np
import matplotlib.pyplot as plt 
import cv2 
import gym
from rl_modules.utils import use_real_depths_and_crop
from mujoco_py.modder import TextureModder

def reset_goal_fetch_reach(env, ach_goal):
    sites_offset = (env.env.sim.data.site_xpos - env.env.sim.model.site_pos).copy()
    site_id = env.env.sim.model.site_name2id('target0')
    env.env.sim.model.site_pos[site_id] = ach_goal - sites_offset[0]
    env.env.sim.forward()
    return env

def reset_goal_fetch_push(env, ach_goal):
    ach_goal = ach_goal - [0.02,0.02,0]
    sites_offset = (env.env.sim.data.site_xpos - env.env.sim.model.site_pos).copy()
    site_id = env.env.sim.model.site_name2id('target0')
    env.env.sim.model.site_pos[site_id] = ach_goal - sites_offset[0]
    env.env.sim.forward()
    return env

def reset_goal_fetch_pick_place(env, ach_goal):
    # ach_goal = ach_goal - [0.02,0.02,0]
    # sites_offset = (env.env.sim.data.site_xpos - env.env.sim.model.site_pos).copy()
    # site_id = env.env.sim.model.site_name2id('target0')
    # env.env.sim.model.site_pos[site_id] = ach_goal - sites_offset[0]
    # env.env.sim.forward()
    return env

def randomize_textures(modder, sim):
    for name in sim.model.geom_names:
        modder.rand_all(name)


def render_image_without_fuss(env, height=100, width=100, depth=True):
    env.env._get_viewer("rgb_array").render(height, width)
    if depth:
        modder = TextureModder(env.sim)
        randomize_textures(modder, env.sim)
        data, dep = env.env._get_viewer("rgb_array").read_pixels(height, width, depth=depth)
        rgb, depth = data[::-1, :, :], dep[::-1, :]

        # create rgbd
        rgb, depth = use_real_depths_and_crop(rgb, depth)
        # show_video1(rgb)

        # from depth_tricks import create_point_cloud
        # create_point_cloud(rgb, depth, vis=True)

        # concatenate ze stuff
        rgbd = np.concatenate((rgb, depth), axis=2)
        return rgbd
    else:
        data = env.env._get_viewer("rgb_array").read_pixels(height, width, depth=depth)
        img = data[::-1, :, :]
        return img



def get_img(env, s, ag, mode="push", depth=True):
    env.sim.set_state(s)
    if mode == 'reach':
        reset_goal_fetch_reach(env, ag)
    elif mode == 'push':
        reset_goal_fetch_push(env, ag)
    elif mode == 'pick_place':
        reset_goal_fetch_pick_place(env, ag)

    curr_img = render_image_without_fuss(env, depth=depth)
    return curr_img

def show_video(img1, img2):
    cv2.imshow('current frame', cv2.resize(img1, (200,200)))
    cv2.imshow('next frame', cv2.resize(img2, (200,200)))
    cv2.waitKey(0)

def show_video1(img1):
    cv2.imshow('img', cv2.resize(img1, (200,200)))
    cv2.waitKey(0)

class her_sampler:
    def __init__(self, replay_strategy, replay_k, reward_func=None, image_based=False, sym_image=False, mode=None):
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k
        self.image_based = image_based
        self.sym_image = sym_image
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + replay_k))
        else:
            self.future_p = 0
        self.reward_func = reward_func
        self.mode = mode

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        batch_size = batch_size_in_transitions
        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}
        
       
        # her idx
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        # replace go with achieved goal
        future_ag = episode_batch['ach_goal_states'][episode_idxs[her_indexes], future_t]
        
        transitions['goal_states'][her_indexes] = future_ag
        # to get the params to re-compute reward
        transitions['r'] = np.expand_dims(self.reward_func(transitions['ach_goal_states_next'], transitions['goal_states'], None), 1)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        # show_video()
        # if 'obs_imgs' in transitions:
        #     for i in range(20):
        #         img1 = transitions['obs_imgs'][i]
        #         img2 = transitions['obs_imgs'][i+1]
        #         show_video(img1, img2)
        return transitions



class her_sampler_new:
    def __init__(self, replay_strategy, replay_k, env, reward_func=None, image_based=False, sym_image=False, mode='reach', depth=True):
        self.replay_strategy = replay_strategy
        self.depth = depth
        self.replay_k = replay_k
        self.image_based = image_based
        self.sym_image = sym_image
        self.env = env
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + replay_k))
        else:
            self.future_p = 0
        self.reward_func = reward_func
        self.mode = mode

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        batch_size = batch_size_in_transitions
        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        if 'env_states' not in episode_batch:
            transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}
        else:
            transitions = {}
            for key in episode_batch.keys():
                if key != 'env_states' and key != 'env_states_next':
                    transitions[key] = episode_batch[key][episode_idxs, t_samples].copy()
            states_tracker = {}
            states_tracker['env_states'] = list(map(lambda x, y: episode_batch['env_states'][x][y], episode_idxs, t_samples))
            states_tracker['env_states_next'] = list(map(lambda x, y: episode_batch['env_states_next'][x][y], episode_idxs, t_samples))

        # her idx
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1)[her_indexes]
        # replace go with achieved goal
        future_ag = episode_batch['ach_goal_states'][episode_idxs[her_indexes], future_t]


        # TODO: extrememly slow implementation, please fix
        if 'env_states' in episode_batch:
            states = list(map(lambda x: states_tracker['env_states'][x], her_indexes[0]))
            next_states = list(map(lambda x: states_tracker['env_states_next'][x], her_indexes[0]))

            img_obs_with_new_goal_curr = []
            img_obs_with_new_goal_next = []
            for s, n_s, ag, idx in zip(states, next_states, future_ag, her_indexes[0]):
                img_obs_curr = get_img(self.env, s, ag, self.mode, depth=self.depth)
                img_obs_next = get_img(self.env, n_s, ag, self.mode, depth=self.depth)
                # show_video(img_obs_curr[:, :, :3], img_obs_curr[:, :, 3])
                # plt.imshow(img_obs_curr)
                # plt.show()

                img_obs_with_new_goal_curr.append(img_obs_curr)
                img_obs_with_new_goal_next.append(img_obs_next)

            img_obs_with_new_goal_curr = np.array(img_obs_with_new_goal_curr)
            img_obs_with_new_goal_next = np.array(img_obs_with_new_goal_next)
            transitions['obs_imgs_with_goals'][her_indexes] = img_obs_with_new_goal_curr
            ## what to do ?
            transitions['obs_imgs_with_goals_next'][her_indexes] = img_obs_with_new_goal_next
            

        transitions['goal_states'][her_indexes] = future_ag
        # to get the params to re-compute reward
        transitions['r'] = np.expand_dims(self.reward_func(transitions['ach_goal_states_next'], transitions['goal_states'], None), 1)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        return transitions
