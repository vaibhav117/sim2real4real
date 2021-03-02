import threading
import numpy as np

"""
the replay buffer here is basically from the openai baselines code

"""
class replay_buffer:
    def __init__(self, env_params, buffer_size, sample_func, image_based, sym_image):
        self.env_params = env_params
        self.T = env_params['max_timesteps']
        self.size = buffer_size // self.T
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        self.sample_func = sample_func
        self.image_based = image_based
        self.sym_image = sym_image
        # create the buffer to store info
        self.buffers = {'obs_states': np.empty([self.size, self.T + 1, self.env_params['obs']]),
                        'obs_imgs': np.empty([self.size, self.T + 1, 100, 100, 3], dtype=np.uint8),
                        'ach_goal_states': np.empty([self.size, self.T + 1, self.env_params['goal']]),
                        'goal_states': np.empty([self.size, self.T+1, self.env_params['goal']]),
                        'actions': np.empty([self.size, self.T, self.env_params['action']]),
                        'her_obs_imgs': np.empty([self.size, self.T + 1, 100, 100, 3],  dtype=np.uint8),
                        }
        # thread lock
        self.lock = threading.Lock()
    
    # store the episode
    def store_episode(self, episode_batch):
        if self.sym_image:
            mb_obs, mb_ag, mb_g, mb_actions, mb_obs_imgs, mb_g_obs = episode_batch
        elif self.image_based:
            mb_obs, mb_ag, mb_g_imgs, mb_actions, mb_obs_imgs = episode_batch
        else:
            mb_obs, mb_ag, mb_g, mb_actions = episode_batch

        batch_size = mb_obs.shape[0] 
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            # store the informations
            if self.sym_image:
                self.buffers['obs_img'][idxs] = mb_obs_imgs
                self.buffers['g_o'][idxs] = mb_g_obs
            elif self.image_based:
                self.buffers['obs_img'][idxs] = mb_obs_imgs
                self.buffers['g_o'][idxs] = mb_obs_imgs
            
            self.buffers['obs'][idxs] = mb_obs
            self.buffers['ag'][idxs] = mb_ag
            self.buffers['g'][idxs] = mb_g
            self.buffers['actions'][idxs] = mb_actions
            self.n_transitions_stored += self.T * batch_size
    

    # store the episode
    def store_trajectories(self, episode_batch):
        obs_states, goal_states, ach_goal_states, obs_imgs, actions, _ = episode_batch

        batch_size = obs_states.shape[0] 
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            # store the information
            self.buffers['obs_states'][idxs] = obs_states
            self.buffers['ach_goal_states'][idxs] = ach_goal_states
            self.buffers['goal_states'][idxs] = goal_states
            self.buffers['actions'][idxs] = actions
            self.buffers['obs_imgs'][idxs] = obs_imgs
            self.n_transitions_stored += self.T * batch_size
    
    def create_batch(self, trajectories):
        obs_states = []
        goal_states = []
        ach_goal_states = []
        obs_imgs = []
        actions = []

        for traj in trajectories:
            obs_states.append(traj.obs_states)
            goal_states.append(traj.goal_states)
            ach_goal_states.append(traj.ach_goal_states)
            obs_imgs.append(traj.obs_imgs)
            actions.append(traj.actions)
        
        obs_states = np.array(obs_states)
        goal_states = np.array(goal_states)
        ach_goal_states = np.array(ach_goal_states)
        obs_imgs = np.array(obs_imgs)
        actions = np.array(actions)
        return [obs_states, goal_states, ach_goal_states, obs_imgs, actions, None]
    
    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
        temp_buffers['obs_states_next'] = temp_buffers['obs_states'][:, 1:, :]
        temp_buffers['obs_imgs_next'] = temp_buffers['obs_imgs'][:, 1:, :]
        temp_buffers['her_obs_imgs_next'] = temp_buffers['her_obs_imgs'][:, 1:, :]
        temp_buffers['ach_goal_states_next'] = temp_buffers['ach_goal_states'][:, 1:, :]
        # sample transitions
        transitions = self.sample_func(temp_buffers, batch_size)
        return transitions

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size+inc)
        if inc == 1:
            idx = idx[0]
        return idx




"""
the replay buffer here is basically from the openai baselines code

"""
class new_replay_buffer:
    def __init__(self, env_params, buffer_size, sample_func, image_based, sym_image, height=100, width=100):
        self.env_params = env_params
        self.T = env_params['max_timesteps']
        self.size = buffer_size // self.T
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        self.sample_func = sample_func
        self.image_based = image_based
        self.sym_image = sym_image
        env_state_dim = 5
        # create the buffer to store info
        self.buffers = {'obs_states': np.empty([self.size, self.T + 1, self.env_params['obs']]),
                        'obs_imgs': np.empty([self.size, self.T + 1, height, width, 3], dtype=np.uint8),
                        'ach_goal_states': np.empty([self.size, self.T + 1, self.env_params['goal']]),
                        'goal_states': np.empty([self.size, self.T+1, self.env_params['goal']]),
                        'actions': np.empty([self.size, self.T, self.env_params['action']]),
                        'env_states': []
                        }
        # thread lock
        self.lock = threading.Lock()

    # store the episode
    def store_trajectories(self, episode_batch):
        obs_states, goal_states, ach_goal_states, obs_imgs, actions, env_states = episode_batch

        batch_size = obs_states.shape[0] 
        with self.lock:
            # print('Dim:', env_states.shape)
            idxs = self._get_storage_idx(inc=batch_size)
            # store the information
            self.buffers['obs_states'][idxs] = obs_states
            self.buffers['ach_goal_states'][idxs] = ach_goal_states
            self.buffers['goal_states'][idxs] = goal_states
            self.buffers['actions'][idxs] = actions
            self.buffers['obs_imgs'][idxs] = obs_imgs
            self.buffers['env_states'].append(env_states)
            self.n_transitions_stored += self.T * batch_size
    
    def create_batch(self, trajectories):
        obs_states = []
        goal_states = []
        ach_goal_states = []
        obs_imgs = []
        actions = []
        env_states = []

        for traj in trajectories:
            obs_states.append(traj.obs_states)
            goal_states.append(traj.goal_states)
            ach_goal_states.append(traj.ach_goal_states)
            obs_imgs.append(traj.obs_imgs)
            actions.append(traj.actions)
            env_states.append(traj.env_states)
        
        obs_states = np.array(obs_states)
        goal_states = np.array(goal_states)
        ach_goal_states = np.array(ach_goal_states)
        obs_imgs = np.array(obs_imgs)
        actions = np.array(actions)
        # env_states = np.array(env_states)
        return [obs_states, goal_states, ach_goal_states, obs_imgs, actions, env_states]
    
    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
        temp_buffers['obs_states_next'] = temp_buffers['obs_states'][:, 1:, :]
        temp_buffers['obs_imgs_next'] = temp_buffers['obs_imgs'][:, 1:, :]
        temp_buffers['ach_goal_states_next'] = temp_buffers['ach_goal_states'][:, 1:, :]
        next_states = []
        curr_states = []
        for buf in temp_buffers['env_states']:
            for all_states in buf:
                next_states.append(all_states[1:])
                curr_states.append(all_states)
        temp_buffers['env_states_next'] = next_states
        temp_buffers['env_states'] = curr_states
        temp_buffers['obs_imgs_with_goals'] = temp_buffers['obs_imgs'].copy()
        temp_buffers['obs_imgs_with_goals_next'] = temp_buffers['obs_imgs_next'].copy()
        # sample transitions
        transitions = self.sample_func(temp_buffers, batch_size)
        return transitions

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size+inc)
        if inc == 1:
            idx = idx[0]
        return idx




