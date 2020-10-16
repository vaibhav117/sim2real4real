import threading
import numpy as np

class image_replay_buffer:
    def __init__(self, env_params, buffer_size):
        self.env_params = env_params
        self.size = buffer_size
        self.T = env_params['max_timesteps']
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        # create the buffer to store info
        self.buffers = {
                        'obs_imgs': np.empty([buffer_size, 100, 100, 3], dtype=np.uint8),
                        'next_obs_imgs': np.empty([buffer_size, 100, 100, 3], dtype=np.uint8),
                        'rewards': np.empty([buffer_size], dtype=np.uint8),
                        'actions': np.empty([buffer_size, self.env_params['action']]),
                        }
        # thread lock
        self.lock = threading.Lock()
    
    # store the episode
    def store_episode(self, obs_img, next_obs_img, action, rew):
        batch_size = obs_img.shape[0]
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            # store the informations
            self.buffers['obs_imgs'][idxs] = obs_img
            self.buffers['next_obs_imgs'][idxs] = next_obs_img
            self.buffers['actions'][idxs] = action
            self.buffers['rewards'][idxs] = rew
            self.n_transitions_stored += self.T * batch_size
    
    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
        return temp_buffers

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

 
class state_replay_buffer:
    def __init__(self, env_params, buffer_size):
        self.env_params = env_params
        self.size = buffer_size
        self.T = env_params['max_timesteps']
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        # create the buffer to store info
        self.buffers = {
                        'obs_states': np.empty([buffer_size, self.env_params['obs']], dtype=np.uint8),
                        'obs_states_next': np.empty([buffer_size, self.env_params['obs']], dtype=np.uint8),
                        'r': np.empty([buffer_size], dtype=np.uint8),
                        'actions': np.empty([buffer_size, self.env_params['action']]),
                        'goal_states': np.empty([buffer_size, self.env_params['goal']]),
                        'goal_states_next': np.empty([buffer_size, self.env_params['goal']])
                        }
        # thread lock
        self.lock = threading.Lock()
    
    # store the episode
    def store_episode(self, obs_states, next_obs_states, actions, rewards, goal_states, next_goal_states):
        batch_size = obs_states.shape[0]
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            # store the informations
            self.buffers['obs_states'][idxs] = obs_states
            self.buffers['obs_states_next'][idxs] = next_obs_states
            self.buffers['actions'][idxs] = actions
            self.buffers['r'][idxs] = rewards
            self.buffers['goal_states'][idxs] = goal_states
            self.buffers['goal_states_next'][idxs] = next_goal_states
            self.n_transitions_stored += self.T * batch_size
    
    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
        return temp_buffers

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