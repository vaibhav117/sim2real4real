import numpy as np

def sample_her_transitions(episode_batch):
    p = 1 - (1. / (1 + 4))
    T = episode_batch['actions'].shape[1]
    rollout_batch_size = episode_batch['actions'].shape[0]
    batch_size = 1
    # select which rollouts and which timesteps to be used
    episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
    t_samples = np.random.randint(T, size=batch_size)
    transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}
    # her idx
    her_indexes = np.where(np.random.uniform(size=batch_size) < p)
    future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
    future_offset = future_offset.astype(int)
    future_t = (t_samples + 1 + future_offset)[her_indexes]
    print(her_indexes)
    print(future_t)
    
    # replace go with achieved goal
    future_ag = episode_batch['ach_goal_states'][episode_idxs[her_indexes], future_t]
    transitions['goal_states'][her_indexes] = future_ag
    # to get the params to re-compute reward
    transitions['r'] = np.expand_dims(transitions['ach_goal_states_next'] - transitions['goal_states'], 1)
    transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

    return transitions

a = {
    'actions': np.array([[ [1,2,3], [4,5,6] ] ]),
    'goal_states': np.array([ [1, 2] ]),
    'ach_goal_states': np.array([ [3,4] ]),
    'ach_goal_states_next': np.array([ [4, 5] ])
}

def test():
    b = sample_her_transitions(a)
    print(b)
# test()