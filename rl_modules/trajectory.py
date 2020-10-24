class Trajectory:

    def __init__(self):
        self.obs_states = []
        self.goal_states = []
        self.ach_goal_states = []
        self.actions = []
        self.obs_imgs = []
        self.env_states = []
        self.her_obs_imgs = []
    
    def add(self, observation):
        """
        Observation: has observation states, goal states, achieved goal states, and images and env renders 
        """
        self.obs_states.append(observation['observation'].copy())
        self.goal_states.append(observation['desired_goal'].copy())
        self.ach_goal_states.append(observation['achieved_goal'].copy())
        if observation['action'] is not None: # we do not get last action in rollout
            self.actions.append(observation['action'].copy())
        self.obs_imgs.append(observation['observation_image'].copy())
        self.env_states.append(observation['env_state'])
