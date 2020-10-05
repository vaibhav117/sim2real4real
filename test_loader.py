import torch
from rl_modules.models import actor, critic, img_actor, new_actor, resnet_actor
import gym 

def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'action': env.action_space.shape[0],
            'action_max': env.action_space.high[0],
            }
    params['max_timesteps'] = env._max_episode_steps
    return params

env = gym.make('FetchReach-v1')
env_params = get_env_params(env)
model = new_actor(env_params)
model_path = './test.pt'
torch.save(model.state_dict(), model_path)
new_model = new_actor(env_params)
new_model.load_state_dict(torch.load(model_path))