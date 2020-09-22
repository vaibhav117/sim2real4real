import torch
from rl_modules.models import actor, new_actor
from arguments import get_args
import gym
from mujoco_py import load_model_from_path, MjSim, MjViewer, MjRenderContextOffscreen
import numpy as np
from mujoco_py.generated import const

# process the inputs
def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs



if __name__ == '__main__':
    image_based = True
    args = get_args()
    # load the model param
    model_path = args.save_dir + args.env_name + '/model.pt'
    o_mean, o_std, g_mean, g_std, model = torch.load(model_path, map_location=lambda storage, loc: storage)
    # create the environment
    env = gym.make(args.env_name)
    # get the env param
    observation = env.reset()
    # get the environment params
    env_params = {'obs': observation['observation'].shape[0], 
                  'goal': observation['desired_goal'].shape[0], 
                  'action': env.action_space.shape[0], 
                  'action_max': env.action_space.high[0],
                  }

    if image_based:
        sim = env.sim
        viewer = MjRenderContextOffscreen(sim)
        viewer.cam.fixedcamid = 3
        viewer.cam.type = const.CAMERA_FIXED
        env.env._viewers['rgb_array'] = viewer

    # create the actor network
    actor_network = new_actor(env_params)
    actor_network.load_state_dict(model)
    actor_network.eval()
    for i in range(args.demo_length):
        observation = env.reset()
        # start to do the demo
        obs = observation['observation']
        g = observation['desired_goal']
        for t in range(env._max_episode_steps):
            env.render()

            if image_based:
                obs_img = env.render(mode="rgb_array", height=100, width=100).copy()
                print(obs_img.shape)
                obs_img = torch.tensor(obs_img, dtype=torch.float32).unsqueeze(0)
                obs_img = obs_img.permute(0, 3, 1, 2)
                g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
                g_norm = torch.tensor(np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range),  dtype=torch.float32).unsqueeze(0)
            else:
                inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std, args)

            with torch.no_grad():
                if not image_based:
                    pi = actor_network(inputs)
                else:
                    pi = actor_network(obs_img, g_norm)
            action = pi.detach().numpy().squeeze()
            # put actions into the environment
            observation_new, reward, _, info = env.step(action)
            obs = observation_new['observation']
        print('the episode is: {}, is success: {}'.format(i, info['is_success']))
