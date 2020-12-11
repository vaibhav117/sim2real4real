import torch
from rl_modules.models import actor, critic, asym_goal_outside_image, sym_image, sym_image_critic
from arguments import get_args
import gym
from mujoco_py import load_model_from_path, MjSim, MjViewer, MjRenderContextOffscreen
import numpy as np
from mujoco_py.generated import const
from rl_modules.ddpg_agent import randomize_textures
from mujoco_py.modder import TextureModder, MaterialModder, CameraModder, LightModder
from rl_modules.ddpg_agent import model_factory
import cv2
import numpy as np


video=cv2.VideoWriter('video.mp4',-1,1,(100,100))


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

# pre_process the inputs
def _preproc_inputs_image(obs_img, g, cuda):
    obs_img = torch.tensor(obs_img, dtype=torch.float32)
    obs_img = obs_img.permute(0, 3, 1, 2)
    g_norm = torch.tensor(g, dtype=torch.float32)
    if cuda:
        obs_img = obs_img.cuda()
        g_norm = g_norm.cuda()
    return obs_img, g_norm

def _eval_agent(args, paths, image_based=True, cuda=False):

        # load model
        env = gym.make(args.env_name)
        sim = env.sim
        viewer = MjRenderContextOffscreen(sim)
        # self.viewer.cam.fixedcamid = 3
        # self.viewer.cam.type = const.CAMERA_FIXED
        viewer.cam.distance = 1.2
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -25
        env.env._viewers['rgb_array'] = viewer

        # model_path = '../../test.pt'
        # model_path = './server_weights/asym_goal_in_image/FetchPush-v1/model.pt'
        # model_path = './weird_weights/FetchSlide-v1/model.pt'
        # model_path = './randomized_server_weights/FetchReach-v1/model.pt'
        # model_path = args.save_dir + args.env_name + '/model.pt'


        # env params
        env_params = get_env_params(env)
        env_params["model_path"] = paths[args.env_name][args.task] # TODO: fix bad practice
        env_params["load_saved"] = args.loadsaved

        loaded_model, _, _, _ = model_factory(args.task, env_params)

        # if args.task == 'sym_image':
        #     loaded_model = sym_image(get_env_params(env))
        # else:
        #     loaded_model = asym_goal_outside_image(get_env_params(env))
        model_path = paths[args.env_name][args.task] + '/model.pt'
        obj = torch.load(model_path, map_location=lambda storage, loc: storage)
        loaded_model.load_state_dict(obj['actor_net'])

        # loaded_model.load_state_dict(obj[4])

        if cuda:
            loaded_model.cuda()

        total_success_rate = []
        rollouts = []

        for _ in range(args.n_test_rollouts):
            per_success_rate = []
            observation = env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            rollout = []
            obs_img = env.render(mode="rgb_array", height=100, width=100)
            modder = TextureModder(env.sim)
            # randomize_textures(modder, env.sim)
            for _ in range(env._max_episode_steps):
                # env.render()
                rollout.append(obs_img)
                video.write(obs_img)
                cv2.imshow('frame', cv2.resize(obs_img, (200,200)))
                cv2.waitKey(0)
                with torch.no_grad():
                    if args.task == 'sym_image':
                        o_tensor, _ = _preproc_inputs_image(obs_img.copy()[np.newaxis, :], g[np.newaxis, :], cuda)
                        pi = loaded_model(o_tensor)
                    else:
                        if image_based:
                            o_tensor, g_tensor = _preproc_inputs_image(obs_img.copy()[np.newaxis, :], g[np.newaxis, :], cuda)
                            pi = loaded_model(o_tensor, g_tensor)
                        else:
                            input_tensor = self._preproc_inputs(obs, g)
                            pi = actor_network(input_tensor)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()
                observation_new, _, _, info = env.step(actions)
                obs = observation_new['observation']
                obs_img = env.render(mode="rgb_array", height=100, width=100)
                g = observation_new['desired_goal']
                per_success_rate.append(info['is_success'])
            total_success_rate.append(per_success_rate)
            rollouts.append(rollout)
        cv2.destroyAllWindows()
        video.release()
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        print(local_success_rate)
        #global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)



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
    paths = {
        'FetchReach-v1': {
            'sym_state': './weights/saved_models/FetchReach-v1/',
            'asym_goal_outside_image': './randomized_server_weights/asym_goal_outside_image/FetchReach-v1/',
            'asym_goal_in_image': '',
            'sym_image': ''
        },
        'FetchPush-v1': {

        },
        'FetchSlide-v1': {

        },
        'FetchPickAndPlace-v1': {

        }
    }
    args = get_args()
    args.env_name = 'FetchReach-v1'
    args.task = 'asym_goal_outside_image'
    _eval_agent(args, paths)
  