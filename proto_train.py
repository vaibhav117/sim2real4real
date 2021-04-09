import os
os.environ['MUJOCO_GL'] = 'egl'

import copy
import math
import pickle as pkl
import sys
import time

import numpy as np

# import dmc
import hydra
import torch
import proto.utils as utils
from proto.logger import Logger
from proto.replay_buffer import ReplayBuffer
from proto.video import VideoRecorder

from xarm_env.pick_and_place import PickAndPlaceXarm
from rl_modules.utils import load_viewer, use_real_depths_and_crop


torch.backends.cudnn.benchmark = True

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


def get_rgbd(env, height, width):
    rgb, depth = env.render(mode="rgb_array", height=height, width=width, depth=True)
    rgb = rgb.astype(np.float32)
    rgb = rgb / 255 # normalize image data between 0 and 1
    depth = depth[:, :, np.newaxis]
    # depth = depth + np.random.uniform(-0.01, 0.01, size=depth.shape) # randomise depth by 1 cm

    rgb, depth = use_real_depths_and_crop(rgb, depth)
    rgbd = np.concatenate((rgb, depth), axis=2)
    rgbd = np.transpose(rgbd, (2, 0, 1))
    return rgbd

class Workspace(object):
    def __init__(self, cfg):
        self.height = 100
        self.width = 100

        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')
        self.model_dir = utils.make_dir(self.work_dir, 'model')
        self.buffer_dir = utils.make_dir(self.work_dir, 'buffer')

        self.cfg = cfg

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency_step,
                             action_repeat=cfg.action_repeat,
                             agent=cfg.agent.name)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        # TODO: change env
        # import os
        os.system("pwd")
        self.env = PickAndPlaceXarm(xml_path='../../../assets/fetch/pick_and_place_xarm.xml')
        viewer1 = load_viewer(self.env.sim)
        self.env.env._viewers['rgb_array'] = viewer1

        self.eval_env = PickAndPlaceXarm(xml_path='../../../assets/fetch/pick_and_place_xarm.xml')
        viewer2 = load_viewer(self.eval_env.sim)
        self.eval_env.env._viewers['rgb_array'] = viewer2

        self.env_params = get_env_params(self.env)
        obs_spec = [4, self.height, self.width]
        action_spec = [4]

        cfg.agent.params.obs_shape = obs_spec
        cfg.agent.params.action_shape = action_spec
        cfg.agent.params.action_range = [
            float(-self.env_params['action_max']),
            float(self.env_params['action_max'])
        ]
        # exploration agent uses intrinsic reward
        self.expl_agent = hydra.utils.instantiate(cfg.agent,
                                                  task_agnostic=True)
        # task agent uses extr extrinsic reward
        self.task_agent = hydra.utils.instantiate(cfg.agent,
                                                  task_agnostic=False)
        self.task_agent.assign_modules_from(self.expl_agent)

        if cfg.load_pretrained:
            pretrained_path = utils.find_pretrained_agent(
                cfg.pretrained_dir, cfg.env, cfg.seed, cfg.pretrained_step)
            print(f'snapshot is taken from: {pretrained_path}')
            pretrained_agent = utils.load(pretrained_path)
            self.task_agent.assign_modules_from(pretrained_agent)

        # buffer for the task-agnostic phase
        self.expl_buffer = ReplayBuffer(obs_spec, action_spec,
                                        cfg.replay_buffer_capacity,
                                        self.device)
        # buffer for task-specific phase
        self.task_buffer = ReplayBuffer(obs_spec, action_spec,
                                        cfg.replay_buffer_capacity,
                                        self.device)

        self.eval_video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)
        self.step = 0

    def get_agent(self):
        if self.step < self.cfg.num_expl_steps:
            return self.expl_agent
        return self.task_agent

    def get_buffer(self):
        if self.step < self.cfg.num_expl_steps:
            return self.expl_buffer
        return self.task_buffer

    def evaluate(self):
        avg_episode_reward = 0
        for episode in range(self.cfg.num_eval_episodes):
            observation = self.eval_env.reset()
            self.eval_video_recorder.init(enabled=(episode == 0))
            episode_reward = 0
            episode_success = 0
            episode_step = 0
            for i in range(50): # 50 is the max time steps
                agent = self.get_agent()
                with utils.eval_mode(agent):
                    obs = get_rgbd(self.eval_env, self.height, self.width)
                    # obs = time_step.observation['pixels']
                    action = agent.act(obs, sample=False)
                observation, rew, _ , _ = self.eval_env.step(action)
                self.eval_video_recorder.record(self.eval_env)
                episode_reward += rew
                episode_step += 1

            avg_episode_reward += episode_reward
            self.eval_video_recorder.save(f'{self.step}.mp4')
        avg_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward', avg_episode_reward, self.step)
        self.logger.dump(self.step, ty='eval')

    def run(self):
        episode, episode_reward, episode_step = 0, 0, 0
        start_time = time.time()
        done = True
        while self.step <= self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    fps = episode_step / (time.time() - start_time)
                    self.logger.log('train/fps', fps, self.step)
                    start_time = time.time()
                    self.logger.log('train/episode_reward', episode_reward, self.step)
                    self.logger.log('train/episode', episode, self.step)
                    self.logger.dump(self.step, ty='train')

                observation = self.env.reset()
                obs = get_rgbd(self.env, self.height, self.width)
                print(obs.shape)
                # obs = time_step.observation['pixels']
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            agent = self.get_agent()
            replay_buffer = self.get_buffer()
            # evaluate agent periodically
            if self.step % self.cfg.eval_frequency == 0:
                self.logger.log('eval/episode', episode - 1, self.step)
                self.evaluate()

            # save agent periodically
            if self.cfg.save_model and self.step % self.cfg.save_frequency == 0:
                utils.save(
                    self.expl_agent,
                    os.path.join(self.model_dir, f'expl_agent_{self.step}.pt'))
                utils.save(
                    self.task_agent,
                    os.path.join(self.model_dir, f'task_agent_{self.step}.pt'))
            if self.cfg.save_buffer and self.step % self.cfg.save_frequency == 0:
                replay_buffer.save(self.buffer_dir, self.cfg.save_pixels)

            # sample action for data collection
            if self.step < self.cfg.num_random_steps:
                # spec = self.env.action_spec()
                action = np.random.uniform(-self.env_params['action_max'], self.env_params['action_max'],
                                           [4])
            else:
                with utils.eval_mode(agent):
                    action = agent.act(obs, sample=True)

            agent.update(replay_buffer, self.step)

            observation, rew, is_done, _ = self.env.step(action)
            next_obs = get_rgbd(self.env, self.height, self.width)

            # allow infinite bootstrap
            done = is_done
            episode_reward += rew

            replay_buffer.add(obs, action, rew, next_obs, done)

            obs = next_obs
            episode_step += 1
            self.step += 1


@hydra.main(config_path='proto/config.yaml', strict=True)
def main(cfg):
    from proto_train import Workspace as W
    workspace = W(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
