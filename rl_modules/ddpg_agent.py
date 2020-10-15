import torch
import os
from datetime import datetime
import numpy as np
from mpi4py import MPI
from mpi_utils.mpi_utils import sync_networks, sync_grads
from rl_modules.replay_buffer import replay_buffer
from rl_modules.image_only_replay_buffer import image_replay_buffer
from rl_modules.models import actor, critic, asym_goal_outside_image, sym_image, sym_image_critic
from mpi_utils.normalizer import normalizer
from her_modules.her import her_sampler
import cv2
import itertools
import matplotlib.pyplot as plt
from rl_modules.cheap_model import cheap_cnn
from mujoco_py import load_model_from_path, MjSim, MjViewer, MjRenderContextOffscreen
from mujoco_py.generated import const
from rl_modules.utils import plot_grad_flow
from torch import autograd
import time
import torch.nn as nn

def show_video(img):
    cv2.imshow('frame', cv2.resize(img, (200,200)))
    cv2.waitKey(0)


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
        self.obs_states.append(observation['observation'].copy())
        self.goal_states.append(observation['desired_goal'].copy())
        self.ach_goal_states.append(observation['achieved_goal'].copy())
        if observation['action'] is not None: # we do not get last action in rollout
            self.actions.append(observation['action'].copy())
        self.obs_imgs.append(observation['observation_image'].copy())
        self.env_states.append(observation['env_state'])

    def get(self):
        raise NotImplementedError
    
    def sample_her(self, env):
        '''
        Stateful function which computes the her sampled image observations

        Return [image, action, next_image, reward]

        80% of these tuples will have HER goals. 20% of these will have regular stuff
        '''
        her_p = 0.8 # with prob 80% sample golas with HER else just leave state as it is.
        end_goal = self.ach_goal_states[-1]
        obs = []
        acts = []
        next_obs = []
        rews = []

        T = len(self.actions)
        for i in range(T):
            s = self.env_states[i]
            # acquired_goal = self.ach_goal_states[i]
            if np.random.uniform(size=1) < her_p:
                # do HER
                env.sim.set_state(s)
                reset_goal_fetch_reach(env, self.ach_goal_states[i+1])
                curr_img = render_image_without_fuss(env)
                env.sim.set_state(self.env_states[i+1])
                reset_goal_fetch_reach(env, self.ach_goal_states[i+1])
                next_img = render_image_without_fuss(env)
                obs.append(curr_img)
                next_obs.append(next_img)
                rews.append(1)
                acts.append(self.actions[i])
                # show_video(curr_img)
                # show_video(next_img)
            else:
                # add normal sample
                obs.append(self.obs_imgs[i])
                rews.append(env.compute_reward(self.goal_states[i], self.ach_goal_states[i+1], None))
                next_obs.append(self.obs_imgs[i+1])
                acts.append(self.actions[i])

        return np.array(obs), np.array(next_obs), np.array(acts), np.array(rews)


def reset_goal_fetch_reach(env, ach_goal):
    sites_offset = (env.env.sim.data.site_xpos - env.env.sim.model.site_pos).copy()
    site_id = env.env.sim.model.site_name2id('target0')
    env.env.sim.model.site_pos[site_id] = ach_goal - sites_offset[0]
    env.env.sim.forward()
    return env

def render_image_without_fuss(env):
    env.env._get_viewer("rgb_array").render(100, 100)
    data = env.env._get_viewer("rgb_array").read_pixels(100, 100, depth=False)
    img = data[::-1, :, :]
    return img


def get_actor_critic_and_target_nets(actor_fn, critic_fn, env_params):
    """
    Creates actor, critic, target nets and syncs nets across CPUs
    """

    actor_network = actor_fn(env_params)
    critic_network = critic_fn(env_params)

    # sync the networks across the cpus
    sync_networks(actor_network)
    sync_networks(critic_network)

    # init target nets
    actor_target_network = actor_fn(env_params)
    critic_target_network = critic_fn(env_params)

    # load same weights as non target nets
    actor_target_network.load_state_dict(actor_network.state_dict())
    critic_target_network.load_state_dict(critic_network.state_dict())

    return actor_network, actor_target_network, critic_network, critic_target_network

def model_factory(task, env_params) -> [nn.Module, nn.Module]:
    """
    Returns actor critic for experiment setup
    """
    if task == "sym_state":
        return get_actor_critic_and_target_nets(actor, critic, env_params)
    elif task == "asym_goal_outside_image":
        return get_actor_critic_and_target_nets(asym_goal_outside_image, critic, env_params)
    elif task == "asym_goal_in_image":
        return get_actor_critic_and_target_nets(asym_goal_in_image, critic, env_params)
    elif task == "sym_image":
        return get_actor_critic_and_target_nets(sym_image, sym_image_critic, env_params)

"""
ddpg with HER (MPI-version)
"""
class ddpg_agent:
    def __init__(self, args, env, env_params):
        self.args = args
        self.env = env
        self.env_params = env_params
        sim = self.env.sim
        self.viewer = MjRenderContextOffscreen(sim)
        # self.viewer.cam.fixedcamid = 3
        # self.viewer.cam.type = const.CAMERA_FIXED
        self.critic_loss = []
        self.actor_loss = []
        self.viewer.cam.distance = 1.2
        self.viewer.cam.azimuth = 180
        self.viewer.cam.elevation = -25
        env.env._viewers['rgb_array'] = self.viewer

        self.env_params = env_params

        # TODO: remove
        self.image_based = True
        self.sym_image = True

        # create the network
        self.actor_network, self.actor_target_network, self.critic_network, self.critic_target_network = model_factory(args.task, env_params)

        # if use gpu
        if self.args.cuda:
            print("Using the GPU")
            self.actor_network.cuda(MPI.COMM_WORLD.Get_rank())
            self.critic_network.cuda(MPI.COMM_WORLD.Get_rank())
            self.actor_target_network.cuda(MPI.COMM_WORLD.Get_rank())
            self.critic_target_network.cuda(MPI.COMM_WORLD.Get_rank())

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)
        # her sampler
        self.her_module = her_sampler(self.args.replay_strategy,
                            self.args.replay_k, 
                            self.env.compute_reward,
                            self.image_based,
                            self.sym_image)
        # create the replay buffer
        self.buffer = self.get_buffer(args.task)
        # create the normalizer
        self.o_norm = normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range)
        # create the dict for store the model
        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)
            # path to save the model
            self.model_path = os.path.join(self.args.save_dir, self.args.env_name)
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)

    def get_buffer(self, task):
        if task == 'sym_state' or task == 'asym_goal_outside_image':
            return replay_buffer(self.env_params, 
                    self.args.buffer_size, 
                    self.her_module.sample_her_transitions,
                    self.image_based,
                    self.sym_image)
        elif task == 'asym_goal_in_image' or task == 'sym_image':
            return image_replay_buffer(self.env_params, 
                    self.args.buffer_size)

    def get_obs(self, task, action=None, step=False):
        if step == False:
            obs = self.env.reset()
        else:
            obs, _, _, info = self.env.step(action)
        if task == "sym_state":
            obs["observation_image"] = self.env.render(mode="rgb_array", height=100, width=100)
            obs["env_state"] = self.env.env.sim.get_state()
            return obs
        elif task == "asym_goal_outside_image":
            obs["observation_image"] = self.env.render(mode="rgb_array", height=100, width=100)
            obs["env_state"] = self.env.env.sim.get_state()
            return obs
        elif task == "asym_goal_in_image":
            obs["observation_image"] = self.env.render(mode="rgb_array", height=100, width=100)
            obs["env_state"] = self.env.env.sim.get_state()
            return obs
        elif task == "sym_image":
            obs["observation_image"] = self.env.render(mode="rgb_array", height=100, width=100)
            obs["env_state"] = self.env.env.sim.get_state()
            return obs

    

    def get_policy(self, task, observation):
        if task == "sym_state":
            input_tensor = self._preproc_inputs(observation["observation"].copy(), observation["desired_goal"].copy())
            pi = self.actor_network(input_tensor)
            return pi
        elif task == "asym_goal_outside_image":
            o_tensor, g_tensor = self._preproc_inputs_image(observation["observation_image"][np.newaxis, :].copy(), observation["desired_goal"][np.newaxis, :].copy())
            pi = self.actor_network(o_tensor, g_tensor)
            return pi
        elif task == "asym_goal_in_image":
            raise NotImplementedError
        elif task == "sym_image":
            obs_img = observation["observation_image"][np.newaxis, :].copy()
            obs_img = torch.tensor(obs_img, dtype=torch.float32)
            obs_img = obs_img.permute(0, 3, 1, 2)
            if self.args.cuda:
                obs_img.cuda(MPI.COMM_WORLD.Get_rank())
            pi = self.actor_network(obs_img)
            return pi
    
    def record_trajectory(self, observation):
        raise NotImplementedError

    def normalize_states_and_store(self, trajectories, task):
        if task == 'sym_state' or task == 'asym_goal_outside_image':
            episode_batch = self.buffer.create_batch(trajectories)
            self.buffer.store_trajectories(episode_batch)
            self._update_normalizer(episode_batch)
    
    def learn(self):
        """
        train the network

        """
        # start to collect samples
        for epoch in range(self.args.n_epochs):
            for _ in range(self.args.n_cycles):
                # start_of_lel = time.time()
                trajectories = []
                for _ in range(self.args.num_rollouts_per_mpi):
                    start_per_rollout = time.time()
                    # reset the rollouts
                    trajectory = Trajectory()
                    # reset the environment
                    observation = self.get_obs(self.args.task)

                    obs = observation['observation']
                    obs_img = observation['observation_image']

                    ag = observation['achieved_goal']
                    g = observation['desired_goal']
                    # start to collect samples
                    for t in range(self.env_params['max_timesteps']):
                        with torch.no_grad():
                            pi = self.get_policy(self.args.task, observation)
                            action = self._select_actions(pi)
                        
                        observation["action"] = action

                        # append rollouts
                        trajectory.add(observation)

                        # feed the actions into the environment
                        observation_new = self.get_obs(self.args.task, action=action, step=True)
                        # reassign observation
                        observation = observation_new
                        observation["action"] = None

                    # add final obs to trajectory
                    trajectory.add(observation)

                    # save trajectory
                    trajectories.append(trajectory)
                    
                    # store images all all steps of trajectory with achieved goal in the image
                    obs, next_obs, actions, rews = trajectory.sample_her(self.env)

                    if self.args.task == 'asym_goal_in_image' or self.args.task == 'sym_image':
                        self.buffer.store_episode(obs, next_obs, actions, rews)


                # TODO: Normalize stuff ? Is this needed ? It helps with purely state based training.

                self.normalize_states_and_store(trajectories, self.args.task)

                for _ in range(self.args.n_batches):
                    self._update_network()
                # soft update
                self._soft_update_target_network(self.actor_target_network, self.actor_network)
                self._soft_update_target_network(self.critic_target_network, self.critic_network)
                # end_of_lel = time.time()
                # print("Updating stuff {}".format(end_of_lel - start_of_lel))
            # start to do the evaluation
            success_rate = self._eval_agent()
            if MPI.COMM_WORLD.Get_rank() == 0:
                print('[{}] epoch is: {}, eval success rate is: {:.3f}'.format(datetime.now(), epoch, success_rate))
                torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, self.actor_network.state_dict()], \
                            self.model_path + '/model.pt')

    # pre_process the inputs
    def _preproc_inputs(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda(MPI.COMM_WORLD.Get_rank())
        return inputs
    
    # pre_process the inputs
    def _preproc_inputs_image(self, obs_img, g):
        obs_img = torch.tensor(obs_img, dtype=torch.float32)
        obs_img = obs_img.permute(0, 3, 1, 2)
        g_norm = torch.tensor(self.g_norm.normalize(g), dtype=torch.float32)
        # g_norm = torch.tensor(g, dtype=torch.float32)
        if self.args.cuda:
            obs_img = obs_img.cuda(MPI.COMM_WORLD.Get_rank())
            g_norm = g_norm.cuda(MPI.COMM_WORLD.Get_rank())
        return obs_img, g_norm
    
    # this function will choose action for the agent and do the exploration
    def _select_actions(self, pi):
        action = pi.cpu().numpy().squeeze()
        # add the gaussian
        action += self.args.noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])
        # random actions...
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                            size=self.env_params['action'])
        # choose if use the random actions
        action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_actions - action)
        return action

    # update the normalizer
    def _update_normalizer(self, episode_batch):
        mb_obs, mb_g, mb_ag, obs_imgs, mb_actions = episode_batch
        
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {'obs_states': mb_obs, 
            'ach_goal_states': mb_ag,
            'goal_states': mb_g, 
            'actions': mb_actions, 
            'obs_states_next': mb_obs_next,
            'ach_goal_states_next': mb_ag_next,
        }
        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs_states'], transitions['goal_states']
        # pre process the obs and g
        transitions['obs_states'], transitions['goal_states'] = self._preproc_og(obs, g)
        # update
        self.o_norm.update(transitions['obs_states'])
        self.g_norm.update(transitions['goal_states'])
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    def get_image_obs_input(self, obs_img, g, target=False):
        obs_img = torch.tensor(obs_img.copy()).to(torch.float32)
        obs_img = obs_img.permute(0, 3, 1, 2)
        if self.args.cuda:
            g_norm = torch.tensor(self.g_norm.normalize(g), dtype=torch.float32).cuda(MPI.COMM_WORLD.Get_rank())
            obs_img = obs_img.cuda(MPI.COMM_WORLD.Get_rank())
        else:
            g_norm = torch.tensor(self.g_norm.normalize(g), dtype=torch.float32)
        if target == False:
            feature_obs_img = self.actor_img_featurizer(obs_img)
        else:
            feature_obs_img = self.actor_img_featurizer_target(obs_img)

        inputs = torch.cat([feature_obs_img, g_norm], dim=1)
        return inputs


    def _prepare_inputs_for_state_only(self, transitions):
        o, o_next, g = transitions['obs_states'], transitions['obs_states_next'], transitions['goal_states']

        transitions['obs_states'], transitions['goal_states'] = self._preproc_og(o, g)
        transitions['obs_states_next'], transitions['goal_states_next'] = self._preproc_og(o_next, g)
        # start to do the update
        obs_norm = self.o_norm.normalize(transitions['obs_states'])
        g_norm = self.g_norm.normalize(transitions['goal_states'])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        obs_next_norm = self.o_norm.normalize(transitions['obs_states_next'])
        g_next_norm = self.g_norm.normalize(transitions['goal_states_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)

        # transfer them into the tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)

        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32)

        if self.args.cuda:
            inputs_norm_tensor = inputs_norm_tensor.cuda(MPI.COMM_WORLD.Get_rank())
            inputs_next_norm_tensor = inputs_next_norm_tensor.cuda(MPI.COMM_WORLD.Get_rank())
            actions_tensor = actions_tensor.cuda(MPI.COMM_WORLD.Get_rank())
            r_tensor = r_tensor.cuda(MPI.COMM_WORLD.Get_rank())
        
        return transitions, inputs_norm_tensor, inputs_next_norm_tensor, actions_tensor, r_tensor

    def _get_losses(self, task, transitions):
        if task == 'sym_state':
            transitions, inputs_norm_tensor, inputs_next_norm_tensor, actions_tensor, r_tensor = self._prepare_inputs_for_state_only(transitions)
            with torch.no_grad():
                actions_next = self.actor_target_network(inputs_next_norm_tensor)
                q_next_value = self.critic_target_network(inputs_next_norm_tensor, actions_next)
                q_next_value = q_next_value.detach()
                target_q_value = r_tensor + self.args.gamma * q_next_value
                target_q_value = target_q_value.detach()
                clip_return = 1 / (1 - self.args.gamma)
                target_q_value = torch.clamp(target_q_value, -clip_return, 0)
            real_q_value = self.critic_network(inputs_norm_tensor, actions_tensor)
            critic_loss = (target_q_value - real_q_value).pow(2).mean()
            actions_real = self.actor_network(inputs_norm_tensor)
            critic_loss = (target_q_value - real_q_value).pow(2).mean()
            actor_loss = -self.critic_network(inputs_norm_tensor, actions_real).mean()
            actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()
            return actor_loss, critic_loss
        elif task == 'asym_goal_outside_image':
            transitions, inputs_norm_tensor, inputs_next_norm_tensor, actions_tensor, r_tensor = self._prepare_inputs_for_state_only(transitions)
            tensor_img, tensor_g = self._preproc_inputs_image(transitions['obs_imgs_next'], transitions['goal_states_next'])
            with torch.no_grad():
                actions_next = self.actor_target_network(tensor_img, tensor_g)
                q_next_value = self.critic_target_network(inputs_next_norm_tensor, actions_next)
                q_next_value = q_next_value.detach()
                target_q_value = r_tensor + self.args.gamma * q_next_value
                target_q_value = target_q_value.detach()
                clip_return = 1 / (1 - self.args.gamma)
                target_q_value = torch.clamp(target_q_value, -clip_return, 0)
            real_q_value = self.critic_network(inputs_norm_tensor, actions_tensor)
            critic_loss = (target_q_value - real_q_value).pow(2).mean()
            tensor_img, tensor_g = self._preproc_inputs_image(transitions['obs_imgs'], transitions['goal_states'])
            actions_real = self.actor_network(tensor_img, tensor_g)
            critic_loss = (target_q_value - real_q_value).pow(2).mean()
            actor_loss = -self.critic_network(inputs_norm_tensor, actions_real).mean()
            actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()
            return actor_loss, critic_loss
        elif task == 'asym_goal_in_image':
            raise NotImplementedError
        elif task == 'sym_image':
            tensor_img =  transitions['obs_imgs']
            tensor_img = torch.tensor(tensor_img.copy()).to(torch.float32)
            tensor_img = tensor_img.permute(0, 3, 1, 2)

            tensor_img_next = transitions['next_obs_imgs']
            tensor_img_next = torch.tensor(tensor_img_next.copy()).to(torch.float32)
            tensor_img_next = tensor_img_next.permute(0, 3, 1, 2)

            actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
            r_tensor = torch.tensor(transitions['rewards'], dtype=torch.float32)

            if self.args.cuda:
                tensor_img_next = tensor_img_next.cuda(MPI.COMM_WORLD.Get_rank())
                tensor_img = tensor_img.cuda(MPI.COMM_WORLD.Get_rank())
                actions_tensor = actions_tensor.cuda(MPI.COMM_WORLD.Get_rank())
                r_tensor = r_tensor.cuda(MPI.COMM_WORLD.Get_rank())


            with torch.no_grad():
                actions_next = self.actor_target_network(tensor_img_next)
                q_next_value = self.critic_target_network(tensor_img_next, actions_next)
                q_next_value = q_next_value.detach()
                target_q_value = r_tensor + self.args.gamma * q_next_value
                target_q_value = target_q_value.detach()
                clip_return = 1 / (1 - self.args.gamma)
                target_q_value = torch.clamp(target_q_value, -clip_return, 0)
            real_q_value = self.critic_network(tensor_img, actions_tensor)
            critic_loss = (target_q_value - real_q_value).pow(2).mean()
            actions_real = self.actor_network(tensor_img)
            critic_loss = (target_q_value - real_q_value).pow(2).mean()
            actor_loss = -self.critic_network(tensor_img, actions_real).mean()
            actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()
            return actor_loss, critic_loss

    # update the network
    def _update_network(self):
        # sample the episodes
        transitions = self.buffer.sample(self.args.batch_size)
        # calculate the target Q value function
        actor_loss, critic_loss = self._get_losses(self.args.task, transitions)
        # start to update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        # plot_grad_flow(self.actor_network.named_parameters())
        sync_grads(self.actor_network)
        self.actor_optim.step()
        # update the critic_network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic_network)
        self.critic_optim.step()
        if MPI.COMM_WORLD.Get_rank() == 0:
            self.critic_loss.append(critic_loss.item())
            self.actor_loss.append(actor_loss.item())
            # print("Critic Loss {} | Actor Loss {}".format(critic_loss.item(), actor_loss.item()))

    # do the evaluation
    def _eval_agent(self):
        total_success_rate = []
        for _ in range(self.args.n_test_rollouts):
            per_success_rate = []
            observation = self.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            obs_img = self.env.render(mode="rgb_array", height=100, width=100)
            observation['observation_image'] = obs_img
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    pi = self.get_policy(self.args.task, observation)
                    actions = pi.detach().cpu().numpy().squeeze()
                observation_new, _, _, info = self.env.step(actions)
                obs_img = self.env.render(mode="rgb_array", height=100, width=100)
                observation_new['observation_image'] = obs_img
                observation = observation_new
                per_success_rate.append(info['is_success'])
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        plt.plot(self.actor_loss, color="red", label="Actor Loss")
        plt.plot(self.critic_loss, color="blue", label="Critic Loss")
        plt.xlabel('num steps')
        plt.ylabel('loss value')
        plt.legend()
        plt.savefig('loss_plot.png')
        plt.clf()
        return global_success_rate / MPI.COMM_WORLD.Get_size()
