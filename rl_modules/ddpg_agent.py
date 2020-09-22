import torch
import os
from datetime import datetime
import numpy as np
from mpi4py import MPI
from mpi_utils.mpi_utils import sync_networks, sync_grads
from rl_modules.replay_buffer import replay_buffer
from rl_modules.models import actor, critic, img_actor, new_actor
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

"""
ddpg with HER (MPI-version)
"""
class ddpg_agent:
    def __init__(self, args, env, env_params):
        self.args = args
        self.env = env
        sim = self.env.sim
        self.viewer = MjRenderContextOffscreen(sim)
        self.viewer.cam.fixedcamid = 3
        self.viewer.cam.type = const.CAMERA_FIXED
        self.critic_loss = []
        self.actor_loss = []
        env.env._viewers['rgb_array'] = self.viewer

        self.env_params = env_params
        self.image_based = True
        # create the network
        if not self.image_based:
            self.actor_network = actor(env_params)
        else:
            self.actor_network = new_actor(env_params)
        self.critic_network = critic(env_params)

        # sync the networks across the cpus
        sync_networks(self.actor_network)
        sync_networks(self.critic_network)
        # build up the target network
        if not self.image_based:
            self.actor_target_network = actor(env_params)
        else:
            self.actor_target_network = new_actor(env_params)

        self.critic_target_network = critic(env_params)
        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        # if use gpu
        if self.args.cuda:
            print("use the GPU")
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
                            self.image_based)
        # create the replay buffer
        self.buffer = replay_buffer(self.env_params, 
                                    self.args.buffer_size, 
                                    self.her_module.sample_her_transitions,
                                    self.image_based)
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

    def learn(self):
        """
        train the network

        """
        # start to collect samples
        for epoch in range(self.args.n_epochs):
            for _ in range(self.args.n_cycles):
                mb_obs, mb_ag, mb_g, mb_actions, mb_img_obs = [], [], [], [], []
                for _ in range(self.args.num_rollouts_per_mpi):
                    # reset the rollouts
                    ep_obs, ep_ag, ep_g, ep_actions, ep_img_obs = [], [], [], [], []
                    # reset the environment
                    observation = self.env.reset()
                    obs = observation['observation']

                    if self.image_based:
                        obs_img = self.env.render(mode="rgb_array", height=100, width=100)
                        # plt.imshow(obs_img)
                        # plt.savefig('image_observation.png')
                        # plt.show()
                        # exit()

                    ag = observation['achieved_goal']
                    g = observation['desired_goal']
                    # start to collect samples
                    for t in range(self.env_params['max_timesteps']):
                        with torch.no_grad():
                            if not self.image_based:
                                input_tensor = self._preproc_inputs(obs, g)
                                pi = self.actor_network(input_tensor)
                            else:
                                o_tensor, g_tensor = self._preproc_inputs_image(obs_img.copy()[np.newaxis, :], g[np.newaxis, :])
                                pi = self.actor_network(o_tensor, g_tensor)
                            action = self._select_actions(pi)
                        # feed the actions into the environment
                        observation_new, _, _, info = self.env.step(action)
                        obs_new = observation_new['observation']

                        if self.image_based:
                            obs_image_new = self.env.render(mode="rgb_array", height=100, width=100)
                            ep_img_obs.append(obs_img.copy())

                        ag_new = observation_new['achieved_goal']
                        # append rollouts
                        ep_obs.append(obs.copy())
                        ep_ag.append(ag.copy())
                        ep_g.append(g.copy())
                        ep_actions.append(action.copy())
                        # re-assign the observation
                        obs = obs_new
                        ag = ag_new

                        if self.image_based:
                            obs_img = obs_image_new

                    ep_obs.append(obs.copy())
                    ep_ag.append(ag.copy())
                    if self.image_based:
                        ep_img_obs.append(obs_img.copy())
                    mb_obs.append(ep_obs)
                    mb_ag.append(ep_ag)
                    mb_g.append(ep_g)
                    mb_actions.append(ep_actions)
                    if self.image_based:
                        mb_img_obs.append(ep_img_obs)
                # convert them into arrays
                mb_obs = np.array(mb_obs)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                mb_actions = np.array(mb_actions)
                if self.image_based:    
                    mb_img_obs = np.array(mb_img_obs)
                # store the episodes
                if self.image_based:
                    self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions, mb_img_obs])
                    self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions, mb_img_obs])
                else:
                    self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
                    self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])

                for _ in range(self.args.n_batches):
                    # train the network
                    self._update_network()
                # soft update
                self._soft_update_target_network(self.actor_target_network, self.actor_network)
                self._soft_update_target_network(self.critic_target_network, self.critic_network)
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
        if self.image_based:
            mb_obs, mb_ag, mb_g, mb_actions, mb_obs_img = episode_batch
            mb_obs_img_next = mb_obs_img[:, 1:, :]
        else:
            mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        if self.image_based:
            buffer_temp = {'obs': mb_obs, 
                'obs_img': mb_obs_img,
                'ag': mb_ag,
                'g': mb_g, 
                'actions': mb_actions, 
                'obs_next': mb_obs_next,
                'obs_img_next': mb_obs_img_next,
                'ag_next': mb_ag_next,
                'g_o': mb_obs_img
            }
        else:
            buffer_temp = {'obs': mb_obs, 
                'ag': mb_ag,
                'g': mb_g, 
                'actions': mb_actions, 
                'obs_next': mb_obs_next,
                'ag_next': mb_ag_next,
            }
        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # update
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
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

    # update the network
    def _update_network(self):
        # sample the episodes
        transitions = self.buffer.sample(self.args.batch_size)
        # pre-process the observation and goal
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)
        # start to do the update
        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
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
        # calculate the target Q value function
        with torch.no_grad():
            # do the normalization
            # concatenate the stuffs

            if not self.image_based:
                actions_next = self.actor_target_network(inputs_next_norm_tensor)
            else:
                tensor_img, tensor_g = self._preproc_inputs_image(transitions['obs_img_next'], transitions['g_next'])
                actions_next = self.actor_target_network(tensor_img, tensor_g)

            q_next_value = self.critic_target_network(inputs_next_norm_tensor, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + self.args.gamma * q_next_value
            target_q_value = target_q_value.detach()
            # clip the q value
            clip_return = 1 / (1 - self.args.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)
        # the q loss
        real_q_value = self.critic_network(inputs_norm_tensor, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()
        # the actor loss
        if not self.image_based:
            actions_real = self.actor_network(inputs_norm_tensor)
        else:
            tensor_img, tensor_g = self._preproc_inputs_image(transitions['obs_img'], transitions['g'])
            actions_real = self.actor_network(tensor_img, tensor_g)
            

        actor_loss = -self.critic_network(inputs_norm_tensor, actions_real).mean()
        actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()
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
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    if self.image_based:
                        o_tensor, g_tensor = self._preproc_inputs_image(obs_img.copy()[np.newaxis, :], g[np.newaxis, :])
                        pi = self.actor_network(o_tensor, g_tensor)
                    else:
                        input_tensor = self._preproc_inputs(obs, g)
                        pi = self.actor_network(input_tensor)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()
                observation_new, _, _, info = self.env.step(actions)
                obs = observation_new['observation']
                obs_img = self.env.render(mode="rgb_array", height=100, width=100)
                g = observation_new['desired_goal']
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
