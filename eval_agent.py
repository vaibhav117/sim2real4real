from gym.core import ObservationWrapper
import numpy as np
import torch
from pcd_utils import display_interactive_point_cloud
from rl_modules.utils import use_real_depths_and_crop
from mujoco_py.modder import TextureModder, MaterialModder, CameraModder, LightModder
from depth_tricks import create_point_cloud, create_point_cloud2
import cv2 
import imageio
from rl_modules.utils import show_video, scripted_action, use_real_depths_and_crop, scripted_action_new
from mpi4py import MPI
from rl_modules.utils import _preproc_inputs_state, _preproc_inputs_image_goal, _preproc_image
import time

def eval_agent_and_save(ep, env, args, loaded_model, obj, task):
    
    modder = TextureModder(env.sim)

    total_success_rate = []
    rollouts = []

    def get_policy(obs, args, is_np):
        if task == "sym_state":
            inputs = _preproc_inputs_state(obs, g)
            if args.cuda:
                inputs = inputs.cuda(MPI.COMM_WORLD.Get_rank())
            pi = loaded_model(inputs)
            return pi
        if task == "asym_goal_outside_image":
            o_tensor, g_tensor, _ = _preproc_inputs_image_goal(obs, args, is_np)
            g_tensor = g_tensor.squeeze(1)
            # g_tensor = torch.tensor(np.asarray([0.2, 0.2, 0.2])).view(1, -1).to(torch.float32)
            pi = loaded_model(o_tensor, g_tensor)
            return pi
        if task == "asym_goal_in_image":
            pi = loaded_model(_prepoc_image(obs_img))
            return pi
        if task == "sym_image":
            o_tensor, _ = _preproc_inputs_image_goal(obs_img, g)
            pi = loaded_model(o_tensor)
            return pi
    

    def display_state(rgb):
        """
        Display state
        """
        cv2.imshow("frame", rgb)
        cv2.waitKey(1)

    for _ in range(args.n_test_rollouts):
        per_success_rate = []
        observation = env.reset()
        obs = observation['observation']
        g = observation['desired_goal']
        rollout = []
        all_info = []
        pcds = []
        
        if args.randomize:
            randomize_textures(modder, env.sim)
            # randomize_camera(viewer)
        
        max_steps = env._max_episode_steps
        max_steps = 100
        picked_object = False # only use for scripted policy
        for _ in range(max_steps):
            
            if args.randomize:
                # randomize_camera(viewer)
                randomize_textures(modder, env.sim)
            
            obs_img, depth_image = env.render(mode="rgb_array", height=100, width=100, depth=True)
            obs_img, depth_image = use_real_depths_and_crop(obs_img, depth_image)
            
            show_video(obs_img)

            observation["rgb"] = obs_img
            observation["dep"] = depth_image
            observation["obj"] = obj
            with torch.no_grad():
                # create_point_cloud(env, dep_img=depth_image, col_img=obs_img)
                pi = get_policy(observation, args, is_np=True)
                actions = pi.detach().cpu().numpy().squeeze()
                # actions, picked_object = scripted_action_new(observation, picked_object)

            #if args.scripted:
            #    actions, picked_object = scripted_action(observation, picked_object=picked_object)

            observation_new, _, _, info = env.step(actions)
            rollout.append({
                'obs_img': obs_img,
                'depth_img': depth_image,
                'actions': actions,
            })

            observation = observation_new
            per_success_rate.append(info['is_success'])

        
        total_success_rate.append(per_success_rate)
        rollouts.append(rollout)
        
    total_success_rate = np.array(total_success_rate)
    local_success_rate = np.mean(total_success_rate[:, -1])

    if args.record:
        # save trajectory
        frames = []
        for rollout in rollouts:
            frames += [r["obs_img"] for r in rollout]
        path = "temp.mp4"
        imageio.mimsave(path, frames, fps=30)
    print(f"Epoch {ep}: Success rate {local_success_rate}")
    return local_success_rate
