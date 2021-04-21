import numpy as np
import torch
from pcd_utils import display_interactive_point_cloud
from rl_modules.utils import use_real_depths_and_crop
from mujoco_py.modder import TextureModder, MaterialModder, CameraModder, LightModder
from depth_tricks import create_point_cloud, create_point_cloud2
import cv2 
import imageio

def eval_agent_and_save(ep, env, args, loaded_model, obj, task):

    
    modder = TextureModder(env.sim)

    total_success_rate = []
    rollouts = []

    def _preproc_inputs_image_goal(obs_img, g, depth=None):
        if args.depth:
            # add depth observation
            obs_img = obs_img.squeeze(0)
            obs_img = obs_img.astype(np.float32)
            # obs_img = obs_img / 255 # normalize image data between 0 and 1
            obs_img, depth = use_real_depths_and_crop(obs_img, depth)
            obs_img = np.concatenate((obs_img, depth), axis=2)
            obs_img = torch.tensor(obs_img, dtype=torch.float32).unsqueeze(0)
            obs_img = obs_img.permute(0, 3, 1, 2)
        else:
            obs_img = torch.tensor(obs_img, dtype=torch.float32)
            obs_img = obs_img.permute(0, 3, 1, 2)
        
     
        g = np.clip((g - obj['g_mean'])/obj['g_std'], -args.clip_range, args.clip_range)
        g_norm = torch.tensor(g, dtype=torch.float32)
        # g_norm = torch.zeros((1, 3))
        if args.cuda:
            obs_img = obs_img.cuda(MPI.COMM_WORLD.Get_rank())
            g_norm = g_norm.cuda(MPI.COMM_WORLD.Get_rank())
        return obs_img, g_norm
    
    def _prepoc_image(obs_img):
        obs_img = torch.tensor(obs_img, dtype=torch.float32)
        obs_img = obs_img.permute(0, 3, 1, 2)
        if args.cuda:
            obs_img = obs_img.cuda(MPI.COMM_WORLD.Get_rank())
        return obs_img

    # pre_process the inputs
    def _preproc_inputs_state(obs, g):
        print(obs.shape, obj['o_mean'].shape)
        obs_norm = np.clip((obs - obj['o_mean'])/obj['o_std'], -args.clip_range, args.clip_range).reshape(1,-1)
        g_norm = np.clip((g - obj['g_mean'])/obj['g_std'], -args.clip_range, args.clip_range)
        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, g_norm], axis=1)
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        return inputs

    def get_policy(obs_img, g, obs=None, depth=None):
        if task == "sym_state":
            inputs = _preproc_inputs_state(obs, g)
            pi = loaded_model(inputs)
            return pi
        if task == "asym_goal_outside_image":
            o_tensor, g_tensor = _preproc_inputs_image_goal(obs_img, g, depth)
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
    
    def create_folder_and_save(obj, folder_name='rollout_records'):            
        Path(folder_name).mkdir(parents=True, exist_ok=True)
        timestamp =  str(datetime.datetime.now())
        path_name = os.path.join(folder_name, timestamp)

        torch.save(obj, path_name)
        print(f"Trajectory saved to {path_name}")

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
        for _ in range(max_steps):
            
            if args.randomize:
                # randomize_camera(viewer)
                randomize_textures(modder, env.sim)
            
            obs_img, depth_image = env.render(mode="rgb_array", height=100, width=100, depth=True)
            save_obs_img, save_depth_image = use_real_depths_and_crop(obs_img, depth_image)

            display_state(obs_img)
            
            pcd = create_point_cloud(save_obs_img, save_depth_image, fovy=45)
            pcds.append(("none", pcd))
          
            if args.depth:
                # create_point_cloud(env, dep_img=depth_image, col_img=obs_img)
                pi = get_policy(obs_img.copy()[np.newaxis, :], g[np.newaxis, :], depth=depth_image[:, :, np.newaxis])
                actions = pi.detach().cpu().numpy().squeeze()
            else:
                with torch.no_grad():
                    if task != 'sym_state':
                        pi = get_policy(obs_img.copy()[np.newaxis, :], g[np.newaxis, :])
                    else:
                        pi = get_policy(obs_img=None, g=g[np.newaxis, :], obs=observation["observation"])
                    actions = pi.detach().cpu().numpy().squeeze()

            observation_new, _, _, info = env.step(actions)
            rollout.append({
                'obs_img': save_obs_img,
                'depth_img': save_depth_image,
                'actions': actions,
            })

            obs = observation_new['observation']
            
            g = observation_new['desired_goal']
            observation = observation_new
            per_success_rate.append(info['is_success'])

        # # hide under a flag
        # if args.record:
        #     frames = [r["obs_img"] for r in rollout]
        #     path = "temp.mp4"
        #     imageio.mimsave(path, frames, fps=30)
        #     create_folder_and_save({'traj': rollout, 'goal': g})

        
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
