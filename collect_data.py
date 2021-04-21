import gym
from xarm_env.pick_and_place import PickAndPlaceXarm
import pickle
from os.path import join
import datetime
from torch.utils.data import Dataset, DataLoader
import os 
from rl_modules.ddpg_agent import model_factory
from rl_modules.utils import get_env_params, _preproc_inputs_image_goal, display_state, load_viewer_to_env
from arguments import get_args
import torch
import torch.nn.functional as F
import numpy as np
from eval_agent import eval_agent_and_save

env = PickAndPlaceXarm(xml_path='./assets/fetch/pick_and_place_xarm.xml')
env = load_viewer_to_env(env)

num_episodes = 10
height = 100
width = 100

paths = {
    'FetchReach-v1': {
        'old_robo': {
            'sym_state': './all_weigths/FetchReach-v1/',
            'asym_goal_outside_image': './randomized_server_weights/asym_goal_outside_image/FetchReach-v1',
            'asym_goal_in_image': 'sym_server_weights/distill/',
            'sym_image': ''
        },
        'xarm': {
            'asym_goal_in_image': './sym_server_weights/saved_models/asym_goal_in_image/FetchReach-v1',
            'asym_goal_outside_image': './sym_server_weights/saved_models/asym_goal_outside_image/FetchReach-v1'
        }
    },
    'FetchPush-v1': {
        'sym_state': '',
        'asym_goal_in_image': 'sym_server_weights/saved_models/distill/image_only/',
        'asym_goal_outside_image': './sym_server_weights/asym_goal_outside_image_distill/FetchPush-v1/',
    },
    'FetchSlide-v1': {

    },
    'FetchPickAndPlace-v1': {
        'xarm': {
            'asym_goal_outside_image': './sym_server_weights/saved_models/asym_goal_outside_image/FetchPickAndPlace-v1',
            'sym_state': './sym_server_weights/saved_models/sym_state/FetchPickAndPlace-v1',
            'asym_goal_outside_image_distill': './sym_server_weights/saved_models/asym_goal_outside_image_distill/FetchPickAndPlace-v1',
        }
    }
}

def get_policy(model, obs, args, is_np=True):
    obs, g_norm, state_input = _preproc_inputs_image_goal(obs, args, is_np)
    return model(state_input).detach().cpu().numpy().squeeze()


def save_image(obs, parent_path='./offline_dataset'):
    pickle.dump(obs, open(join(parent_path, str(datetime.datetime.now())), "wb"))
    print(f"file saved to {join(parent_path, str(datetime.datetime.now()))}")


def generate_dataset(state_based_model, obj, args, parent_path='./offline_dataset'):
    # Deleting dataset folder
    os.system(f"rm -rf {parent_path}")
    # creating dataset folder
    os.system(f"mkdir {parent_path}")

    for i in range(num_episodes):
        obs = env.reset()
        for i in range(50):
            rgb, dep = env.render(mode='rgb_array', height=height, width=width, depth=True)

            # sampling policy used saved model ?
            obs["rgb"] = rgb
            obs["dep"] = dep
            obs["obj"] = obj
            actions = get_policy(state_based_model, obs, args)
            obs["actions"] = actions

            # step actions
            new_obs, rew, _ , _ = env.step(actions)

            save_image(obs)

            display_state(obs)
            obs = new_obs



class OfflineDataset(Dataset):

    def __init__(self, parent_path='./offline_dataset'):
        self.parent_path = parent_path
        self.files = os.listdir(parent_path)


    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = join(self.parent_path, self.files[idx])
        obj = pickle.load(open(file_path, "rb"))
        return obj

def get_offline_dataset(batch_size=512):
    dt = OfflineDataset()
    dt_loader = DataLoader(dataset=dt, batch_size=batch_size, shuffle=True, num_workers=4)

    # for obj in dt_loader:
    #     rgb = obj["rgb"]
    #     dep = obj["depth"]
    #     obs = obj["obs"]
    return dt_loader



def bc_train(env):
    args = get_args()
    
    env_params = get_env_params(env)
    env_params["load_saved"] = True
    env_params["model_path"] = paths[args.env_name]['xarm'][args.task] + '/model.pt'

    args.cuda = False
    state_based_model, _, _, _ = model_factory(task='sym_state', env_params=env_params)
    env_params["depth"] = args.depth
    env_params["load_saved"] = False
    student_model, _, _, _ = model_factory(task='asym_goal_outside_image', env_params=env_params)

    obj = torch.load(env_params["model_path"], map_location=torch.device('cpu'))

    o_mean = obj["o_mean"]
    o_std = obj["o_std"]
    g_mean = obj["g_mean"]
    g_std = obj["g_std"]

    # generate_dataset(state_based_model, obj, args)
    # exit()

    dt_loader = get_offline_dataset()


    optimizer = torch.optim.Adam(params=student_model.parameters(), lr=0.001)

    num_epochs = 100
    losses = []
    rewards = []
    best_succ_rate = 0

    for ep in range(num_epochs):
        eval_agent_and_save(ep, env, args, student_model, obj, task='asym_goal_outside_image')
        # TODO:
        #add epoch init stuff here

        for dt in dt_loader:

            # TODO:

            # get data

            # run through model

            # compute the loss

            # optimize the loss

            dt["obj"] = obj
            obs_state = dt["observation"]
            g = dt["desired_goal"].to(torch.float32)

            # TODO normalize
            obs_img, g_norm, state_based_input = _preproc_inputs_image_goal(dt, args, is_np=False)

            inp = torch.cat((obs_state, g), 1)
            inp = inp.to(torch.float32)

            # run through model
            with torch.no_grad():
                acts = state_based_model(state_based_input)

            student_acts = student_model(obs_img, g_norm)
            # compute the loss
            loss = F.mse_loss(student_acts, acts)

            # step the loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            # TODO: add plotting for training
            losses.append(loss.item())
        


        if ep % 10 == 0:
            # save video of agent
            args.record = True
        else:
            args.record = False
        
        succ_rate = eval_agent_and_save(ep, env, args, student_model, obj, task='asym_goal_outside_image')
        rewards.append(succ_rate)
        save_dict = {
            'actor_net': student_model.state_dict(),
            'o_mean': obj["o_mean"],
            'o_std': obj["o_std"],
            'g_mean': obj["g_mean"],
            'g_std': obj["g_std"],
            'reward_plots': rewards,
            'losses': losses,
        }
        if succ_rate >= best_succ_rate:
            torch.save(save_dict, "best_bc_model.pt")
            best_succ_rate = succ_rate
        else:
            torch.save(save_dict, "curr_bc_model.pt")

        # eval_agent_and_save(env, student_model, save_record)

            
            
    # for i in range(10):
    #     obs = env.reset()
    #     for ep in range(50):
    #         rgb, dep = env.render(mode='rgb_array', height=height, width=width, depth=True)
    #         obs["rgb"] = rgb
    #         obs["dep"] = dep
    #         obs["obj"] = obj
    #         actions = get_policy(state_based_model, obs)
    #         new_obs, rew, _ , _ = env.step(actions)

    #         display_state(obs)
    #         obs = new_obs


bc_train(env)
