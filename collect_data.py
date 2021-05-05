import gym
from xarm_env.pick_and_place import PickAndPlaceXarm
import pickle
from os.path import join
import datetime
from torch.utils.data import Dataset, DataLoader
import os 
from rl_modules.ddpg_agent import model_factory
from rl_modules.utils import get_env_params, _preproc_inputs_image_goal, display_state, load_viewer_to_env, scripted_action, show_video
from arguments import get_args
import torch
import torch.nn.functional as F
import numpy as np
import copy
from eval_agent import eval_agent_and_save
from torch.optim.lr_scheduler import ReduceLROnPlateau
from rl_modules.utils import plot_grad_flow
import time 
import cv2
import matplotlib.pyplot as plt
from mpi4py import MPI

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

def plot_model_stats(obj):
    print(obj.keys())
    rew_asym = obj["reward_plots"]
    # two = len(obj['losses']) / len(obj['reward_plots'])
    # plt.plot(np.arange(len(rew_asym)), rew_asym, color='red')
    # plt.plot(np.arange(len(obj["losses"][10:])), obj["losses"][10:], color='red')
    plt.plot(np.arange(len(obj['reward_plots'])), obj['reward_plots'], color='blue')
    plt.show()
    exit()

def get_policy(model, obs, args, is_np=True):
    obs, g_norm, state_input = _preproc_inputs_image_goal(obs, args, is_np)
    return model(state_input).detach().cpu().numpy().squeeze()


def save_image(j, obs, parent_path):
    outfile = join(parent_path, str(datetime.datetime.now()))
    np.save(outfile, obs)
    print(f"{j} file saved to {outfile}")


def generate_dataset(state_based_model, obj, args):
    parent_path = args.bc_dataset_path
    # Deleting dataset folder
    os.system(f"rm -rf {parent_path}")
    # creating dataset folder
    os.system(f"mkdir {parent_path}")
    num_episodes = 2000
    for j in range(num_episodes):
        obs = env.reset()
        picked_object = False
        for i in range(100): # more needed for pick and place
            rgb, dep = env.render(mode='rgb_array', height=height, width=width, depth=True)

            # sampling policy used saved model ?
            obs["rgb"] = rgb
            obs["dep"] = dep
            if args.scripted:
                actions, picked_object = scripted_action(obs, picked_object)
            else:
                obs["obj"] = obj
                actions = get_policy(state_based_model, obs, args)
                del obs['obj']
            obs["actions"] = actions

            # step actions
            new_obs, rew, _ , _ = env.step(actions)
        
            save_image(j, obs, parent_path)

            #display_state(obs)
            obs = new_obs


class OfflineDataset(Dataset):

    def __init__(self, parent_path):
        self.parent_path = parent_path
        self.files = os.listdir(parent_path)
        self.all_objs = []
        # for i, f_path in enumerate(self.files):
        #     file_path = join(self.parent_path, f_path)
        #     with open(file_path, "rb") as f:
        #         obj = np.load(f, allow_pickle=True)
        #     self.all_objs.append(obj)
        #     print(i)



    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # file_path = join(self.parent_path, self.files[idx])
        # f = open(file_path, "rb")
        # # with open(file_path, "rb") as f:
        # obj = pickle.load(f)
        # obj = copy.deepcopy(obj)
        # f.close()
        # print(obj.keys())
        # print(type(obj["rgb"]))
        # obj["rgb"] = torch.from_numpy(obj["rgb"])
        # obj["dep"] = torch.from_numpy(obj["dep"])
        # obj["actions"] = torch.from_numpy(obj["actions"])
        # obj["desired_goal"] = torch.from_numpy(obj["desired_goal"])
        # obj["achieved_goal"] = torch.from_numpy(obj["achieved_goal"])
        # obj["observation"] = torch.from_numpy(obj["observation"])
        # obj = torch.randn((1,2,3))
        f_path = self.files[idx]
        file_path = join(self.parent_path, f_path)
        # print(file_path)
        with open(file_path, "rb") as f:
            obj = np.load(f, allow_pickle=True)
        d = obj[()]
        d["actions"] = d["actions"].astype(np.float32)
        # print(d.items())
        return d
        
        # return torch.randn((1,400,400))

def get_offline_dataset(args):
    dt = OfflineDataset(parent_path=args.bc_dataset_path)
    dt_loader = DataLoader(dataset=dt, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # for obj in dt_loader:
    #     rgb = obj["rgb"]
    #     dep = obj["depth"]
    #     obs = obj["obs"]
    return dt_loader


def bc_train(env):
    args = get_args()

    env_params = get_env_params(env)
    env_params["load_saved"] = False
    env_params["model_path"] = paths[args.env_name]['xarm']['sym_state'] + '/model.pt'

    # if not args.scripted:
    #args.cuda = True
    state_based_model, _, _, _ = model_factory(task='sym_state', env_params=env_params)
    env_params["depth"] = args.depth
    env_params["load_saved"] = False

    student_model, _, _, _ = model_factory(task=args.task, env_params=env_params)
    print(student_model)

    obj = torch.load(env_params["model_path"], map_location=torch.device('cpu'))

    o_mean = obj["o_mean"]
    o_std = obj["o_std"]
    g_mean = obj["g_mean"]
    g_std = obj["g_std"]

    dt_loader = get_offline_dataset(args)

    optimizer = torch.optim.SGD(params=student_model.parameters(), lr=0.01)

    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True)

    num_epochs = 500
    losses = []
    rewards = []
    best_succ_rate = 0

    if args.cuda:
        student_model = student_model.cuda(MPI.COMM_WORLD.Get_rank())
        if not args.scripted:
            state_based_model = state_based_model.cuda(MPI.COMM_WORLD.Get_rank())
    
    if args.just_eval:
        # model_path = 'best_bc_model.pt'
        # model_path = 'curr_img_model.pt'
        model_path = 'curr_rgb_model.pt'
        obj = torch.load(model_path, map_location=lambda storage, loc: storage)
        student_model.load_state_dict(obj['actor_net'])

    student_model.train()
    
    # plot_model_stats(obj)

    print("start training")
    for ep in range(num_epochs):
        total_loss = 0
        if args.just_eval:
            succ_rates = []
            for i in range(20):
                succ_rate = eval_agent_and_save(ep, env, args, student_model, obj, task=args.task)
                succ_rates.append(succ_rate)
            plt.plot(np.arange(len(succ_rates)), succ_rates, color='red')
            plt.show()
            exit()

        # TODO:
        #add epoch init stuff here
        start = time.time()
        for idx, dt in enumerate(dt_loader):

            dt["obj"] = obj
            obs_state = dt["observation"]
            g = dt["desired_goal"].to(torch.float32)

            # TODO normalize
            obs_img, g_norm, state_based_input = _preproc_inputs_image_goal(dt, args, is_np=False)

            # print(obs_img.shape)
            # obs_img = obs_img.permute(0,2,3,1).numpy()[0].astype(np.uint8)
            # show_video(obs_img)
            # continue

            # run through model
            if args.scripted:
                with torch.no_grad():
                    acts = dt["actions"].clone().detach()
                    if args.cuda:
                        acts = acts.cuda(MPI.COMM_WORLD.Get_rank())
            else:
                with torch.no_grad():
                    acts = state_based_model(state_based_input)

            # zero_inp = torch.zeros_like(obs_img)
            # zero_g = torch.zeros_like(g_norm)
            # z_s_b = torch.zeros_like(state_based_input)
            # print(state_based_input)
            print(obs_img.shape, g_norm.shape)
            if args.task != 'sym_state':
                student_acts = student_model(obs_img, g_norm)
            else:
                student_acts = student_model(state_based_input)
            # print(acts, student_acts)
            # compute the loss
            loss = F.mse_loss(student_acts, acts)

            # step the loss
            
            optimizer.zero_grad()
            loss.backward()
            # plot_grad_flow(state_based_model.named_parameters())
            optimizer.step()

            total_loss += loss.item()
            

            # TODO: add plotting for training
            losses.append(loss.item())
            
        print(total_loss)
        scheduler.step(total_loss)
         
        end = time.time()
        
        # run after every epoch

        print(f"Epoch {ep} | Total time taken {end - start} | Loss {total_loss / len(dt_loader)}")


        if ep % 10 == 0:
            # save video of agent
            args.record = True
        else:
            args.record = False
        
        succ_rate = eval_agent_and_save(ep, env, args, student_model, obj, task=args.task)
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
        exit()
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


def dagger():
    env = PickAndPlaceXarm(xml_path='./assets/fetch/pick_and_place_xarm.xml')

    env = load_viewer_to_env(env)
    modder = get_texture_modder(env)

    num_steps = 50
    while True:
        obs = env.reset()
        pick_object = False
        r = -1
        j = 0
        is_succ = 0
        k = 0
        ended = True
        m = 0
        distances = []
        while (not is_succ) or k < 50:
            j += 1
            # rgb, depth = env.render(mode='rgb_array', depth=True, height=100, width=100)
            # rgb, depth = use_real_depths_and_crop(rgb, depth)
            # show_video(rgb)
            # print(obs["observation"][:3])
            
            act, pick_object = scripted_action(obs, pick_object)
            
            env.render()
            if ended or obs["observation"][0] < 1.1:
                ended = True
                act, pick_object = scripted_action(obs, pick_object) 
            else:
                act = np.asarray([-1, 0, 0, 0])

            obs,  r, _, infor = env.step(act)
            
            left_gripper = obs['observation'][:3]
            right_gripper = obs['observation'][-3:]
            distances.append(abs(right_gripper[1] - left_gripper[1]))
            # print(abs(right_gripper[1] - left_gripper[1]))
            if infor['is_success'] != 1:
                k = 0
            else:
                k += 1

            is_succ = infor['is_success']

            m += 1

            if out_of_bounds(obs):
                print("object out of bounds, ending episode...")
                break

            show_big_ball(env, obs['observation'][-3:])
            
            if m > 500:
                print("episode is too long, breaking...")
                break

        # plt.plot(np.arange(len(distances)), distances, color='red')
        # plt.show()
        print(is_succ)


def create_off_dataset():
    args = get_args()

    generate_dataset(None, None, args)

bc_train(env)
# create_off_dataset()
