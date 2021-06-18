import gym
from xarm_env.pick_and_place import PickAndPlaceXarm
import pickle
from os.path import join
import datetime
from torch.utils.data import Dataset, DataLoader
import os 
from rl_modules.ddpg_agent import model_factory, randomize_camera
from rl_modules.utils import get_env_params, _preproc_inputs_image_goal, display_state, load_viewer_to_env, scripted_action, scripted_action_new, show_video, randomize_textures, get_texture_modder, out_of_bounds, use_real_depths_and_crop, get_viewer
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
from arguments import get_args
from stupid_net import StupidNet
import random
from pcd_utils import visualize, return_pcd

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
            'sym_state': './saved_models/sym_state/FetchPickAndPlace-v1',
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


def save_image(j, obs, parent_path, verbose=False):
    outfile = join(parent_path, str(datetime.datetime.now()))
    np.save(outfile, obs)
    if verbose:
        print(f"{j} file saved to {outfile}")


def check_if_dataset_folder_exists(args):
    parent_path = args.bc_dataset_path
    # Deleting dataset folder
    os.system(f"rm -rf {parent_path}")
    # creating dataset folder
    os.system(f"mkdir {parent_path}")

def generate_dataset(sc_policy, env, args):
    height = 100
    width = 100
    parent_path = args.bc_dataset_path
    # Deleting dataset folder
    os.system(f"rm -rf {parent_path}")
    # creating dataset folder
    os.system(f"mkdir {parent_path}")
    num_episodes = 4000
    for j in range(num_episodes):
        obs = env.reset()
        picked_object = False
        for _ in range(100): # more needed for pick and place
            rgb, dep = env.render(mode='rgb_array', height=height, width=width, depth=True)
            new_rgb, new_dep = use_real_depths_and_crop(rgb, dep, vis=False)
           
            # sampling policy used saved model ?
            obs["rgb"] = new_rgb
            obs["dep"] = new_dep
            # obs["last_two"] = np.stack(last_two)

            actions, picked_object = sc_policy(obs, picked_object)
            obs["actions"] = actions

            # step actions
            new_obs, _, _ , _ = env.step(actions)
        
            save_image(j, obs, parent_path)
            obs = new_obs
        print(f"{j} number of episodes completed")

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
        # d = copy.deepcopy(d)
        d["actions"] = d["actions"].astype(np.float32)
        # print(d.keys())
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

    state_based_model, _, _, _ = model_factory(task='sym_state', env_params=env_params)
    env_params["depth"] = args.depth
    env_params["load_saved"] = False

    student_model, _, _, _ = model_factory(task=args.task, env_params=env_params)

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
        model_path = 'curr_rgb_model.pt'
        # model_path = 'dagger_rgb_model.pt'
        obj = torch.load(model_path, map_location=lambda storage, loc: storage)
        student_model.load_state_dict(obj['actor_net'])

    student_model.train()
    
    # plot_model_stats(obj)
    rand_i = str(np.random.uniform(0,1))
    print(f"start training for {rand_i}")
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
        if succ_rate >= best_succ_rate:
            torch.save(save_dict, f"best_bc_model_{rand_i}.pt")
            best_succ_rate = succ_rate
        else:
            torch.save(save_dict, f"curr_bc_model_{rand_i}.pt")

        # eval_agent_and_save(env, student_model, save_record
            
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


def train_1_epoch(ep, env, obj, args, loaded_net, scheduler, optimizer, rand_i, losses):

    dt_loader = get_offline_dataset(args)

    num_epochs = 500
    rewards = []
    best_succ_rate = 0

    if args.cuda:
        loaded_net = loaded_net.cuda(MPI.COMM_WORLD.Get_rank())
        if not args.scripted:
            state_based_model = state_based_model.cuda(MPI.COMM_WORLD.Get_rank())

    loaded_net.train()
    # plot_model_stats(obj)
    for update_step in range(args.n_batches):
        total_loss = 0
        start = time.time()
        for idx, dt in enumerate(dt_loader):

            dt["obj"] = obj
            obs_state = dt["observation"]
            g = dt["desired_goal"].to(torch.float32)

            # TODO normalize
            obs_img, g_norm, state_based_input = _preproc_inputs_image_goal(dt, args, is_np=False)

            if args.scripted:
                with torch.no_grad():
                    acts = dt["actions"]
                    if args.cuda:
                        acts = acts.cuda(MPI.COMM_WORLD.Get_rank())
            else:
                with torch.no_grad():
                    acts = loaded_net(state_based_input)
           
            # show_video(obs_img[0].permute(1,2,0).numpy().astype(np.uint8))
            if args.task != 'sym_state':
                student_acts = loaded_net(obs_img, g_norm)
            else:
                student_acts = loaded_net(state_based_input)
            

            loss = F.mse_loss(student_acts, acts)

            # step the loss
            optimizer.zero_grad()
            loss.backward()
            # plot_grad_flow(state_based_model.named_parameters())
            optimizer.step()

            total_loss += loss.item()

            # TODO: add plotting for training
            losses.append(loss.item())

        # scheduler.step(total_loss)

        end = time.time()

        # run after every epoch
        if len(dt_loader) != 0:
            print(f"-----------Epoch {(ep*args.n_batches)+update_step} | Loss {total_loss / len(dt_loader)} | length of dataset: {len(dt_loader) * args.batch_size}-----------")

    return loaded_net, scheduler, optimizer, losses


def add_to_dataset(ep, observations, parent_path):
    for i in range(len(observations)):
        # obs = {}
        # obs["rgb"] = rgbs[i]
        # obs["dep"] = depths[i]
        # obs["actions"] = actions[i]
        # for k in observations[i].keys():
        #     obs[k] = observations[i][k]
        del observations[i]['obj']
        # print(observations[i].keys())
        save_image(ep, observations[i], parent_path)


def dagger():

    rand_i = str(np.random.uniform(0,1))
    print(f"start training for {rand_i}")
    env = PickAndPlaceXarm(xml_path='./assets/fetch/pick_and_place_xarm.xml')
    env = load_viewer_to_env(env)
    modder = get_texture_modder(env)
    args = get_args()
    env_params = get_env_params(env)
    best_rate_yet = 0
    check_if_dataset_folder_exists(args)
    
    ######## get model stuff
    env_params["depth"] = args.depth
    env_params["load_saved"] = False
    env_params["model_path"] = paths['FetchPickAndPlace-v1']['xarm']['sym_state'] + '/model.pt'

    student_model, _, _, _ = model_factory(task=args.task, env_params=env_params)
    

    # get model object mean/std stats
    # obj = torch.load(env_params["model_path"], map_location=torch.device('cpu'))
    # obj = torch.load('curr_bc_model_0.6380849388222776.pt', map_location=torch.device('cpu'))
    # obj = torch.load('curr_rgb_model.pt', map_location=torch.device('cpu'))
    # student_model.load_state_dict(obj['actor_net'])
    
    #student_model = StupidNet()
    
    if args.cuda:
        student_model = student_model.cuda(MPI.COMM_WORLD.Get_rank())
    ######### optimizer and scheduler
    # dt_loader = get_offline_dataset(args)

    optimizer = torch.optim.SGD(params=student_model.parameters(), lr=0.01)

    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True)
    losses = []
    rewards = []
    #########
    ep = 0
    max_ep_len = 500

    # generate 400,000 transitions
    generate_dataset(None, env, None, args)

    # obj
    obj = torch.load('curr_rgb_model.pt', map_location=torch.device('cpu'))

    args.n_batches = 50

    # train for 50 epochs
    student_model, scheduler, optimizer, losses = train_1_epoch(0, env, obj, args, student_model, scheduler, optimizer, rand_i, losses)
    
    args.n_batches = 1

    print("now doing dagger")
    while True:
        obs = env.reset()

        # some episodic book keeping
        pick_object = False
        is_succ = 0
        since_success = 0
        ep_len = 0
        
        observations = []

        # episode run
        while ((not is_succ) or since_success < 50) and ep_len < max_ep_len:
            rgb, depth = env.render(mode='rgb_array', depth=True, height=100, width=100)
            
            obs["rgb"] = rgb
            obs["dep"] = depth
            obs["obj"] = obj
            obs["pick_object"] = pick_object
            obs_img, g_norm, state_based_input = _preproc_inputs_image_goal(obs, args, is_np=True)
            g_norm = g_norm.squeeze(0)
            #print(obs_img.shape, g_norm.shape) 

            # display_state(obs)
            # env.render()
            # time.sleep(0.1)
            act, pick_object = scripted_action(obs, pick_object)

            if args.task != 'sym_state':
                student_acts = student_model(obs_img, g_norm).detach().cpu().numpy().reshape((4))
            else:
                student_acts = student_model(state_based_input).detach().cpu().numpy().reshape((4))
            
            obs['actions'] = act
            observations.append(obs)

            # add probability to prevent distribution drift
            if np.random.uniform(0,1) > 0.5:
                a = student_acts
            else:
                a = act

            obs, _, _, infor = env.step(a)
            
            if infor['is_success'] != 1:
                since_success = 0
            else:
                since_success += 1

            is_succ = infor['is_success']

            ep_len += 1

            if out_of_bounds(obs):
                print("going out of bounds !")
                break
        if is_succ:
            add_to_dataset(ep, observations, args.bc_dataset_path)
        else:
            print("successful trajectory, going back to unsuccessful")
            continue

        # train for 1 epoch
        student_model, scheduler, optimizer, losses = train_1_epoch(ep, env, obj, args, student_model, scheduler, optimizer, rand_i, losses)

        succ_rate = eval_agent_and_save(ep, env, args, student_model, obj, args.task)

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

        if ep % 1 == 0:
            print(f"Evaluating Model: Success Rate {succ_rate}")
            if succ_rate >= best_rate_yet:
                print("saving model")
                best_rate_yet = succ_rate
                torch.save(save_dict, f"best_dagger_model_{rand_i}.pt")

        ep += 1
        torch.save(save_dict, f"curr_dagger_model_{rand_i}.pt")



def create_off_dataset():
    args = get_args()
    sc_policy = scripted_action_new
    env = PickAndPlaceXarm(xml_path='./assets/xarm/fetch/pick_and_place_xarm.xml')
    env = load_viewer_to_env(env)
    generate_dataset(sc_policy, env, args)

bc_train(env)
#create_off_dataset()
# dagger()
