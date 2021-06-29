from xarm_env.pick_and_place import PickAndPlaceXarm
from xarm_env.reach import XarmFetchReachEnv
import time
from utils import load_viewer_to_env
from rl_modules.utils import scripted_action, scripted_action_new
import numpy as np

def check_distance():
    reach_combined_gripper = 'assets/fetch/reach_xarm_with_gripper.xml'

    env = XarmFetchReachEnv(xml_path=reach_combined_gripper)
    obs = env.reset()

    s = time.time()
    for i in range(2):
        action = np.asarray([1, 0, 0, 1])
        env.step(action)
        env.render()
    e = time.time()



def show_big_ball(env, pos):
    sites_offset = (env.sim.data.site_xpos - env.sim.model.site_pos).copy()
    # print(f"Sites Offset {sites_offset}")
    site_id = env.sim.model.site_name2id('target1')
    env.sim.model.site_pos[site_id] = pos - sites_offset[0]
    env.sim.forward()
    # env.sim.step()

def out_of_bounds(obs):
    '''
    Return true if object is out of bounds from robot reach
    '''
    #if x pos is < 0.9  < 1.8 then bad, 0.05 < y < 1.3
    object_pos = obs['observation'][4:7]
    x = object_pos[0]
    y = object_pos[1]

    if x < 0.9 or x >= 1.75:
        return True
    
    if y < 0.05 or y > 1.25:
        return True
    
    return False

def test_scripted_policy():
    env = PickAndPlaceXarm(xml_path='./assets/xarm/fetch/pick_and_place_xarm.xml')
    # RUN : export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/x86_64-linux-gnu/libGL.so
    # env = load_viewer_to_env(env)
    num_steps = 50
    succ_count = 0
    fail_count = 0
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
        while (not is_succ) or k < num_steps:
            j += 1

            # rgb, depth = env.render(mode='rgb_array', depth=True, height=100, width=100)
            # rgb, depth = use_real_depths_and_crop(rgb, depth)
            # show_video(rgb)
            # print(obs["observation"][:3])
            
            act, pick_object = scripted_action_new(obs, pick_object)
            # RUN : unset LD_PRELOAD
            env.render()
            if ended or obs["observation"][0] < 1.1:
                ended = True
                act, pick_object = scripted_action_new(obs, pick_object) 
            else:
                act = np.asarray([-1, 0, 0, 0, 0])

            obs,  r, _, infor = env.step(act)
            
            left_gripper = obs['observation'][:3]
            right_gripper = obs['observation'][-3:]
            distances.append(abs(right_gripper[1] - left_gripper[1]))
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
        if is_succ:
            succ_count+=1
        else:
            fail_count+=1
        print(f"Succful Count: {succ_count} , Failure Count: {fail_count}")
        if succ_count + fail_count >= 101:
            break

test_scripted_policy()

