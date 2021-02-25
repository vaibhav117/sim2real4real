import torch
import gym 
import mujoco_py

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

import math 

# env = gym.make('FetchReach-v1')
model = mujoco_py.load_model_from_path('./assets/robot.xml')
sim = mujoco_py.MjSim(model, nsubsteps=20)

viewer = mujoco_py.MjViewer(sim)
t = 0
while True:
    sim.data.ctrl[0] = math.cos(t / 10.) * 0.01
    sim.data.ctrl[1] = math.sin(t / 10.) * 0.01
    t += 1
    sim.step()
    viewer.render()
    if t > 100 and os.getenv('TESTING') is not None:
        break

print(model)
env = gym.make('FetchReach-v1')
# env_params = get_env_params(env)
# model = new_actor(env_params)
# model_path = './test.pt'
# torch.save(model.state_dict(), model_path)
# new_model = new_actor(env_params)
# new_model.load_state_dict(torch.load(model_path))
import time
new_obs = env.reset()
print(f"New obs {new_obs}")
env.render()
for i in range(20):
    act = new_obs["desired_goal"]
    actual_act = [act[0], act[1], act[2], 0]
    print(actual_act)
    new_obs, r, done, _ = env.step(actual_act)
    # new_obs, r, done, _ = env.step([-10, 10, 10, 0])
    print(f"New obs {new_obs}")
    env.render()

act = new_obs["desired_goal"]
actual_act = [act[0], act[1], act[2], 0]
print(actual_act)
new_obs, r, done, _  = env.step(actual_act)
print(f"Final obs {new_obs}")
env.render()

time.sleep(3)
# print(env.action_space.sample())

