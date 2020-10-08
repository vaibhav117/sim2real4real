import gym
import matplotlib.pyplot as plt

env = gym.make('FetchPush-v1')


def reset_goal_fetch_reach(env, ach_goal):
    sites_offset = (env.env.sim.data.site_xpos - env.env.sim.model.site_pos).copy()
    site_id = env.env.sim.model.site_name2id('target0')
    env.env.sim.model.site_pos[site_id] = ach_goal - sites_offset[0]

states = []
observation = env.reset()

for x in range(1000):
    env.render()
    
    # img_obs = env.render(mode="rgb_array", height=100, width=100)
    ach_goal = observation["achieved_goal"]
    states.append(env.env.sim.get_state())
    new_obs, _, _, _ = env.step(env.action_space.sample())
    # reset_goal(env, ach_goal)
    # env.render()


for x in range(1000):
    state = states[x]
    env.env.sim.set_state(state)

    env.render()
    