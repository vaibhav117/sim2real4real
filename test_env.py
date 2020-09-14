import gym
import matplotlib.pyplot as plt
env = gym.make('FetchReach-v1')

while True:
    observation = env.reset()
    img_obs = env.render(mode="rgb_array")
    img_obs = env.render(mode="rgb_array", height=100, width=100)
    plt.imshow(img_obs)
    plt.show()
    goal = observation["desired_goal"]
