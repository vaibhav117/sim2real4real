import gym

env = gym.make('FetchReach-v1')

while True:
    observation = env.reset()
    goal = observation["desired_goal"]
    print("hey")