import gym
from mujoco_py import MjRenderContextOffscreen
import time

env = gym.make('FetchReach-v1')
env.reset()
def benchmark_gpu_rendering(env):
    # add rendering context
    viewer = MjRenderContextOffscreen(env.env.sim, True, 2, "cuda")
    viewer.cam.distance = 1.2
    viewer.cam.azimuth = 180
    viewer.cam.elevation = -25
    env.env._viewers['rgb_array'] = viewer
    avg_time = 0
    for i in range(100):
        s = time.time()
        env.step(env.action_space.sample())
        im = env.render(mode="rgb_array")
        e = time.time()
        avg_time = avg_time + (e - s)
    print(avg_time / 100)

benchmark_gpu_rendering(env)
