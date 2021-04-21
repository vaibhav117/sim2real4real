import matplotlib.pyplot as plt
import time 
import cv2
import numpy as np
from mpi4py import MPI
from mujoco_py import MjRenderContextOffscreen
from mujoco_py.modder import TextureModder

def load_viewer(sim, device_id=MPI.COMM_WORLD.Get_rank()):
    viewer = MjRenderContextOffscreen(sim, device_id=device_id)
    viewer.cam.distance = 1.2 # this will be randomized baby: domain randomization FTW
    viewer.cam.azimuth = 180 # this will be randomized baby: domain Randomization FTW
    viewer.cam.elevation = -25 # this will be randomized baby: domain Randomization FTW
    viewer.cam.lookat[2] = 0.5 # IMPORTANT FOR ALIGNMENT IN SIM2REAL !!
    return viewer

def normalize_depth(img):
    near = 0.021
    far = 2.14
    img = near / (1 - img * (1 - near / far))
    return img*15.5

def use_real_depths_and_crop(rgb, depth, height=100, width=100, vis=False):
    # TODO: add rgb normalization as well
    depth = normalize_depth(depth)
    depth = (depth - 0.021) / (2.14 - 0.021)
    
    depth = cv2.resize(depth[10:80, 10:90], (height, width))
    rgb = cv2.resize(rgb[10:80, 10:90, :], (height, width))
    
    if vis:
        from depth_tricks import create_point_cloud
        create_point_cloud(rgb, depth, vis=True)

    return rgb, depth[:, :, np.newaxis]

class Benchmark:

    def __init__(self):
        self.counter = {}
        self.total_time = {}
    
    def add(self, k, t):
        if k in self.counter:
            self.counter[k] += 1
            self.total_time[k] += t
        else:
            self.counter[k] = 1
            self.total_time[k] = t
    
    def plot(self):
        times = []
        keys = []
        freq = []
        for k, num_times in self.counter.items():
            total_time = self.total_time[k]
            times.append(total_time / num_times)
            keys.append(k)
            freq.append(num_times)

        #print(keys)
        fig, axs = plt.subplots(1,2)

        axs[0].plot(range(len(keys)), times)
        axs[0].set_xlabel("keys")
        axs[0].set_ylabel("function time")
        axs[1].plot(range(len(keys)), freq)
        axs[1].set_xlabel("keys")
        axs[1].set_ylabel("freq of functions")
        plt.xticks(range(len(keys)), keys, rotation=90)
        plt.savefig('speed_plot.png')
        plt.clf()
    
    def __call__(self, fn):
        # *args and **kwargs are to support positional and named arguments of fn
        def get_time(*args, **kwargs): 
            start = time.time() 
            output = fn(*args, **kwargs)
            time_taken = time.time() - start
            self.add(fn.__name__, time_taken)
            return output  # make sure that the decorator returns the output of fn
        return get_time 

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    print(layers)
    plt.show()

def timeit(fn): 
    # *args and **kwargs are to support positional and named arguments of fn
    def get_time(*args, **kwargs): 
        start = time.time() 
        output = fn(*args, **kwargs)
        time_taken = time.time() - start
        print(f"Time taken in {fn.__name__}: {time_taken:.7f}")

        return output  # make sure that the decorator returns the output of fn
    return get_time 


def show_video(img):
    cv2.imshow('frame', cv2.resize(img, (200,200)))
    cv2.waitKey()

def get_texture_modder(env):
    modder = TextureModder(env.sim)
    return modder

def randomize_textures(modder, env):
    for name in env.sim.model.geom_names:
        modder.rand_all(name)
