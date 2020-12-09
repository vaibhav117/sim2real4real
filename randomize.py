#!/usr/bin/env python3
"""
Displays robot fetch at a disco party.
"""
from mujoco_py import load_model_from_path, MjSim, MjViewer, MjRenderContextOffscreen
from mujoco_py.modder import TextureModder, MaterialModder, CameraModder, LightModder
import os
import gym
import random
import torch
from mujoco_py.generated import const
import matplotlib.pyplot as plt
import cv2

def show_video(img1):
    cv2.imshow('current frame', cv2.resize(img1, (200,200)))
    cv2.waitKey(0)

def randomize_camera(viewer):
    viewer.cam.distance = random.randrange(1,3)
    viewer.cam.azimuth = random.randint(160,190)
    viewer.cam.elevation = random.randint(-45,-25)

def randomize_textures(sim):
    for name in sim.model.geom_names:
        modder.rand_all(name)

def randomize_lights(sim):
    model = sim.model 
    names = model.light_names
    for name in names:
        lightid = model.light_name2id(name)
        value = model.light_ambient[lightid]
        print(value)
        model.light_ambient[lightid] = value +1

# model = load_model_from_path("/Users/karanchahal/projects/mujoco-py/xmls/fetch/main.xml")
# model = load_model_from_path("/Users/karanchahal/miniconda3/envs/rlkit/lib/python3.6/site-packages/gym/envs/robotics/assets/fetch/pick_and_place.xml")
# sim = MjSim(model)

env = gym.make('FetchPickAndPlace-v1')
# exit()
# env.sim = sim

sim = env.sim
viewer = MjRenderContextOffscreen(sim)
# viewer.cam.fixedcamid = 3
viewer.cam.distance = 1.2
viewer.cam.azimuth = 180
viewer.cam.elevation = -25
# viewer.cam.type = const.CAMERA_FIXED
env.env._viewers['rgb_array'] = viewer
im = env.render(mode="rgb_array")
# plt.imshow(im)
# plt.show()

modder = TextureModder(sim)
# modder = CameraModder(sim)
modder.whiten_materials()
# modder = MaterialModder(sim)

t = 1

# viewer.cam.fixedcamid = 3
# viewer.cam.type = const.CAMERA_FIXED

while True:
    
    randomize_textures(sim)
    # randomize_lights(sim)
    # if t % 10 == 0:
    randomize_camera(viewer)
    # for name in sim.model.camera_names:
    #     loc = modder.get_pos(name) + torch.randn(3).numpy()
    #     # print(loc)
    #     modder.set_pos(name, loc)
    # #     # modder.set_fovy(name, t)

    # viewer.render()
    im = env.render(mode="rgb_array")

    plt.imshow(im)
    plt.show()
    # env.render()
    t += 1
    if t > 100 and os.getenv('TESTING') is not None:
        break