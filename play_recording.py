from rl_modules.ddpg_agent import show_video
import torch 
import numpy as np


path = './recording_xarm/recording_40.pt'
depth = True

obj = torch.load(path)


for ob in obj:
    for o in ob:
        print(o["observation_image"].shape)
        if depth:
            img = o["observation_image"][:,:, :3]
            depth = o["observation_image"][:, :, 3]
            img = img.astype(np.uint8)
            show_video(img)
            show_video(depth)
        else:
            img = o["observation_image"]
            show_video(img)
