from rl_modules.ddpg_agent import show_video
import torch 
import numpy as np


path = './recording_xarm/recording_10.pt'
depthz = True

obj = torch.load(path)


for ob in obj:
    for o in ob:
        # print(o["observation_image"].shape)
        if depthz:
            img = o["observation_image"][:,:, :3]
            depth = o["observation_image"][:, :, 3]
            print(depth.mean())
            img = img.astype(np.uint8)
            show_video(img)
            show_video(depth)
        else:
            img = o["observation_image"]
            show_video(img)
