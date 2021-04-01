from rl_modules.ddpg_agent import show_video
import torch 
import numpy as np
from depth_tricks import create_point_cloud, display_interactive_point_cloud


path = './recording_xarm/recording_425.pt'
depthz = True

obj = torch.load(path)


for ob in obj['traj']:
    pcds = []
    for o in ob:
        # print(o["observation_image"].shape)
        if depthz:
            img = o["observation_image"][:,:, :3]
            depth = o["observation_image"][:, :, 3]
            img = img * 255
            img = img.astype(np.uint8)
            descrip = f" Goal : {o['desired_goal']} | Achieved Goal: {o['achieved_goal']} | Difference: {abs(o['desired_goal'] - o['achieved_goal']).mean()}"
            pcds.append( (descrip, create_point_cloud(img, depth, vis=False)))
            # show_video(img)
            # show_video(depth)
        else:
            img = o["observation_image"]
            show_video(img)

    display_interactive_point_cloud(pcds)
    
