import torch
import matplotlib.pyplot as plt
import numpy as np
# push_state_obj = torch.load('sym_server_weights/saved_models/sym_state/FetchPush-v1/model.pt', map_location=lambda storage, loc: storage)
# push_asym_obj = torch.load('sym_server_weights/saved_models/asym_goal_outside_image/FetchPush-v1/model.pt', map_location=lambda storage, loc: storage)
# push_distill = torch.load('sym_server_weights/saved_models/distill/image_only/model.pt', map_location=lambda storage, loc: storage)

push_state_obj = torch.load('sym_server_weights/saved_models/sym_state/FetchPickAndPlace-v1/model.pt', map_location=lambda storage, loc: storage)
push_asym_obj = torch.load('sym_server_weights/saved_models/asym_goal_outside_image/FetchPickAndPlace-v1/model.pt', map_location=lambda storage, loc: storage)
push_distill = torch.load('sym_server_weights/saved_models/distill/image_only/fetch_pick_and_place/model.pt', map_location=lambda storage, loc: storage)


for k,v in push_state_obj.items():
    print(k)
# print(push_state_obj.items())
rew_state = push_state_obj['reward']
rew_asym = push_asym_obj['reward']
rew_distill = push_distill['reward']
# print(len(push_state_obj['actor_losses']), len(push_asym_obj['actor_losses']), len(push_distill['losses']))
# print(len(push_state_obj['reward']), len(push_asym_obj['reward']), len(push_distill['reward']))
one = len(push_state_obj['actor_losses']) / len(push_state_obj['reward'])
two = len(push_asym_obj['actor_losses']) / len(push_asym_obj['reward'])
three = len(push_distill['losses']) / len(push_distill['reward'])


plt.plot(np.arange(len(rew_state))*one*4, rew_state, color='blue', label='states')
plt.plot(np.arange(len(rew_asym))*two*4, rew_asym,  color='red', label='asym')
plt.plot(np.arange(len(rew_distill))*three*4, rew_distill,  color='green', label='distill')
plt.title('Comparison of approaches for the FetchPickAndPlace env')
plt.xlabel("Steps")
plt.ylabel("Mean Reward")
plt.legend()
plt.show()
