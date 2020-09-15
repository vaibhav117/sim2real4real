import torch
import torch.nn as nn
import torch.nn.functional as F
from featurizers.factory import get_encoder
"""
the input x in both networks should be [o, g], where o is the observation and g is the goal.

"""

# define the actor network
class actor(nn.Module):
    def __init__(self, env_params):
        super(actor, self).__init__()
        self.max_action = env_params['action_max']
        # self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc1 = nn.Linear(16931, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, env_params['action'])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions

class critic(nn.Module):
    def __init__(self, env_params):
        super(critic, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)

    def forward(self, x, actions):
        x = torch.cat([x, actions / self.max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)

        return q_value

class img_actor(nn.Module):
    def __init__(self):
        super(img_actor, self).__init__()
        self.encoder = get_encoder('resnet18')
        self.compress = nn.Conv2d(in_channels=512, out_channels=32, kernel_size=2)
        self.compress2 = nn.Conv2d(in_channels=32, out_channels=10, kernel_size=2)
        self.fc1 = nn.Linear(40, 10)

    def forward(self, x):
        bt_sz = x.size(0)
        x = self.encoder(x)[-1]
        x = F.relu(self.compress(x))
        x = F.relu(self.compress2(x))
        x = self.fc1(x.view(bt_sz,-1))
        return x
