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
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, env_params['action'])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions

class asym_goal_outside_image(nn.Module):
    def __init__(self, env_params):
        super(asym_goal_outside_image, self).__init__()
        self.max_action = env_params['action_max']
        if env_params["depth"]:
            in_chan = 4
        else:
            in_chan = 3
        self.cnn1 = nn.Conv2d(in_channels=in_chan, out_channels=32, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.cnn2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.cnn3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.cnn4 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(num_features=1)
        self.g_fc1 = nn.Linear(3, 16)
        self.g_fc2 = nn.Linear(16, 32)
        self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc1 = nn.Linear(816, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.action_out = nn.Linear(512, env_params['action'])

    def forward(self, x, g):
        x = F.relu(self.bn1(self.cnn1(x)))
        x = F.relu(self.bn2(self.cnn2(x)))
        x = F.relu(self.bn3(self.cnn3(x)))
        x = F.relu(self.bn4(self.cnn4(x)))
        x = self.flatten(x)

        g = F.relu(self.g_fc1(g))
        g = F.relu(self.g_fc2(g))

        x = torch.cat([x, g], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions


class resnet_actor(nn.Module):
    def __init__(self, env_params):
        super(resnet_actor, self).__init__()
        self.max_action = env_params['action_max']
        self.encoder = get_encoder('resnet18')
        self.conv_compress = nn.Conv2d(in_channels=512, out_channels=16, kernel_size=2)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc1 = nn.Linear(144+3, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.action_out = nn.Linear(512, env_params['action'])

    def forward(self, x, g):
        x = self.encoder(x)[-1]
        x = F.relu(self.bn1(self.conv_compress(x)))
        x = self.flatten(x)
        # g = F.relu(self.g_fc1(g))
        # g = F.relu(self.g_fc2(g))

        x = torch.cat([x, g], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions


class sym_image(nn.Module):
    def __init__(self, env_params):
        super(sym_image, self).__init__()
        self.max_action = env_params['action_max']
        if env_params["depth"]:
            in_chan = 4
        else:
            in_chan = 3
        self.cnn1 = nn.Conv2d(in_channels=in_chan, out_channels=32, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.cnn2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.cnn3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.cnn4 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(num_features=1)
        self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, env_params['action'])

    def forward(self, x):
        x = F.relu(self.bn1(self.cnn1(x)))
        x = F.relu(self.bn2(self.cnn2(x)))
        x = F.relu(self.bn3(self.cnn3(x)))
        x = F.relu(self.bn4(self.cnn4(x)))
        x = self.flatten(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions



class sym_image_critic(nn.Module):
    def __init__(self, env_params):
        super(sym_image_critic, self).__init__()
        if env_params["depth"]:
            in_chan = 4
        else:
            in_chan = 3
        self.max_action = env_params['action_max']
        self.cnn1 = nn.Conv2d(in_channels=in_chan, out_channels=32, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.cnn2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.cnn3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.cnn4 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(num_features=1)
        self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc1 = nn.Linear(85, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, 1)

    def forward(self, x, actions):
        x = F.relu(self.bn1(self.cnn1(x)))
        x = F.relu(self.bn2(self.cnn2(x)))
        x = F.relu(self.bn3(self.cnn3(x)))
        x = F.relu(self.bn4(self.cnn4(x)))
        x = self.flatten(x)
        x = torch.cat([x, actions / self.max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.action_out(x)
        return q_value

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

