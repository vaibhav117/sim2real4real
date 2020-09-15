import torch.nn as nn
import torch
import torch.nn.functional as F
import time 

class cheap_cnn(nn.Module):

    def __init__(self):
        super(cheap_cnn, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.cnn2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.cnn3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3)
        self.cnn4 = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3)
        self.flatten = nn.Flatten()

    def forward(self, x):
        bt_sz = x.size(0)
        x = F.relu(self.cnn1(x))
        x = F.relu(self.cnn2(x))
        x = F.relu(self.cnn3(x))
        x = F.relu(self.cnn4(x))
        return self.flatten(x)

# define the actor network
class actor2(nn.Module):
    def __init__(self):
        super(actor2, self).__init__()
        self.fc1 = nn.Linear(13, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = torch.tanh(self.action_out(x))
        return actions

def test():
    m = cheap_cnn()
    # m = actor()
    total = 0
    a = torch.randn(1,3,100,100)
    # a = torch.randn(1,13)
    for i  in range(100):
        s = time.time()
        b = m(a)
        print(b.size())
        e = time.time()
        total += (e - s)
    
    print(float(total/100))
    # print(b.size())

# test()
