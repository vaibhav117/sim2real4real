
from pprint import pprint
import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
# model_names = timm.list_models(pretrained=True)
# pprint(model_names)

class StupidNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.classifier = timm.create_model('efficientnet_b0', pretrained=True)
        self.classifier.train()

        self.fc1 = nn.Linear(in_features=1000, out_features=4)

    def forward(self, x):
        x = F.relu(self.classifier(x))
        x = self.fc1(x)
        return x

def do_something(net):
    o = net(torch.randn(2, 3, 100, 100))
    
    print(f'Unpooled shape: {o.shape}')

# net = StupidNet()
# do_something(net)

