from transformers import BertTokenizer, BertModel, BertPreTrainedModel,BertConfig
from torch import nn
import torch
from methods.modules.dataset import BertInput, ModelInput
from methods.run.utils import*
from torch.utils.checkpoint import checkpoint
import sys


class MLP(nn.Module):
    def __init__(self,cfg):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(47, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, 10)
        )
        
    def forward(self, x):
        # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)

        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x
    
class MLP_R(nn.Module):
    def __init__(self,cfg):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(47, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, 10)
        )
        
    def forward(self, x):
        # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)

        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x
    
    
