import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import sys

class EntropyLoss(nn.Module):

    def __init__(self,cfg):
        super().__init__()
        self.device=cfg.device
        self.entropy=nn.CrossEntropyLoss()

    def forward(self, pred,target):
        loss=self.entropy(pred,target)
        return loss
        
