import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import sys

class NTXent(nn.Module):

    def __init__(self,cfg):
        super().__init__()
        self.tau=cfg.ntxent.tau

    def forward(self, query,key):
        query=query/torch.sqrt(self.tau)
        key=key/torch.sqrt(self.tau)
        n=query.shape[0]
        logits = torch.sigmoid(query @ key.t())
        logprob = F.log_softmax(logits, dim=1)
        loss=-torch.diagonal(logprob).sum()/n
        return loss

        
