import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import sys

# class Acc(nn.Module):

#     def __init__(self,cfg):
#         super().__init__()

#     def forward(self, query,key):
#         n=query.shape[0]
#         logits = query @ key.t()
#         logprob = F.log_softmax(logits, dim=1)
#         maxarg=torch.argmax(logprob,dim=1).cpu().numpy()
#         print(maxarg)
#         acc=sum([maxarg[i]==i for i in range(n)])
#         return acc,n
    

def metric_acc(pred,target,cfg):
    # query=F.normalize(query, p=2, dim=1)
    # key=F.normalize(key, p=2, dim=1)
    # query=F.normalize(query, p=2, dim=1)
    # key=F.normalize(key, p=2, dim=1)
    pred=pred.to(cfg.device)
    target=target.to(cfg.device)
    maxarg=torch.argmax(pred,dim=1)
    n=pred.shape[0]
    acc= torch.sum(maxarg == target).cpu().numpy()
    return acc,n