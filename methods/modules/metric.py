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
    

def metric_acc(query,key,cfg=None):
    # query=F.normalize(query, p=2, dim=1)
    # key=F.normalize(key, p=2, dim=1)
    # query=F.normalize(query, p=2, dim=1)
    # key=F.normalize(key, p=2, dim=1)
    n=query.shape[0]
    logits = query @ key.t()
    prob = F.softmax(logits, dim=1)
    maxarg=torch.argmax(prob,dim=1).detach().cpu().numpy()
    maxprob=torch.max(prob,dim=1)[0].detach().cpu().numpy()
    diagprob=torch.diagonal(prob).detach().cpu().numpy()
    # acc=sum([maxarg[i]==i for i in range(n)])

    # pred_flat = torch.argmax(logprob, dim=1).detach().cpu().numpy()

    ys = np.array(list(range(n)))
    # print("->max index:",maxarg,"->prob diff",np.mean(maxprob-diagprob))
    # print(torch.max(logprob,dim=1).detach().cpu().numpy())
    # acc=sum([maxarg[i]==i for i in range(n)]))
    acc= np.sum(maxarg == ys) 
    print(maxarg)
    if cfg is not None:
        cfg.logger.info(f"maxprob {np.mean(maxprob)},diagprob {np.mean(diagprob)}, probdiff {np.mean(maxprob-diagprob)}")
    else:
        print(f"maxprob {np.mean(maxprob)},diagprob {np.mean(diagprob)}, probdiff {np.mean(maxprob-diagprob)}")
    # print(acc,acc2)
    return acc,n