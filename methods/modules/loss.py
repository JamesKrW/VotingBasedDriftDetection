import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import sys

class NTXent(nn.Module):

    def __init__(self,cfg):
        super().__init__()
        self.device=cfg.device
        self.tau=torch.tensor([[cfg.ntxent.tau]],requires_grad=False).to(self.device)
        self.entropy=nn.CrossEntropyLoss()

    def forward(self, query,key):
        # query=F.normalize(query, p=2, dim=1)/torch.sqrt(self.tau)
        # key=F.normalize(key, p=2, dim=1)/torch.sqrt(self.tau)
        query=query/torch.sqrt(self.tau)
        key=key/torch.sqrt(self.tau)
        n=query.shape[0]
        # logits = query @ key.t()
        # logprob = F.log_softmax(logits, dim=1)
        # # print(logprob)
        # loss1=-torch.diagonal(logprob).sum()/n
        
    
        pred_logits = torch.mm(query, key.transpose(0, 1))
        ys = torch.tensor(list(range(n))).to(self.device)
        loss2=self.entropy(pred_logits,ys)

        # print(loss1,loss2)
        return loss2
        
