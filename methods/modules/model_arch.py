from transformers import BertTokenizer, BertModel, BertPreTrainedModel,BertConfig
from torch import nn
import torch
from dataset import BertInput, ModelInput
from refsum.run.utils import*
from torch.utils.checkpoint import checkpoint

class BERTClass(nn.Module):
    def __init__(self,cfg):
        super(BERTClass, self).__init__()
        self.query = BertModel.from_pretrained(cfg.bert.query)
        self.key = BertModel.from_pretrained(cfg.bert.key)
        self.key_drop = torch.nn.Dropout(cfg.bert.dropout)
        self.query_drop = torch.nn.Dropout(cfg.bert.dropout)
        self.key_fc = torch.nn.Linear(768, cfg.bert.outputdim)
        self.query_fc = torch.nn.Linear(768, cfg.bert.outputdim)

        self.key.gradient_checkpointing_enable()
        self.query.gradient_checkpointing_enable()
    def forward(self, input:ModelInput)->ModelOutput :
        _, key=self.key(input['key']['ids'], attention_mask = input['key']['mask'], token_type_ids = input['key']['token_type_ids'])
        _, query=self.query(input['query']['ids'], attention_mask = input['query']['mask'], token_type_ids = input['query']['token_type_ids'])
       
        query=self.query_drop(query)
        key = self.key_drop(key)
        key=self.key_fc(key)
        query=self.query_fc(query)
        return {'key':key,'query':query}
    
    
    
