from transformers import BertTokenizer, BertModel, BertPreTrainedModel,BertConfig
from torch import nn
import torch
from refsum.modules.dataset import BertInput, ModelInput
from refsum.run.utils import*
from torch.utils.checkpoint import checkpoint
import sys
class BERTClass(nn.Module):
    def __init__(self,cfg):
        super(BERTClass, self).__init__()
        # configuration = BertConfig()
        self.query_bert = BertModel.from_pretrained(cfg.bert.tokenizer)
        self.key_bert = BertModel.from_pretrained(cfg.bert.tokenizer)
        self.key_drop = torch.nn.Dropout(cfg.bert.dropout)
        self.query_drop = torch.nn.Dropout(cfg.bert.dropout)
        self.key_fc = torch.nn.Linear(768, cfg.bert.outputdim)
        self.query_fc = torch.nn.Linear(768, cfg.bert.outputdim)
        
        self.key_bert.gradient_checkpointing_enable()
        self.query_bert.gradient_checkpointing_enable()

        self.rawoutput=cfg.bert.rawoutput
        # print(self.key_bert)
    def forward(self, input:ModelInput)->ModelOutput :
        key_out=self.key_bert(input['key']['ids'], attention_mask = input['key']['mask'], token_type_ids = input['key']['token_type_ids'])
        query_out=self.query_bert(input['query']['ids'], attention_mask = input['query']['mask'], token_type_ids = input['query']['token_type_ids'])

        # key_last_hidden_state=key_out[0]
        # query_last_hidden_state=query_out[0]
        key_pooler_output=key_out[1]
        query_pooler_output=query_out[1]

        key_embedding=key_pooler_output
        query_embedding=query_pooler_output
        if not self.rawoutput:
            query_embedding=self.query_drop(query_embedding)
            key_embedding= self.key_drop(key_embedding)
            key_embedding=self.key_fc(key_embedding)
            query_embedding=self.query_fc(query_embedding)
        # print('-->key_embedding shape:',key_embedding.shape,'-->key_embedding:',key_embedding,'-->query_embedding shape:',query_embedding.shape,'-->query_embedding:',query_embedding)
        return {'key':key_embedding,'query':query_embedding}
    
    
    
