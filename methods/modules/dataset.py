import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from methods.run.utils import *
import pickle
from transformers import AutoTokenizer
import random
import numpy as np

class DirftDataset(Dataset):

    def __init__(self, cfg,mode='train',ratio=0.8):
        def dealdata(datas):
            datas=datas[1:]
            datas=datas[:,:-1]
            i=0
            window=5
            newdatas=[]
            while i+window<=datas.shape[0]:
                mat=datas[i:i+window,:]
                item=mat.flatten()
                data=item[:-3]
                label=item[-1]
                i+=1
                newdatas.append((data,label))
            return newdatas
        #get arxiv_dict
        datas = np.genfromtxt('./data/avg_all.csv', delimiter=',')
        # random.shuffle(cite_pair)
        new_data=dealdata(datas)

        train_size=int(len(new_data)*ratio)
        assert mode in ['train','test']
        if mode=='train':
            self.data=new_data[:train_size]
        else:
            self.data=new_data[train_size:]
        random.shuffle(self.cite_pair)

       


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # print(self.cite_pair[index])
        key_id,query_id=self.cite_pair[index]
        key_text = self.data[key_id]['abstract'].strip().lower()
        query_text=self.data[query_id]['abstract'].strip().lower()
        # query_text=key_text
        key_inputs = self.tokenizer.encode_plus(
            key_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
        )
        key_ids = key_inputs['input_ids']
        key_mask = key_inputs['attention_mask']
        key_token_type_ids = key_inputs["token_type_ids"]

        key:BertInput={
            'ids': torch.tensor(key_ids, dtype=torch.long),
            'mask': torch.tensor(key_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(key_token_type_ids, dtype=torch.long),
            'text': key_text
        }

        query_inputs = self.tokenizer.encode_plus(
            query_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
        )
        query_ids = query_inputs['input_ids']
        query_mask = query_inputs['attention_mask']
        query_token_type_ids = query_inputs["token_type_ids"]

        query:BertInput={
            'ids': torch.tensor(query_ids, dtype=torch.long),
            'mask': torch.tensor(query_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(query_token_type_ids, dtype=torch.long),
            'text': query_text
        }
        data:ModelInput={
            'key':key,
            'query':query
        }

        return data,torch.tensor(1, dtype=torch.float)