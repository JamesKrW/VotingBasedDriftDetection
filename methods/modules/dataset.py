import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from methods.run.utils import *
import pickle
import random
import numpy as np

class DriftDataset(Dataset):

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


        datas = np.genfromtxt(cfg.data.path, delimiter=',')
        datas[np.isnan(datas)]=0

        new_data=dealdata(datas)


        train_size=int(len(new_data)*ratio)
        assert mode in ['train','test','predict']
        if mode=='train':
            self.data=new_data[:train_size]
            random.shuffle(self.data)
        elif mode=='test':
            self.data=new_data[train_size:]
            random.shuffle(self.data)
        else:
            self.data=new_data
       


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data=self.data[index][0]
        label=np.array(int(min(self.data[index][1],99)/10))
        data_tensor=torch.tensor(data,dtype=torch.float32)
        label_tensor=torch.from_numpy(label)
        return data_tensor,label_tensor