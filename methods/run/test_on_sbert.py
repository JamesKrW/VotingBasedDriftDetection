from sentence_transformers import SentenceTransformer, util

from config import Config
import sys
sys.path.append("/home/cc/github/ref-sum")
sys.path.append("/home/cc/github/ref-sum/refsum")
from utils import *
from refsum.modules.model_arch import BERTClass
from refsum.mgr.model import Model
from refsum.modules.dataset import NXTENTDataset
import itertools
from tqdm import tqdm
import math
from torch.utils.data import DataLoader,RandomSampler,SequentialSampler
from refsum.modules.metric import metric_acc
import random
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
query_embedding = model.encode('How big is London')
passage_embedding = model.encode(['London has 9,787,426 inhabitants at the 2011 census',
                                  'London is known for its finacial district'])

print("Similarity:", util.dot_score(query_embedding, passage_embedding))

def main(cfg):
    #set random seed
    set_seed(cfg.seed)


    train_dataset=NXTENTDataset(cfg,'train')
   
  
    train_loader=DataLoader(train_dataset,cfg.batch_size,shuffle=False)
    

    for data,label in train_loader:
    # data, label=next(iter(train_loader))
        keytext=data['key']['text']
        querytext=data['query']['text']

        print(len(querytext))
        query_embedding=torch.tensor(model.encode(querytext))
        key_embedding=torch.tensor(model.encode(keytext))
        print(query_embedding.shape)
        print(key_embedding.shape)
        acc,n=metric_acc(query_embedding,key_embedding,None)
        print(acc,n)
    
    
if __name__=='__main__':
    cfg=Config()
    cfg.init()
    print(cfg.get_info())
    main(cfg)



