import argparse
import torch
from utils import *
class Config:
    def __init__(self):
        pass

    def init(self):
        parser = argparse.ArgumentParser(description='training template')
        parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                            help='input batch size for training (default: 128)')
        parser.add_argument('--num_epoch', type=int, default=10, metavar='N',
                            help='number of epochs to train (default: 10)')
        parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                            help='learning rate')
        parser.add_argument('--seed', type=int, default=2022, metavar='N',
                            help='random seed')
        args = parser.parse_args()
        

        self.batch_size=args.batch_size
        self.num_epoch=args.num_epoch
        self.lr=args.lr
        self.seed=args.seed

        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.work_dir="./work"
        self.train_optimizer_mode='adam'
        self.optimizer_cfg={'lr':self.lr,'weight_decay':0.1,'betas':(0.9, 0.99)}
        self.chkpt_dir="checkpoints"
        self.name="model"
        self.chkpt_interval=1000
        self.summary_interval=100

        self.load=Config()
        self.load.resume_state_path=None
        self.load.network_chkpt_path=None
        self.load.strict_load=None

        self.bert=Config()
        self.bert.query='bert-base-uncased'
        self.bert.key='bert-base-uncased'
        self.bert.dropout=0.3
        self.bert.outputdim=768

    def get_info(self):
        _dict={}
        for key in self.__dict__.keys():
            if isinstance(self.__dict__[key],Config):
                _dict[key]=self.__dict__[key].get_info()
            else:
                _dict[key]=self.__dict__[key]
        return _dict
    
    

