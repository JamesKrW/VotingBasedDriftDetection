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
        parser.add_argument('--num_epoch', type=int, default=1000, metavar='N',
                            help='number of epochs to train (default: 10)')
        parser.add_argument('--lr', type=float, default=1e-5, metavar='N',
                            help='learning rate')
        parser.add_argument('--seed', type=int, default=2022, metavar='N',
                            help='random seed')
        parser.add_argument('--data_path', type=str, default=2022, metavar='N',
                            help='data path to load data')
        args = parser.parse_args()
        

        self.batch_size=args.batch_size
        self.num_epoch=args.num_epoch
        self.lr=args.lr
        self.seed=args.seed

        self.data=Config()
        self.data.path=args.data_path

        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.work_dir=f"./{self.data.path.split('/')[-1].split('.')[0].strip()}"
        self.optimizer_cfg={'lr':self.lr,'weight_decay':0.1,'betas':(0.9, 0.99)}
        self.chkpt_dir="checkpoints"
        self.savename="model"
        self.chkpt_interval=10
        self.summary_interval=30

        self.load=Config()
        self.load.resume_state_path=None
        # "/home/cc/github/ref-sum/work/2023-02-28T05-04-37/checkpoints/model_2.state"
        self.load.network_pth_path=None
        # "/home/cc/github/ref-sum/work/2023-02-28T05-04-37/checkpoints/model_2.pth"
        self.load.strict_load=None


        self.ntxent=Config()
        self.ntxent.tau=0.05

        self.scheduler=Config()
        self.scheduler.num_warmup_steps=20
        self.scheduler.num_training_steps=None #caculated in main()

    def get_info(self):
        _dict={}
        for key in self.__dict__.keys():
            if isinstance(self.__dict__[key],Config):
                _dict[key]=self.__dict__[key].get_info()
            else:
                _dict[key]=self.__dict__[key]
        return _dict
    
    

