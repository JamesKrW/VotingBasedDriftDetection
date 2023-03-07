from config import Config
import sys
sys.path.append("/home/cc/github/ref-sum")
sys.path.append("/home/cc/github/ref-sum/refsum")
from utils import *
from methods.modules.model_arch import BERTClass
from methods.mgr.model import Model
from methods.modules.dataset import NXTENTDataset
import itertools
from tqdm import tqdm
import math
from torch.utils.data import DataLoader,RandomSampler,SequentialSampler
from methods.modules.metric import metric_acc

def train_model(cfg, model, train_loader):

    model.net.train()
    logger=cfg.logger
    pbar = tqdm(train_loader, postfix=f"loss: {model.loss_v:.04f}")
    
    total_train_loss = 0
    train_loop_len = 0
    for data, label in pbar:
        model.optimize_parameters(data, label)
        loss = model.loss_v
        
        pbar.postfix = f"Train, loss: {model.loss_v:.04f}"
        total_train_loss += loss
        train_loop_len+=1
        if (loss > 1e8 or math.isnan(loss)):
            logger.error("Loss exploded to %.02f at step %d!" % (loss, model.step))
            raise Exception("Loss exploded")

        if model.step % cfg.summary_interval == 0:
            with torch.no_grad():
                output = model.inference(data)
                acc,n=metric_acc(output['query'], output['key'],cfg)
            logger.info(
                "Train Loss %.04f, Train Acc %.04f  at (epoch: %d / step: %d)"
                % (loss, acc/n, model.epoch + 1, model.step)
            )
    logger.info(
                "Train Loss %.04f, at (epoch: %d)"
                % (total_train_loss/train_loop_len, model.epoch + 1)
            )
            

def test_model(cfg, model, test_loader):
    model.net.eval()
    logger=cfg.logger
    pbar = tqdm(test_loader, postfix=f"loss: {model.loss_v:.04f}")
    total_test_loss = 0
    test_loop_len = 0
    total_acc=0
    total_n=0
    with torch.no_grad():
        for data, label in pbar:
            output = model.inference(data)
            loss_v = model.loss_f(output, label.to(cfg.device))
            acc,n=metric_acc(output['query'], output['key'],cfg)
            total_acc+=acc
            total_n+=n
            total_test_loss += loss_v.to("cpu").item()
            test_loop_len += 1
            pbar.postfix = f"Test, loss: {loss_v:.04f}, acc: {acc/n}"

        acc=total_acc/total_n
        total_test_loss /= test_loop_len

        
        logger.info(
                "Test Loss %.04f, Test Acc %.04f, at (epoch: %d)"
                % (total_test_loss, acc,model.epoch + 1)
            )

def main(cfg):
    #initialize logger
    logger=loadLogger(cfg)

    cfg.logger=logger

    #set random seed
    set_seed(cfg.seed)


    train_dataset=NXTENTDataset(cfg,'train')
    test_dataset=NXTENTDataset(cfg,'test')
  
    train_loader=DataLoader(train_dataset,cfg.batch_size,shuffle=False)
    test_loader=DataLoader(test_dataset,cfg.batch_size,shuffle=False)
    # test_loader=DataLoader(train_dataset,cfg.batch_size,shuffle=False)    #debug  

    cfg.scheduler.num_training_steps = cfg.num_epoch * len(train_loader)
    
    
     #initilize model 
    net_arch = BERTClass(cfg)
    model = Model(cfg, net_arch=net_arch)


    # load model from trained
    if cfg.load.resume_state_path is not None:
        model.load_training_state()
    elif cfg.load.network_pth_path is not None:
        model.load_network()
    else:
        logger.info("Starting new training run.")

    
    
    epoch_step = 1
    for model.epoch in itertools.count(model.epoch + 1, epoch_step):
        if model.epoch > cfg.num_epoch:
            break
        train_model(cfg, model, train_loader)
        if model.epoch % cfg.chkpt_interval == 0:
            model.save_network()
            model.save_training_state()
        test_model(cfg, model, test_loader)
    logger.info("End of Train")
    
if __name__=='__main__':
    cfg=Config()
    cfg.init()
    # print(cfg.get_info())
    main(cfg)
    
   