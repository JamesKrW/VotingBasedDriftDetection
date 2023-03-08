from config import Config
import sys
sys.path.append("/home/cc/github/VotingBasedDriftDetection")
sys.path.append("/home/cc/github/VotingBasedDriftDetection/methods")
from utils import *
from methods.mgr.model import Model
from methods.modules.dataset import DriftDataset
from methods.modules.model_arch import MLP
import itertools
from tqdm import tqdm
import math
from torch.utils.data import DataLoader
from methods.modules.metric import metric_acc
import pickle
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
                acc,n=metric_acc(output, label,cfg)
            logger.info(
                "Train Loss %.04f, Train Acc %.04f  at (epoch: %d / step: %d)"
                % (loss, acc/n, model.epoch + 1, model.step)
            )
            cfg.train_acc.append(acc/n)
        cfg.train_loss.append(loss)
    logger.info(
                "Train Loss %.04f, at (epoch: %d)"
                % (total_train_loss/train_loop_len, model.epoch + 1)
            )

def predict_model(cfg, model, predict_loader):
    model.net.eval()
    logger=cfg.logger
    pbar = tqdm(predict_loader, postfix=f"loss: {model.loss_v:.04f}")
    total_test_loss = 0
    test_loop_len = 0
    total_acc=0
    total_n=0
    outputs=np.array([])
    with torch.no_grad():
        for data, label in pbar:
            output = model.inference(data)
            maxarg=torch.argmax(output,dim=1)
            maxarg=maxarg*10
            outputs=np.concatenate((outputs,maxarg.detach().cpu().numpy().flatten()))
            loss_v = model.loss_f(output, label.to(cfg.device))
            acc,n=metric_acc(output,label,cfg)
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
    print(outputs.shape)
    return outputs     

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
            acc,n=metric_acc(output,label,cfg)
            total_acc+=acc
            total_n+=n
            total_test_loss += loss_v.to("cpu").item()
            test_loop_len += 1
            pbar.postfix = f"Test, loss: {loss_v:.04f}, acc: {acc/n}"

        acc=total_acc/total_n
        total_test_loss /= test_loop_len
        cfg.test_acc.append(acc)
        cfg.test_loss.append(total_test_loss)
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


    train_dataset=DriftDataset(cfg,'train')
    test_dataset=DriftDataset(cfg,'test')
    pred_dataset=DriftDataset(cfg,'predict')

    train_loader=DataLoader(train_dataset,cfg.batch_size,shuffle=True)
    test_loader=DataLoader(test_dataset,cfg.batch_size,shuffle=False)
    pred_loader=DataLoader(pred_dataset,128,shuffle=False)

    cfg.scheduler.num_training_steps = cfg.num_epoch * len(train_loader)
    
    
     #initilize model 
    net_arch = MLP(cfg)

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

    #run prediction
    output=predict_model(cfg, model, pred_loader)
    # Open a file for writing binary data
    with open(osp.join(cfg.work_dir,'pred.pickle'), 'wb') as f:
        # Use pickle to serialize the NumPy array and write it to the file
        pickle.dump(output, f)

    ogdata = np.genfromtxt(cfg.data.path, delimiter=',')
    ogdata=ogdata[1:,-2]
    plot(output,osp.join(cfg.work_dir,'pred.jpg'),(0,100))
    plot(ogdata,osp.join(cfg.work_dir,'og.jpg'),(0,100))
    plot(cfg.test_acc,osp.join(cfg.work_dir,'test_acc.jpg'),"test_acc")
    plot(cfg.train_loss,osp.join(cfg.work_dir,'train_loss.jpg'),"train_loss")
    plot(cfg.train_acc,osp.join(cfg.work_dir,'train_acc.jpg'),"train_acc")
    plot(cfg.test_loss,osp.join(cfg.work_dir,'test_loss.jpg'),"test_loss")

if __name__=='__main__':
    cfg=Config()
    cfg.init()
    cfg.train_loss=[]
    cfg.test_loss=[]
    cfg.train_acc=[]
    cfg.test_acc=[]
    # print(cfg.get_info())
    main(cfg)
    
   