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
from methods.modules.metric import metric_acc
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import time
import pickle
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



def main(cfg):
    #set random seed
    set_seed(cfg.seed)
    #initialize logger
    logger=loadLogger(cfg)

    cfg.logger=logger


    pred_dataset=DriftDataset(cfg,'predict')
  
    pred_loader=DataLoader(pred_dataset,128,shuffle=False)
    # test_loader=DataLoader(train_dataset,cfg.batch_size,shuffle=False)    #debug  

    cfg.scheduler.num_training_steps = 0
    
    
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

    
    output=predict_model(cfg, model, pred_loader)

    
    # Open a file for writing binary data
    with open(cfg.array_path, 'wb') as f:
        # Use pickle to serialize the NumPy array and write it to the file
        pickle.dump(output, f)


    
    plot(output,cfg.fig_path)
    
if __name__=='__main__':
    cfg=Config()
    cfg.init()
    cfg.load.resume_state_path="/home/cc/github/VotingBasedDriftDetection/test/2023-03-08T01-59-04/checkpoints/model_32017.state"
    cfg.load.network_pth_path="/home/cc/github/VotingBasedDriftDetection/test/2023-03-08T01-59-04/checkpoints/model_32017.pth"
    # print(cfg.get_info())
    time=int(time.time())
    cfg.fig_path=f'/home/cc/github/VotingBasedDriftDetection/methods/results/MLP-{time}.jpg'
    cfg.array_path=f'/home/cc/github/VotingBasedDriftDetection/methods/results/MLP-{time}.pkl'
    main(cfg)
   