from config import Config
from utils import *
from refsum.modules.model_arch import BERTClass
from refsum.mgr.model import Model
import itertools
from tqdm import tqdm
import math




def train_model(cfg, model, train_loader):

    model.net.train()
    logger=cfg.logger
    pbar = tqdm(train_loader, postfix=f"loss: {model.loss_v:.04f}")
    

    for model_input, model_target in pbar:
        model.optimize_parameters(model_input, model_target)
        loss = model.loss_v

        pbar.postfix = f"loss: {model.loss_v:.04f}"

        if (loss > 1e8 or math.isnan(loss)):
            logger.error("Loss exploded to %.02f at step %d!" % (loss, model.step))
            raise Exception("Loss exploded")

        if model.step % cfg.summary_interval == 0:
            logger.info(
                "Train Loss %.04f at (epoch: %d / step: %d)"
                % (loss, model.epoch + 1, model.step)
            )

def test_model(cfg, model, test_loader):
    model.net.eval()
    logger=cfg.logger
    total_test_loss = 0
    test_loop_len = 0
    with torch.no_grad():
        for model_input, model_target in test_loader:
            output = model.inference(model_input)
            loss_v = model.loss_f(output, model_target.to(cfg.device))
            total_test_loss += loss_v.to("cpu").item()
            test_loop_len += 1

        total_test_loss /= test_loop_len

        
        logger.info(
                "Test Loss %.04f at (epoch: %d / step: %d)"
                % (total_test_loss, model.epoch + 1, model.step)
            )

def main(cfg):
    logger=loadLogger(cfg)

    cfg.logger=logger
    set_seed(cfg.seed)
    net_arch = BERTClass(cfg)
    
    
    train_loader=None
    test_loader=None

    
    model = Model(cfg, net_arch)
    if cfg.load.resume_state_path is not None:
        model.load_training_state()
    elif cfg.load.network_chkpt_path is not None:
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
    print(cfg.get_info())
    main(cfg)
    
   