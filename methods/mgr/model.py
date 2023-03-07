import os
import os.path as osp
from collections import OrderedDict

import torch
import torch.nn

from refsum.run.utils import *
from refsum.modules.loss import NTXent


class Model:
    def __init__(self, cfg, net_arch,optimizer):
        self.cfg = cfg
        self.device = self.cfg.device
        self.net = net_arch.to(self.device)
        self.step = 0
        self.epoch = -1

        # init logger
        self._logger = self.cfg.logger
        self._logger.info(f"{self.cfg.get_info()}")
        self._logger.info(f"training start")

        # init optimizer
        if optimizer==None:
            optimizer_mode = self.cfg.train_optimizer_mode
            if optimizer_mode == "adam":
                self.optimizer = torch.optim.Adam(
                    self.net.parameters(), **(self.cfg.optimizer_cfg)
                )
            else:
                raise Exception("%s optimizer not supported" % optimizer_mode)
        else:
            self.optimizer=optimizer

        # init loss

        self.loss_v = 0

        #init metric
        self.metric_f=None

        self.metric_v=0
        
    def loss_f(self,input:ModelOutput,target):
        return NTXent()(input["key"],input["query"])
    
    def optimize_parameters(self, model_input:ModelInput, model_target):
        self.net.train()
        self.optimizer.zero_grad()
        output = self.run_network(model_input)
        loss_v = self.loss_f(output, model_target.to(self.device))
        loss_v.backward()
        self.optimizer.step()
        self.step+=1
     
        self.loss_v = loss_v.item()

    def inference(self, model_input:ModelInput):
        self.net.eval()
        output = self.run_network(model_input)
        return output

    def run_network(self, model_input:ModelInput):
        model_input = model_input.to(self.device)
        output = self.net(model_input)
        return output

    def save_network(self, save_file=True):
        net = self.net
        state_dict = net.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.to("cpu")
        if save_file:
            save_filename = "%s_%d.pt" % (self.cfg.name, self.step)
            save_path = osp.join(self.cfg.chkpt_dir, save_filename)
            save_path = osp.join(self.cfg.work_dir,save_path)
            mkdir(save_path)
            torch.save(state_dict, save_path)
            self._logger.info("Saved network checkpoint to: %s" % save_path)
        return state_dict

    def load_network(self, loaded_net=None):
        if loaded_net is None:
            loaded_net = torch.load(
                self.cfg.network_chkpt_path,
                map_location=torch.device(self.device),
            )
        loaded_clean_net = OrderedDict()  # remove unnecessary 'module.'
        for k, v in loaded_net.items():
            if k.startswith("module."):
                loaded_clean_net[k[7:]] = v
            else:
                loaded_clean_net[k] = v

        self.net.load_state_dict(loaded_clean_net, strict=self.cfg.load.strict_load)
        
        self._logger.info(
            "Checkpoint %s is loaded" % self.cfg.load.network_chkpt_path)
            

    def save_training_state(self):
        
        save_filename = "%s_%d.state" % (self.cfg.name, self.step)
        save_path = osp.join(self.cfg.chkpt_dir, save_filename)
        save_path = osp.join(self.cfg.work_dir,save_path)
        mkdir(save_path)
        net_state_dict = self.save_network(False)
        state = {
            "model": net_state_dict,
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
        }
        torch.save(state, save_path)
        
        self._logger.info("Saved training state to: %s" % save_path)

    def load_training_state(self):
        
        resume_state = torch.load(
            self.cfg.load.resume_state_path,
            map_location=torch.device(self.device),
        )

        self.load_network(loaded_net=resume_state["model"])
        self.optimizer.load_state_dict(resume_state["optimizer"])
        self.step = resume_state["step"]
        self.epoch = resume_state["epoch"]
        
        self._logger.info(
            "Resuming from training state: %s" % self.cfg.load.resume_state_path)
            