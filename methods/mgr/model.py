import os
import os.path as osp
from collections import OrderedDict

import torch
import torch.nn
import sys
from methods.run.utils import *
from methods.modules.loss import EntropyLoss
from transformers import get_scheduler
from torch.optim import Adam
class Model:
    def __init__(self, cfg, net_arch):
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
        self.optimizer=Adam(self.net.parameters(), **self.cfg.optimizer_cfg)
        # param_total=0
        # print(len(self.optimizer.param_groups))
        # for param_group in self.optimizer.param_groups:
        #     # print(param_group['params'])
        #     for param in param_group['params']:
        #         param_total+=param.abs().mean()
            
        print(self.optimizer)
        # print(param_total)
        self.lr_scheduler = get_scheduler(
    name="linear", optimizer=self.optimizer, num_warmup_steps=self.cfg.scheduler.num_warmup_steps, num_training_steps=self.cfg.scheduler.num_training_steps
    )
        # print(self.lr_scheduler)
        # init loss

        self.loss_v = 0
        self.lossfunc=EntropyLoss(cfg)

    def loss_f(self,data,target):
        return self.lossfunc(data,target)
    
    def optimize_parameters(self, data,target):
        self.net.train()
        self.optimizer.zero_grad()
        output = self.run_network(data)
        loss_v = self.loss_f(output, target.to(self.device))
        loss_v.backward()
        # for name, param in self.net.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.shape)

        # param_total=0
        # grad_total=0
        # for name,params in self.net.named_parameters():
        #     # print(name,'grad_requires:',params.requires_grad,'grad_value:', params.grad.mean(),'param_value:',params.mean())
        #     # print(name,'grad_value:', params.grad.abs().mean())
        #     if params is not None:
        #         param_total+=params.abs().mean()
        #         if params.grad is not None:
        #             grad_total+=params.grad.abs().mean()
        # print('param before update:', param_total, "grad before update:", grad_total)
        self.optimizer.step()
        # params_before = {name: param.detach().clone() for name, param in self.net.named_parameters()}
        # self.optimizer.step()
        # params_after = {name: param.detach() for name, param in self.net.named_parameters()}
        # diffs = {name: (param_after - param_before).abs().mean() for name, param_before, param_after in zip(params_before.keys(), params_before.values(), params_after.values())}
        # for name, diff in diffs.items():
        #     print(name, diff)
        # sys.exit()

        # param_total=0
        # for name,params in self.net.named_parameters():
        #     # print(name,'grad_requires:',params.requires_grad,'grad_value:', params.grad.mean(),'param_value:',params.mean())
        #     # print(name,'grad_value:', params.grad.abs().mean())
        #     param_total+=params.abs().mean().detach()
        # print('param after update:', param_total)
        self.lr_scheduler.step()
        self.step+=1
        self.loss_v = loss_v.item()

    def inference(self, data):
        # with torch.no_grad():
        self.net.eval()
        output = self.run_network(data)
        return output

    def run_network(self, data):
        # wraping forward function 
        data=data.to(self.device)
        # model_input = model_input.to(self.device)
        output = self.net(data)
        return output

    def save_network(self, save_file=True):
        net = self.net
        state_dict = net.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.to("cpu")
        # print(state_dict)
        if save_file:
            save_filename = f"{self.cfg.savename}_{self.step}.pth"
            save_path = osp.join(self.cfg.work_dir, self.cfg.chkpt_dir)
            mkdir(save_path)
            save_path = osp.join(save_path,save_filename)
            torch.save(state_dict, save_path)
            self._logger.info("Saved network checkpoint to: %s" % save_path)
        return state_dict

    def load_network(self, loaded_net=None):
        if loaded_net is None:
            loaded_net = torch.load(
                self.cfg.load.network_pth_path,
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
            "Checkpoint %s is loaded" % self.cfg.load.network_pth_path)
            

    def save_training_state(self):
        
        save_filename = f"{self.cfg.savename}_{self.step}.state"
        save_path = osp.join(self.cfg.work_dir,self.cfg.chkpt_dir)
        mkdir(save_path)
        save_path = osp.join(save_path, save_filename)
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
            