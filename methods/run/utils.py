import os.path as osp
import numpy as np
import os
import random
import torch
import logging
import time
from typing import TypedDict

class BertInput(TypedDict):
    ids: torch.Tensor
    mask: torch.Tensor
    token_type_ids: torch.Tensor

class ModelInput(TypedDict):
    key: BertInput
    query: BertInput

class ModelOutput(TypedDict):
    key: torch.Tensor
    query: torch.Tensor

def mkdir(path):
    if not osp.exists(path):
        os.makedirs(path)

def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

def savenp(dir,name,a):
    mkdir(dir)
    np.save(osp.join(dir,name),a)

def loadLogger(cfg):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="[ %(asctime)s ] %(message)s",
                                    datefmt="%a %b %d %H:%M:%S %Y")

    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)

    logger.addHandler(sHandler)

    if cfg.work_dir:
        work_dir = osp.join(cfg.work_dir,
                                time.strftime("%Y-%m-%dT%H-%M-%S", time.localtime()))
        mkdir(work_dir)
        cfg.work_dir=work_dir
        fHandler = logging.FileHandler(work_dir + '/log.txt', mode='w')
        fHandler.setLevel(logging.DEBUG)
        fHandler.setFormatter(formatter)

        logger.addHandler(fHandler)


    return logger