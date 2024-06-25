from collections import OrderedDict
import copy
from itertools import chain
import multiprocessing
import os
import pickle
import random
import time
import io
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from einops import rearrange

logger = logging.getLogger("auto_scheduler")
from tvm.auto_scheduler.cost_model.pam_model import PAMModule

DEFAULT_GEMM_BUFFER_SIZE = 8



params = {
            "type": "PAM",
            "fea_size": DEFAULT_GEMM_BUFFER_SIZE,
            "in_dim": 164 + 10, 
            "buf_in_dim": 23,  
            "hidden_dim": 256, # 256 - head 4; 512 - 8
            "mha_hidden_dim": 128, # 256 - head 4; 512 - 8
            "attention_head": 4, # 256 - head 4; 512 - 8
            "out_dim": 1,
        }

def convertckpt(filename, save_filename):
    target_model =  PAMModule(params["in_dim"], params["buf_in_dim"], params["hidden_dim"], params["mha_hidden_dim"],
                        params["out_dim"], params["attention_head"]).cuda()
    model1_state_dict = torch.load(filename)
    few_shot_learning, fea_norm_vec, fea_norm_vec_buf = pickle.load(open(filename.split('.')[0]+'.pkl', 'rb'))
    target_model.load_state_dict(model1_state_dict)


    pickle.dump((target_model,  few_shot_learning, fea_norm_vec, fea_norm_vec_buf),
                open(save_filename, 'wb'))
    

if __name__ == "__main__":
    filename = "./ckpt/a100/500k/fine_tune_pam_a100.pth"
    save_filename = "./ckpt/a100/500k/fine_tune_pam_a100.pkl"
    convertckpt(filename, save_filename)