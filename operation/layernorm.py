import sys
import os
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
#from transformers import AutoConfig, BertForSequenceClassificationWrapper

import crypten
import crypten.nn as cnn
import crypten.communicator as comm
from crypten.config import cfg

import math

# 2PC setting
rank = sys.argv[1]
os.environ["RANK"] = str(rank)
os.environ["WORLD_SIZE"] = str(2)
os.environ["MASTER_ADDR"] = '127.0.0.1'
os.environ["MASTER_PORT"] = "29508"
os.environ["RENDEZVOUS"] = "env://"
os.environ["GLOO_SOCKET_IFNAME"] = 'eth0'


timing = defaultdict(float)

# soft_tiny
crypten.init()
cfg.communicator.verbose = True

# setup fake data for timing purpose
commInit = crypten.communicator.get().get_communication_stats()

# dummy inputs
input_0 = (torch.rand(1280, 768)).float().cuda()


# encrpy inputs
input_0_enc = crypten.cryptensor(input_0)

class Crypten_norm(cnn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super().__init__()
        self.weight = crypten.cryptensor(torch.ones(normalized_shape).cuda())
        self.bias = crypten.cryptensor(torch.zeros(normalized_shape).cuda())
        self.eps = eps
        
    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        y = x - mean
        var = x.var(-1, False, True)
        var_sqrt = (self.eps + var).sqrt() 
        var_inv_sqrt = var_sqrt.reciprocal()
        return  self.weight*(y * var_inv_sqrt) + self.bias
    
class Puma_norm(cnn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super().__init__()
        self.weight = crypten.cryptensor(torch.ones(normalized_shape).cuda())
        self.bias = crypten.cryptensor(torch.zeros(normalized_shape).cuda())
        self.eps = eps
        
    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        y = x - mean
        var = x.var(-1, False, True)
        var_sqrt = (self.eps + var).inv_sqrt() 
        x = y * var_sqrt
        return x * self.weight + self.bias
    

class SecFormer_norm(cnn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = crypten.cryptensor(torch.ones(normalized_shape).cuda())
        self.bias = crypten.cryptensor(torch.zeros(normalized_shape).cuda())
        self.eps = eps
        
    def inv_sqrt(self,x):
        gd_iter = 8
        y = 1
        for _ in range(gd_iter):
            m = 0.5 * (3 - x)
            x = x.mul_(m.square())
            y = y * m
        return y

    def forward(self, x):
        x_mean = x.mean(axis=-1, keepdim=True)
        x_u = x - x_mean
        var = x.var(-1, False, True)
        return (x_u * self.inv_sqrt(var + self.eps))*self.weight + self.bias
    
crypten_norm = Crypten_norm(768).encrypt()
puma_norm = Puma_norm(768).encrypt()
secformer_norm =  SecFormer_norm(768).encrypt()


n = 3
for i in range(n):
    
    # my layernorm
    t0 = time.time()
    comm0 = comm.get().get_communication_stats()
    input = secformer_norm(input_0_enc)
    t1 = time.time()
    comm1 = comm.get().get_communication_stats()
    timing["secformer norm time"] += (t1-t0)
    timing["secformer norm commtime"] += (comm1["time"] - comm0["time"])
    timing["secformer norm commbyte"] += (comm1["bytes"] - comm0["bytes"])
    
    # crypten layernorm
    t0 = time.time()
    comm0 = comm.get().get_communication_stats()
    input = crypten_norm(input_0_enc)
    t1 = time.time()
    comm1 = comm.get().get_communication_stats()
    timing["crypt norm time"] += (t1-t0)
    timing["crypt norm commtime"] += (comm1["time"] - comm0["time"])
    timing["crypt norm commbyte"] += (comm1["bytes"] - comm0["bytes"])
    
    # puma layernorm
    t0 = time.time()
    comm0 = comm.get().get_communication_stats()
    input = puma_norm(input_0_enc)
    t1 = time.time()
    comm1 = comm.get().get_communication_stats()
    timing["puma norm time"] += (t1-t0)
    timing["puma norm commtime"] += (comm1["time"] - comm0["time"])
    timing["puma norm commbyte"] += (comm1["bytes"] - comm0["bytes"])
    


for k,v in timing.items():
    timing[k] = timing[k] / n
print(timing)
