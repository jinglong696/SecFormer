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
input_0 = torch.rand(128,128)

# encrpy inputs
input_0_enc = crypten.cryptensor(input_0)


class SecFormer_Softmax(cnn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def div(self, x , y):
        for i in range(13):
            m = 2 - y
            x = x * m 
            y = y * m 
        return x 
    
    def forward(self, x):
        quad = (x+5) * (x+5)
        quad_sum = quad.sum(dim=self.dim, keepdims=True)        
        quad_reciprocal = self.div(1, quad_sum)        
        return quad * quad_reciprocal
    
class softmax_2QUAD(cnn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        quad = (x+5) * (x+5)
        quad_recip = quad.sum(dim=self.dim, keepdims=True).reciprocal()       
        return quad * quad_recip


class Puma_Softmax(cnn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def exp(self, x):
        c0 = x < 1.0e-14
        c1 = 1 - c0
        return c0 + c1 * x.exp() 
    
    def forward(self, x):
        x_max = x.max(dim = self.dim, keepdim = True)
        x_u = self.exp(x) - x_max[0]
        x_sum = x_u.sum(dim=self.dim, keepdims = True)
        return x_u * x_sum.reciprocal()

secformer_softmax = SecFormer_Softmax(dim = -1).encrypt()
puma_softmax = Puma_Softmax(dim = -1).encrypt()
quad = softmax_2QUAD(dim = -1).encrypt()

n = 3
for i in range(n):
    # secformer softmax
    t0 = time.time()
    comm0 = comm.get().get_communication_stats()
    input = secformer_softmax(input_0_enc)
    t1 = time.time()
    comm1 = comm.get().get_communication_stats()
    timing["secformer softmax time"] += (t1-t0)
    timing["secformer softmax commtime"] += (comm1["time"] - comm0["time"])
    timing["secformer softmax commbyte"] += (comm1["bytes"] - comm0["bytes"])
    
    # puma softmax
    t0 = time.time()
    comm0 = comm.get().get_communication_stats()
    input = puma_softmax(input_0_enc)
    t1 = time.time()
    comm1 = comm.get().get_communication_stats()
    timing["puma softmax time"] += (t1-t0)
    timing["puma softmax commtime"] += (comm1["time"] - comm0["time"])
    timing["puma softmax commbyte"] += (comm1["bytes"] - comm0["bytes"])
    

    # mpcformer softmax
    t0 = time.time()
    comm0 = comm.get().get_communication_stats()
    input = quad(input_0_enc)
    t1 = time.time()
    comm1 = comm.get().get_communication_stats()
    timing["mpcformer softmax time"] += (t1-t0)
    timing["mpcformer softmax commtime"] += (comm1["time"] - comm0["time"])
    timing["mpcformer softmax commbyte"] += (comm1["bytes"] - comm0["bytes"])
    
for k,v in timing.items():
    timing[k] = timing[k] / n
print(timing)










