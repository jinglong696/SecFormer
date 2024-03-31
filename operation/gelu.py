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
import numpy as np
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
input_0 = (torch.rand(100,100)).float().cuda()
input_0 = input_0 * 10

# encrpy inputs
input_0_enc = crypten.cryptensor(input_0)

# gelu_operation


class GELU_FUR(cnn.Module):
    def __init__(self, shape):
        super().__init__()
        self.k = torch.tensor([np.pi/10, 2 * np.pi/10, 3 * np.pi/10, 4 * np.pi/10, 5 * np.pi/10, 6 * np.pi/10, 7 * np.pi/10]).reshape(7,1,1).cuda()
        self.beta = torch.tensor([1.25772, - 0.0299154, 0.382155, -0.0519123, 0.196033, -0.0624557, 0.118029]).cuda()
        self.t = (torch.rand(shape)- 1).cuda() * 2**10 
        self.u = torch.sin(self.k * self.t).permute(1,2,0)
        self.v = torch.cos(self.k * self.t).permute(1,2,0)
        self.t = crypten.cryptensor(self.t)
        self.u = crypten.cryptensor(self.u)
        self.v = crypten.cryptensor(self.v)

        
    def fur_sin(self, x):
        delt = (x - self.t).get_plain_text() 
        delt = delt % (2 * np.pi)
        p = torch.sin(self.k * delt).permute(1,2,0)
        q = torch.cos(self.k * delt).permute(1,2,0)
        s = (self.v * p + self.u * q) 
        output = (s * self.beta).sum(dim = -1)
        return output
    
    def fur_erf(self, x):
        c0 = x < -2.5
        c1 = x < 2.5
        z0 = c0 # x < -2.5
        z1 = c1 - c0 # -2.5 = < x < 2.5
        z2 = 1 - c1 # x >= 2.5
        fur_fitted = self.fur_sin(x)
        return (-z0) + (z1 * fur_fitted) + z2

    def forward(self, x):
        return 0.5 * x * (1 + self.fur_erf(0.7071067811865475 * x))
    
gelu_fur = GELU_FUR(input_0.shape).encrypt()


class puma_gelu(cnn.Module):
    def __init__(self):
        super().__init__()
    
    def poly0(self, x):
        x2 = x.square()
        x3 = x* x2
        return -0.011034134030615728 * x3 - 0.11807612951181953 * x2 -0.42226581151983866 * x - 0.5054031199708174
    
    def poly1(self, x):
        x2 = x.square()
        x3 = x* x2
        x4 = x2.square()
        x6 = x3.square()
        return 0.0018067462606141187 * x6 - 0.037688200365904236 * x4 + 0.3603292692789629 * x2 + 0.5 * x +  0.008526321541038084
    
    def forward(self, x):
        c0 = x < -4
        c1 = x < 1.95
        c2 = x.le(3)

        z0 = c0
        z1 = c1 - c0
        z2 = c2 - c1
        z3 = 1 - c2
        return z1 * self.poly0(x) + z1 * self.poly1(x) + z2 * x

gelu_puma = puma_gelu().encrypt()

n = 3
for i in range(n):    
    
    # gelu_puma test
    t0 = time.time()
    comm0 = comm.get().get_communication_stats()
    with crypten.no_grad():
        input = gelu_puma(input_0_enc)
    t1 = time.time()
    comm1 = comm.get().get_communication_stats()
    timing["gelu_puma time"] += (t1-t0)
    timing["gelu_puma commtime"] += (comm1["time"] - comm0["time"])
    timing["gelu_puma commbyte"] += (comm1["bytes"] - comm0["bytes"])    

    # gelu_fur test
    t0 = time.time()
    comm0 = comm.get().get_communication_stats()
    with crypten.no_grad():
        input = gelu_fur(input_0_enc)
    t1 = time.time()
    comm1 = comm.get().get_communication_stats()
    timing["gelu_fur time"] += (t1-t0)
    timing["gelu_fur commtime"] += (comm1["time"] - comm0["time"])
    timing["gelu_fur commbyte"] += (comm1["bytes"] - comm0["bytes"])  
    
for k,v in timing.items():
    timing[k] = timing[k] / n
print(timing)
