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

crypten.init()
cfg.communicator.verbose = True

# setup fake data for timing purpose
commInit = crypten.communicator.get().get_communication_stats()

# dummy inputs
input_0_enc_list = []
input_1_enc_list = []
for i in range(1,11):
    input_0 = torch.rand(i*100,i*100)
    input_1 = torch.rand(i*100,i*100)
    input_0_enc = crypten.cryptensor(input_0)
    input_1_enc = crypten.cryptensor(input_1)
    input_0_enc_list.append(input_0_enc)
    input_1_enc_list.append(input_1_enc)
    
def reset_timing():
    for k,v in timing.items():
        timing[k] = 0
            
def secformer_div(x , y):
    for i in range(7):
        m = 2 - y
        x = x * m 
        y = y * m 
    return x 
    

for input_0_enc,input_1_enc in zip(input_0_enc_list,input_1_enc_list):
    print(input_0_enc.shape, input_0_enc.shape)
    n = 10
    for i in range(n):
        # crypten division
        t0 = time.time()
        comm0 = comm.get().get_communication_stats()
        input = input_0_enc @ input_1_enc.reciprocal()
        t1 = time.time()
        comm1 = comm.get().get_communication_stats()
        timing["crypten div time"] += (t1-t0)
        timing["crypten div commtime"] += (comm1["time"] - comm0["time"])
        timing["crypten div commbyte"] += (comm1["bytes"] - comm0["bytes"])

        # my division
        t0 = time.time()
        comm0 = comm.get().get_communication_stats()
        input = secformer_div(input_0_enc, input_1_enc)
        t1 = time.time()
        comm1 = comm.get().get_communication_stats()
        timing["my div time"] += (t1-t0)
        timing["my div commtime"] += (comm1["time"] - comm0["time"])
        timing["my div commbyte"] += (comm1["bytes"] - comm0["bytes"])

    for k,v in timing.items():
        timing[k] = timing[k] / n
    print(timing)
    reset_timing()










