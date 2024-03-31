import math
import torch

import crypten
import crypten.communicator as comm
import crypten.nn as cnn
import numpy as np

def encrypt_tensor(input):
    """Encrypt data tensor for multi-party setting"""
    # get rank of current process
    rank = comm.get().get_rank()
    # get world size
    world_size = comm.get().get_world_size()
    assert world_size  == 2
    
    # assumes party 1 is the actual data provider
    src_id = 1

    if rank == src_id:
        input_upd = input.cuda()
    else:
        input_upd = torch.empty(input.size()).cuda()
    private_input = crypten.cryptensor(input_upd, src=src_id)
    return private_input

def encrypt_model(model, modelFunc, config, dummy_input):
    rank = comm.get().get_rank()
    
    if rank == 0:
        model_upd = model.cuda()
    else:
        if isinstance(config, tuple):
            model_upd = modelFunc(config[0], config[1]).cuda()
        else:
            model_upd = modelFunc(config).cuda()

    private_model = model_upd.encrypt(src=0)
    return private_model


class softmax_2RELU(cnn.Module):
    def __init__(self, dim):
        super().__init__()
        self.func = cnn.ReLU()
        self.dim = dim

    def forward(self, x):
        func_x = self.func(x)
        return func_x / func_x.sum(keepdim=True, dim=self.dim)

class softmax_2QUAD(cnn.Module):
    def __init__(self, norm, dim):
        super().__init__()
        self.dim = dim
        self.norm = norm
    
    def forward(self, x):
        a, b, c, d = x.size()
        quad = (x+5) * (x+5)
        quad_recip = quad.sum(dim=self.dim, keepdims=True).reciprocal()
        return quad * quad_recip

class activation_quad(cnn.Module):
    def __init__(self):
        super().__init__()
        self.first_coef = torch.tensor([0.125]).item()
        self.second_coef = torch.tensor([0.5]).item()
        self.third_coef = torch.tensor([0.25]).item()
        self.pow = torch.tensor([2]).item()
     
    def forward(self, x):
        return self.first_coef*x*x + self.second_coef*x + self.third_coef

##################################################################################
class secformer_softmax(cnn.Module):
    def __init__(self, norm, dim):
        super().__init__()
        self.dim = dim
        self.norm = norm
    
    def div(self, x , y):
        for i in range(7):
            m = 2 - y
            x = x * m 
            y = y * m 
        return x 
    
    def forward(self, x):
        a, b, c, d = x.size()
        quad = (x+5) * (x+5)
        quad_sum = quad.sum(dim=self.dim, keepdims=True)
        quad_reciprocal = self.div(1, quad_sum)
        return quad * quad_reciprocal

class puma_softmax(cnn.Module):
    def __init__(self, norm, dim):
        super().__init__()
        self.dim = dim
        self.norm = norm
        self.eps = 1.0e-06
    
    def exp(self, x):
        c0 = x < 1.0e-14
        c1 = 1 - c0
        return c0 + c1 * x.exp() 
    
    def forward(self, x):
        a, b, c, d = x.size()
        x_max = x.max(dim = self.dim, keepdim = True)
        x_u = self.exp(x) - x_max[0] - self.eps
        x_sum = x_u.sum(dim=self.dim, keepdims = True)
        return x_u * x_sum.reciprocal()
    

class gelu_erf(cnn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        input = 0.7071067811865475 * x
        return 0.5 * x * (input.erf()) 

class crypten_gelu(cnn.Module):
    def __init__(self):
        super().__init__()

    def func(self, x):
        return 0.5 * x * (1 + (math.sqrt(2 / math.pi) * (x + 0.044715 * x.pow(3))).tanh())

    def forward(self, x):
        x_p = x.pow(3)
        x_tanh = (math.sqrt(2 / math.pi) * (x + 0.044715 * x_p))
        x_tanh = 1 + x_tanh.tanh()            
        result = 0.5 * x * x_tanh            
        return result

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
    
    

class secformer_gelu(cnn.Module):
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

class crypten_norm(cnn.Module):
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
    
class puma_norm(cnn.Module):
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
    

class secformer_norm(cnn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = crypten.cryptensor(torch.ones(normalized_shape).cuda())
        self.bias = crypten.cryptensor(torch.zeros(normalized_shape).cuda())
        self.eps = eps
        
    def inv_sqrt(self,x):
        gd_iter = 7
        y = 1
        for _ in range(gd_iter):
            m = 0.5 * (3 - x)
            x = x * (m.square())
            y = y * m
        return y

    def forward(self, x):
        x_mean = x.mean(axis=-1, keepdim=True)
        x_u = x - x_mean
        var = x.var(-1, False, True)
        return self.weight * (x_u * self.inv_sqrt(var + self.eps)) + self.bias
