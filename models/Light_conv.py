# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict
from einops.einops import rearrange, repeat
from einops.layers.torch import Rearrange

import numpy as np
import math
import torch
import torch.nn.functional as F
import torchvision
from torch import nn, einsum
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
import random
from torch.nn import BatchNorm1d as Normlayer
torch.backends.cudnn.benchmark = True
#from torch.nn import LayerNorm

#from util.misc import NestedTensor, is_main_process

#from .position_encoding import build_position_encoding

from positional_encodings.torch_encodings import PositionalEncodingPermute1D


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
#from timm.models.registry import register_model

class Lightconv(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, 
                 dims=[96, 192, 384, 768], 
                 proj_dim = 256,
                 ):
        super().__init__()
        
        self.convlayers = nn.ModuleList()
        downsample_len = 256
        
        for i in range(len(dims)) :
            if i == 0 :
                convblock = nn.Sequential(
                    nn.Conv2d(64, dims[i], kernel_size=(1,5), stride= (1,3), padding = (0, 1), bias = True),
                    nn.LayerNorm([dims[i], 1, downsample_len], eps=1e-06)
                )
            elif i == len(dims) -1 :
                convblock = nn.Sequential(
                    nn.Conv2d(dims[i-1], dims[i], kernel_size=(1,3), stride= (1,1), padding = (0, 1), bias = True),
                )
            else :
                convblock = nn.Sequential(
                    nn.Conv2d(dims[i-1], dims[i], kernel_size=(1,3), stride= (1,1), padding = (0, 1), bias = True),
                    nn.LayerNorm([dims[i], 1, downsample_len], eps=1e-06)
                )
            
            self.convlayers.append(convblock)
            
        self.proj_layers = nn.Linear(dims[-1], proj_dim, bias = False)
        
        input_dummy = torch.zeros((48, 64, 768)) # batch, channel, slow_time, fast_time
        self.forward(input_dummy, init_check = True)  
        

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, input_tensor, targets=None, init_check = False): #tensor_list: NestedTensor):
        if init_check :
            print("input = ", input_tensor.shape)

        input_tensor = input_tensor.unsqueeze(2)
        x = self.convlayers[0](input_tensor)
        if init_check :
            print("stage 0 = {}".format(x.shape))

        x = self.convlayers[1](x)
        if init_check :
            print("stage 1 = {}".format(x.shape))
        
        
        x = self.convlayers[2](x)
        #x2 = self.proj_layers[2](x)
        if init_check :
            print("stage 2 = {}".format(x.shape))
        
        x = self.convlayers[3](x)
        if init_check :
            print("stage 3 = {}".format(x.shape))
        #x3 = self.proj_layers[3](x)
        x3 = x.clone()
        x3 = rearrange(x3, 'b ch st ft -> b (st ft) ch')
        x3 = self.proj_layers(x3)
        
        
        #x = torch.cat((x1, x2, x3), 1)
        x = x3 #[x1, x2, x3]#x3 #[x1, x2, x3]

        return x #, mask, None


def build_lightbackbone(args):
    if args.model_scale =='t':
        dims = [64*2, 64*2, 64*2, 64*2]

    elif args.model_scale =='s':
        dims = [64*3, 64*3, 64*3, 64*3]

    elif args.model_scale =='m':
        dims = [64*3, 64*3*2, 64*3*2, 64*3*2]

    elif args.model_scale =='l':
        dims = [64*3, 64*3*2, 64*3*2*2, 64*3*2*2]

    
    print(dims)

    backbone = Lightconv(dims = dims)
    
    
    return backbone


