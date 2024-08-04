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

class Lightvit(nn.Module):
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
        
        downsample_len = 256
        
        self.downsample = nn.Sequential(
                    nn.Conv2d(64, dims[0], kernel_size=(1,3), stride= (1,3), padding = (0, 1)),
                    nn.LayerNorm([dims[0], 1, downsample_len], eps=1e-06),
                )
        
        self.convblocks1 = nn.ModuleList()
        
        for i in range(len(dims)) :
            if i == 0 :
                convblock = nn.Sequential(
                    nn.Conv2d(dims[i], dims[i], kernel_size=(1,3), stride= (1,1), padding = (0, 1)),
                    nn.LayerNorm([dims[i], 1, downsample_len], eps=1e-06),
                    nn.Conv2d(dims[i], proj_dim, kernel_size=(1,1), stride= (1,1), padding = (0, 0)), #point wise
                )
            else :
                convblock = nn.Sequential(
                    nn.Conv2d(dims[i-1], dims[i], kernel_size=(1,3), stride= (1,1), padding = (0, 1)),
                    nn.LayerNorm([dims[i], 1, downsample_len], eps=1e-06),
                    nn.Conv2d(dims[i], proj_dim, kernel_size=(1,1), stride= (1,1), padding = (0, 0)), #point wise
                )
            self.convblocks1.append(convblock)
        
        self.vitblocks = nn.ModuleList()
        for i in range(len(dims)) :
            encoder_layer = nn.TransformerEncoderLayer(d_model=proj_dim, nhead=4, dim_feedforward=proj_dim*4, batch_first=True, activation="gelu")
            self.vitblocks.append(encoder_layer)
        
        self.convblocks2 = nn.ModuleList()
        for i in range(len(dims)) :
            convblock = nn.Conv2d(proj_dim, dims[i], kernel_size=(1,1), stride= (1,1), padding = (0, 0)) #point wise
            self.convblocks2.append(convblock)
        
        self.fusion_downsample_block = nn.ModuleList()
        for i in range(len(dims)) :
            convblock = nn.Sequential(
                nn.Conv2d(dims[i]*2, dims[i], kernel_size=(1,3), stride= (1,1), padding = (0, 1)), #point wise
                nn.LayerNorm([dims[i], 1, downsample_len], eps=1e-06)
            )
            self.fusion_downsample_block.append(convblock)
        
        #self.proj_layers = nn.Linear(dims[-1], proj_dim)
        self.proj_layers = nn.Conv2d(dims[-1], proj_dim, kernel_size=(1,1), stride= (1,1), padding = (0, 0)) #point wise embedding
        
        input_dummy = torch.zeros((48, 64, 768)) # batch, channel, slow_time, fast_time
        self.forward(input_dummy, init_check = True)  
        

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
            
    def forward_lightvit_block(self, x, index) :
        x0 = x.clone()
        x = self.convblocks1[index](x)
        b, ch, st, ft = x.shape
        x = rearrange(x, 'b ch st ft -> b (st ft) ch')
        x = self.vitblocks[index](x)
        x = rearrange(x, 'b (st ft) ch -> b ch st ft ', st = st, ft = ft)
        x = self.convblocks2[index](x)
        x = torch.cat((x0, x), 1)
        x = self.fusion_downsample_block[index](x)
        
        return x
    
    def forward(self, input_tensor, targets=None, init_check = False): #tensor_list: NestedTensor):
        if init_check :
            print("input = ", input_tensor.shape)

        x = input_tensor.unsqueeze(2)
        x = self.downsample(x)
        for i in range(len(self.convblocks1)) :
            x = self.forward_lightvit_block(x, i)
            if init_check :
                print("stage {} = {}".format(i, x.shape))
        
        x = self.proj_layers(x)
        x = rearrange(x, 'b ch st ft -> b (st ft) ch')
        if init_check :
            print("stage proj = {}".format(x.shape))

        return x #, mask, None


def build_lightvitbackbone(args):
    if args.model_scale =='t':
        dims = [64*3, 64*3, 64*3, 64*3]

    elif args.model_scale =='s':
        dims = [64*3, 64*3*2, 64*3*2, 64*3*2]

    elif args.model_scale =='m':
        dims = [64*3, 64*3*2, 64*3*2*2, 64*3*2*2]

    elif args.model_scale =='l':
        dims = [64*3, 64*3*2, 64*3*2*2, 64*3*2*2*2]

    
    print(dims)

    backbone = Lightvit(dims = dims)
    
    
    return backbone


