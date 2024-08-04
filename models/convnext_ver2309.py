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
from timm.models.registry import register_model

class DropBlock1D(nn.Module):
    def __init__(self, drop_prob = 0.1, block_size = 16):
        super(DropBlock1D, self).__init__()
        self.drop_prob = drop_prob
        self.block_size= block_size
        #print(self.block_size // 2)

    def __repr__(self):
        return f'DropBlock1D(drop_prob={self.drop_prob}, block_size={self.block_size})'

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        else:
            gamma = self._compute_gamma(x)
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()
            mask = mask.to(x.device)
            block_mask = self._compute_block_mask(mask)
            #print("block_mask", block_mask[:, None, :].shape)
            #print(block_mask.shape, block_mask[0])
            out = x * block_mask[:, None, :]
            out = x * block_mask.numel() / block_mask.sum()
        
        return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool1d(input=mask[:, None, :],
                            kernel_size=self.block_size, 
                            stride=1,
                            padding=self.block_size // 2)
        if self.block_size % 2 == 0: 
            block_mask = block_mask[:, :, :-1]
        
        block_mask = 1 - block_mask.squeeze(1)
        
        return block_mask

    def _compute_gamma(self, x):
        #return self.drop_prob / (self.block_size)
        return self.drop_prob / (self.block_size ** 2)

class LinearScheduler(nn.Module):
    def __init__(self, dropblock, start_value, stop_value, nr_steps):
        super(LinearScheduler, self).__init__()
        self.dropblock = dropblock
        self.i = 0
        self.drop_values = np.linspace(start=start_value, stop=stop_value, num=int(nr_steps))

    def forward(self, x):
        return self.dropblock(x)

    def step(self):
        if self.i < len(self.drop_values):
            self.dropblock.drop_prob = self.drop_values[self.i]

        self.i += 1 

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, rf_interpretation = '2d'):
        super().__init__()
        #self.dwconv = nn.Conv2d(dim, dim, kernel_size=(1, 5), padding=(0,2), groups=dim) # depthwise conv
        if rf_interpretation == '2d' :
            self.dwconv = nn.Conv2d(dim, dim, kernel_size=(3, 5), padding=(1,2), groups=dim) # depthwise conv
        else :
            self.dwconv = nn.Conv2d(dim, dim, kernel_size=(1, 9), padding=(0,4), groups=dim)
            #self.dwconv = nn.Conv2d(dim, dim, kernel_size=(1, 51), padding=(0,25), groups=dim)
            #self.dwconv = nn.Conv2d(dim, dim, kernel_size=(1, 25), padding=(0,12), groups=dim)
        #self.dwconv = nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim) # depthwise conv
        
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
        #print("gamma = ", self.gamma)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        #x = x.permute(0, 2, 1) # (N, C, L) -> (N, L, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        #x = x.permute(0, 2, 1) # (N, L, C) -> (N, C, L)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        #print(self.training)
        x = input + self.drop_path(x)
        
        return x

class ConvNeXt(nn.Module):
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
    def __init__(self, depths=[3, 3, 18, 3], 
                 dims=[96, 192, 384, 768], 
                 drop_path_rate=0., 
                 drop_prob = 0.2, #0.2,
                 drop_size = 4,
                 num_txrx = 8,
                 box_feature = 'x',
                 layer_scale_init_value= 1e-6, #1.0, #1e-6, 
                 proj_dim = 256,
                 rf_interpretation = '2d'
                 ):
        super().__init__()


        in_chans = 64# channels
        self.input_d = in_chans
        self.depths = depths

        self.box_feature = box_feature
        self.rf_interpretation = rf_interpretation


        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        if rf_interpretation == '2d' :
            stem = nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=(3,3), stride= (3,1), padding = (0, 1)),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
            )
        else : # 1d
            stem = nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=(1,3), stride= (1,3), padding = (0, 0)),
                #nn.Conv1d(in_chans, dims[0], kernel_size=3, stride= 3),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
            )
        self.downsample_layers.append(stem)
        
        
        for i in range(3):
            if i < 4 : #default 0
                if rf_interpretation == '2d' :
                    downsample_layer = nn.Sequential(
                            LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                            nn.Conv2d(dims[i], dims[i+1], kernel_size=(2, 2), stride=(2,2), padding = (0, 0)),
                    )
                else :
                    downsample_layer = nn.Sequential(
                            LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                            nn.Conv2d(dims[i], dims[i+1], kernel_size=(1, 2), stride=(1,2), padding = (0, 0)),
                            #nn.Conv1d(dims[i], dims[i+1], kernel_size=3, stride=1, padding = 1),
                    )
            else :
                if rf_interpretation == '2d' :
                    downsample_layer = nn.Sequential(
                            LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                            nn.Conv2d(dims[i], dims[i+1], kernel_size= (3, 3), stride= (1,1), padding = (1, 1)),
                    )
                else :
                    downsample_layer = nn.Sequential(
                            LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                            nn.Conv2d(dims[i], dims[i+1], kernel_size= (1, 3), stride= (1,1), padding = (0, 1)),
                    )
            self.downsample_layers.append(downsample_layer)
        
        self.activation = nn.GELU()

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        print(f"depth = {sum(depths)} dp_rates = ",len(dp_rates), dp_rates)
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value, rf_interpretation = self.rf_interpretation) for j in range(depths[i])]
            )
            self.stages.append(stage)
            
            
            cur += depths[i]
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.apply(self._init_weights)
        
        self.channel_mixer = nn.ModuleList()
        for i in range(3) :
            tmp_mixer = nn.Conv2d(dims[i+1], proj_dim, kernel_size=(1,1), stride= (1,1))
            self.channel_mixer.append(tmp_mixer)
            
        
        '''
        if drop_prob > 0.:
            self.drop_block = LinearScheduler(
                dropblock = DropBlock1D(block_size=drop_size, drop_prob=0.),
                start_value=0.,
                stop_value =drop_prob,
                nr_steps=2e4 #5e3
            )
            print(self.drop_block.dropblock)
        else:
            self.drop_block = None
        '''
        #print(self.downsample_layers)
        #print(self.stages)
        input_dummy = torch.zeros((48, 64, 1, 768)) # batch, channel, slow_time, fast_time
        self.forward(input_dummy, init_check = True)
        self.num_channels = dims[-1]
        
        freeze_backbone = False
        if freeze_backbone:
            for p in self.parameters():
                p.requires_grad_(False)
        
        
        
        

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, input_tensor, targets=None, init_check = False): #tensor_list: NestedTensor):
        if init_check :
            print("initial sig = ", input_tensor.shape)

        #x = rearrange(input_tensor, 'b ch st ft -> b ch st ft')
        '''
        if self.box_feature != 'x':
            vc = self.vc_transformer(x)
        
        #b, ch, length = input_tensor.shape
        if self.rf_interpretation == '2d' :
            input_tensor = input_tensor.view(b, ch, -1, 2**4)
        else :
            input_tensor = input_tensor.view(b, ch, -1, length)
        '''
        if init_check :
            print("input shape = ", input_tensor.shape)
            
            
        x = self.downsample_layers[0](input_tensor)
        x = self.activation(x)
        x = self.stages[0](x)
        if init_check :
            print("stage 0 = {}".format(x.shape))
        #x0 = x.clone()
        #x0 = rearrange(x0, 'b ch st ft -> b st ft ch')
        #x0 = self.proj_layers[0](x0)

        x = self.downsample_layers[1](x)
        x = self.activation(x)
        x = self.stages[1](x)
        x1 = self.channel_mixer[0](x)
        if init_check :
            print("stage 1 = {}".format(x.shape))
        x1 = rearrange(x1, 'b ch hi wi -> b (hi wi) ch')
        
        
            
        x = self.downsample_layers[2](x)
        x = self.activation(x)
        
        x = self.stages[2](x)
        x2 = self.channel_mixer[1](x)
        if init_check :
            print("stage 2 = {}".format(x.shape))
        x2 = rearrange(x2, 'b ch hi wi -> b (hi wi) ch')
        #x2 = x
        #x2 = x2.unsqueeze(1)



        x = self.downsample_layers[3](x)
        x = self.activation(x)
        x = self.stages[3](x)
        x3 = self.channel_mixer[2](x)
        if init_check :
            print("stage 3 = {}".format(x.shape))
        x3 = rearrange(x3, 'b ch hi wi -> b (hi wi) ch')

        #x = np.array(x)
        #x = x3
        x = torch.cat((x1, x2, x3), 1)

        #b, n, d = x.shape
        #root_t = int(d**0.5)
        #x = rearrange(x, 'b multi_scale ch x -> b multi_scale x ch') ##tokenize
        #mask = torch.zeros((b, root_t, root_t), dtype=torch.bool, device=input_tensor.device)
        #mask = torch.zeros((b, d), dtype=torch.bool, device=input_tensor.device)

        return x #, mask, None




class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
'''
class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.num_channels = backbone.num_channels
        self.box_feature = backbone.box_feature

    def forward(self, tensor, targets=None):
        src, mask, vc = self[0](tensor)
        pos = self[1](src, mask).to(src.dtype)
        out = {'src':src, 'mask':mask, 'pos':pos, 'pred_feature':vc}
        return out
        #return src, mask, pos, vc
'''
def build_convnext(args, in_chans):
    if args.model_scale =='t':
        dims = [64*3, 64*3, 64*3, 64*3]
        res4dim = 3 #3
    elif args.model_scale =='s':
        dims = [64*3, 64*3*2, 64*3*2, 64*3*2]
        #dims = [64*6, 64*6*2, 64*6*2, 64*6*2]
        res4dim = 9
    elif args.model_scale =='m':
        dims = [64*3, 64*3*2, 64*3*2*2, 64*3*2*2]
        res4dim = 27
    elif args.model_scale =='l':
        dims = [64*3, 64*3*2, 64*3*2*2, 64*3*2*2*2]
        res4dim = 27
    
    print(dims)
    
    backbone = ConvNeXt(depths=[3,3, res4dim,3], 
                    dims = dims, 
                    drop_path_rate=args.drop_prob,
                    drop_prob=args.dropblock_prob,
                    drop_size=args.drop_size,
                    num_txrx =args.num_txrx,
                    box_feature=args.box_feature,
                    proj_dim = args.hidden_dim,
                    rf_interpretation = args.rf_interpretation
                    )
    
    
    return backbone

