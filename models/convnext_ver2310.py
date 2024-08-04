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

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model


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
        
        '''
        if rf_interpretation == '2d' :
            self.dwconv = nn.Conv2d(dim, dim, kernel_size=(3, 5), padding=(1,2), groups=dim) # depthwise conv
        else :
            self.dwconv = nn.Conv2d(dim, dim, kernel_size=(1, 7), padding=(0,3), groups=dim)
        
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
        #print("gamma = ", self.gamma)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        '''
        if rf_interpretation == '2d' :
            self.dwconv = nn.Conv2d(dim, dim, kernel_size=(3, 5), padding=(1,2), groups=dim) # depthwise conv
        else :
            self.dwconv = nn.Conv2d(dim, 4*dim, kernel_size=(1, 7), padding=(0,3), groups=dim)
        
        self.norm = LayerNorm(4*dim, eps=1e-6)
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
        x = self.norm(x)
        '''
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        '''
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
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
    def __init__(self, in_chans = 64, depths=[3, 3, 18, 3], 
                 dims=[96, 192, 384, 768], 
                 drop_path_rate=0., 
                 drop_prob = 0.2, #0.2,
                 layer_scale_init_value= 1e-6, #1.0, #1e-6, 
                 proj_dim = 256,
                 rf_interpretation = '1d'
                 ):
        super().__init__()

        in_chans = in_chans # channels
        self.in_chans = in_chans
        if in_chans == 1  :
            '''
            downsamples = [8, 4, 4]
            kernels = [downsamples[0], downsamples[1], downsamples[2]]
            paddings = [0, 0, 0]
            dims = []
            for i in range(len(downsamples)) :
                if i == 0 :
                    dims.append(in_chans*4*downsamples[i]) #default 2*d
                else :
                    dims.append(dims[i-1]*downsamples[i]//2)
            depths = [3, 3, 3]
            '''
            downsamples = [1, 1, 3, 4]
            kernels = downsamples
            paddings = [0, 0, 0, 0]
            dims = [in_chans*4*downsamples[0]]
            for i in range(1, len(downsamples)) :
                dims.append(dims[i-1]*downsamples[i]//2)
            depths = [3, 3, 3, 3]
        else :
            #downsamples = [3, 1, 1, 1] #768 256 256 256 baseline
            #downsamples = [1, 3, 1, 1] #768 256 256 256 baseline
            #downsamples = [1, 1, 3, 1] #768 768 256 256 baseline
            #downsamples = [1, 1, 1, 3] #768 768 768 256 baseline
            downsamples = [1,1,3,4] #768 768 256 64 baseline
            #downsamples = [1, 3, 4, 1] #768 256 64 64 baseline
            #downsamples = [1, 3, 1, 4] #768 256 64 64 baseline
            
            dims = [in_chans*4]
            for i in range(1, len(downsamples)) :
                if downsamples[i] == 3 or downsamples[i] == 4 :
                    dims.append(dims[i-1]*2)
                if downsamples[i] == 1 :
                    dims.append(dims[i-1]*1)
            
            kernels = []
            for i in range(len(downsamples)) :
                if downsamples[i] == 1 :
                    kernels.append(3)
                else :
                    kernels.append(downsamples[i])
            paddings = []
            for i in range(len(kernels)) :
                paddings.append((kernels[i]-downsamples[i])//2)
                
        self.depths = depths
        self.proj_dim = proj_dim
        self.rf_interpretation = rf_interpretation

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        # 1d
        if downsamples[0] == 0 :
            self.downsample_layers.append(None)
        else :
            stem = nn.Sequential(
                nn.Conv2d(self.in_chans, dims[0], kernel_size=(1,kernels[0]), stride= (1,downsamples[0]), padding = (0, paddings[0])),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
                )
            self.downsample_layers.append(stem)
        
        for i in range(len(downsamples)-1):
            if downsamples[i+1] == 0 :
                self.downsample_layers.append(None)
            else :
                downsample_layer = nn.Sequential(
                        #LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                        nn.Conv2d(dims[i], dims[i+1], kernel_size=(1, kernels[i+1]), stride=(1,downsamples[i+1]), padding = (0, paddings[i+1])),
                        LayerNorm(dims[i+1], eps=1e-6, data_format="channels_first"),
                )
                self.downsample_layers.append(downsample_layer)
        
        self.activation = nn.GELU()

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        print(f"depth = {sum(depths)} dp_rates = ",len(dp_rates), dp_rates)
        cur = 0
        for i in range(len(downsamples)):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value, rf_interpretation = self.rf_interpretation) for j in range(depths[i])]
            )
            self.stages.append(stage)
            
            cur += depths[i]
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.apply(self._init_weights)
        
        '''
        if in_chans != 1 :
            self.channel_mixer = nn.ModuleList([])
            for i in range(4) :
                tmp_mixer = nn.Conv2d(dims[i], proj_dim, kernel_size=(1,1), stride= (1,1))
                self.channel_mixer.append(tmp_mixer)
        '''
        self.channel_mixer = nn.Conv2d(dims[-1], proj_dim, kernel_size=(1,1), stride= (1,1))
        
            
        #print(self.downsample_layers)
        #print(self.stages)
        input_dummy = torch.zeros((48, self.in_chans, 1,  768)) # batch, channel, slow_time, fast_time
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

    def forward(self, input_tensor, init_check = False): #tensor_list: NestedTensor):
        if init_check :
            print("initial sig = ", input_tensor.shape)

        #x = rearrange(input_tensor, 'b ch st ft -> b ch st ft')
        #b, ch, length = input_tensor.shape
        #input_tensor = input_tensor.view(b, 16, -1, length)
        if init_check :
            print("input shape = ", input_tensor.shape)
        
        x = input_tensor
        for i in range(len(self.downsample_layers)) :
            if self.downsample_layers[i] != None :
                x = self.downsample_layers[i](x)
                x = self.activation(x)
            x = self.stages[i](x)
            if init_check :
                print("stage {} = {}".format(i, x.shape))
        
        x = self.channel_mixer(x)
        
        x = rearrange(x, 'b emb st len -> b (st len) emb')
        
        
        if init_check :
            print("backbone final = {}".format(x.shape))
        
        #x = torch.cat((x1, x2, x3), 1)

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

def build_convnext(args, in_chans, proj_dim):
    if args.model_scale =='t':
        if in_chans == 1 or in_chans == 4 :
            dims = [in_chans*8, in_chans*8*2, in_chans*8*2*2, in_chans*8*2*2*2]
        else :
            #dims = [in_chans*4, in_chans*4*2, in_chans*4*2, in_chans*4*2]
            dims = [in_chans*4, in_chans*4, in_chans*4*2, in_chans*4*2]
            #dims = [in_chans*6, in_chans*6*4, in_chans*6*4, in_chans*6*4]
        res4dim = 3 #3
    elif args.model_scale =='s':
        dims = [3, 3*4, 3*4*4, 3*4*4*4]
        #dims = [64*6, 64*6*2, 64*6*2, 64*6*2]
        res4dim = 9
    elif args.model_scale =='m':
        dims = [3, 64*3*2, 64*3*2*2, 64*3*2*2]
        res4dim = 27
    elif args.model_scale =='l':
        dims = [3, 64*3*2, 64*3*2*2, 64*3*2*2*2]
        res4dim = 27
    

    print(dims)
    
    backbone = ConvNeXt(in_chans = in_chans,
                    depths=[3,3, res4dim,3], 
                    dims = dims, 
                    drop_path_rate=args.drop_prob,
                    drop_prob=args.dropblock_prob,
                    proj_dim = proj_dim,
                    rf_interpretation = args.rf_interpretation
                    )
    
    
    return backbone

