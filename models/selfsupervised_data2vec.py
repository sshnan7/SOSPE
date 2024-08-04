from functools import partial
import argparse
import torch
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncodingPermute2D 
from timm.models.vision_transformer import PatchEmbed, Block
#from .convnext import build_convbackbone, LayerNorm
from .time_convnext import build_convbackbone, LayerNorm
#from .Light_conv import build_convbackbone, LayerNorm
from einops.einops import rearrange, repeat
from einops.layers.torch import Rearrange
from scipy.optimize import linear_sum_assignment
from torch import einsum, nn
import torch.nn.functional as F
import random
import copy


class SelfsupervisedModel(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, embed_dim=1024, depth=6, num_heads=16, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, use_feature = 'HR'):
        super().__init__()
        

        self.proj_layer1 = nn.Sequential(
            nn.Linear(256, embed_dim, bias=False),
            nn.LayerNorm(embed_dim, eps=1e-06),
        )
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        #pos enc 1d fixed
        self.pos_enc_1d = PositionalEncoding1D(embed_dim)

        self.data_query = nn.Parameter(torch.zeros(1, 16*16, embed_dim))

        #encoder imagefeat -> datavec
        self.imgfeat_transformer_decoder = nn.ModuleList([])
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, batch_first=True, activation="gelu")
        for i in range(depth) :
            single_transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
            self.imgfeat_transformer_decoder.append(single_transformer_decoder)
        
        #to datavec proj
        self.proj_layer2 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.GELU(),
            nn.Linear(embed_dim, 256, bias=False)
        )
        
        ##########################rf#############################
        self.rf_transformer_encoder = nn.ModuleList([])
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, batch_first=True, activation="gelu")
        for i in range(depth) :
            single_transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
            self.rf_transformer_encoder.append(single_transformer_encoder)
        
        self.rf_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.GELU(),
            nn.Linear(embed_dim, 256, bias=False)
        )
        
        rf_embed_dummy = torch.zeros((64, 256, 256), requires_grad = True) # batch channel slowtime fasttime
        img_Feat_dummy = torch.zeros((64, 1, 256, 256), requires_grad = True)
        self.forward(rf_embed_dummy, img_Feat_dummy, 1, init_check = True)
    
    def random_masking(self, x, mask_ratio, init_check = False):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        x = x.clone()
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        if init_check :
            print("before masking :", x.shape)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        if init_check :
            print("after masking :", x_masked.shape)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        if init_check :
            print("mask shape :", mask.shape)

        return x_masked, mask, ids_restore
    
    def imgfeat_patchify(self, x) :
        b, slowt, seq, ch = x.shape
        x = x.view(b, seq, ch)
        
        return x
    
    def forward_img_transformer(self, x, memory, init_check = False) :
        if init_check :
            print("img feature vec transformer")
        
        for i in range(len(self.imgfeat_transformer_decoder)) :
            x = self.imgfeat_transformer_decoder[i](x, memory)
            if i== 0 :
                outputs = x.clone().unsqueeze(3)
            else :
                outputs = torch.cat([outputs, x.clone().unsqueeze(3)], dim = 3)
            
        if init_check :
            print("attention finish x : {}".format(outputs.shape))
                
        x = torch.mean(outputs, dim = 3)
        
        if init_check :
            print("attention mean finish x : {}".format(x.shape))
            
        return x
    
    def forward_rf_transformer(self, x, init_check = False) :
        if init_check :
            print("img feature vec transformer")
        
        for i in range(len(self.rf_transformer_encoder)) :
            x = self.rf_transformer_encoder[i](x)
            if i== 0 :
                outputs = x.clone().unsqueeze(3)
            else :
                outputs = torch.cat([outputs, x.clone().unsqueeze(3)], dim = 3)
            
        if init_check :
            print("attention finish x : {}".format(outputs.shape))
                
        x = torch.mean(outputs, dim = 3)
        
        if init_check :
            print("attention mean finish x : {}".format(x.shape))
            
        return x
    

    def forward(self, rf_latent, img_feat, mask_ratio=0, init_check = False):
        device = img_feat.device
        batch = rf_latent.shape[0]
        
        #stage 1
        img_feat = self.imgfeat_patchify(img_feat)
        #img_feat_target = img_feat.clone()
        
        img_feat_vec = self.proj_layer1(img_feat)
            
        pos_enc = self.pos_enc_1d(img_feat_vec).to(device)
        img_feat_vec = img_feat_vec + pos_enc.detach()
        
        data_query = repeat(self.data_query, '1 len emb -> b len emb', b = batch)
        pos_enc = self.pos_enc_1d(data_query).to(device)
        data_query = data_query + pos_enc.detach()
        
        img_feat_vec = self.forward_img_transformer(data_query, img_feat_vec, init_check = init_check)
        
        img_feat_vec = self.proj_layer2(img_feat_vec)
        if init_check :
            print("img feat vec", img_feat_vec.shape)
            
                
        ##rf stage
        pos_enc = self.pos_enc_1d(rf_latent).to(device)
        rf_latent = rf_latent + pos_enc.detach()
        
        rf_latent = self.forward_rf_transformer(rf_latent, init_check = init_check)
        
        rf_latent = self.rf_proj(rf_latent)

        
        outputs = {
                          "imgfeat_latent" : img_feat_vec,
                          "rf_latent" : rf_latent,
                          "imgfeat_pred" : None
        }
        
        return outputs
        

class selfsupervised_criterion(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.mse = nn.MSELoss()
    
    def forward_loss(self, x, y) :
        
        return self.mse(x, y)
    
    def forward(self, outputs, iter_switch) :
        
        loss1 = self.forward_loss(outputs["imgfeat_latent"], outputs["rf_latent"].detach())
        loss2 = self.forward_loss(outputs["rf_latent"], outputs["imgfeat_latent"].detach())
        if iter_switch == 0 :
            loss_coef = {
                'img-rf' : 1,
                'rf-img' : 0
            }
        if iter_switch == 1 :
            loss_coef = {
                'img-rf' : 0,
                'rf-img' : 1
            }
        
        losses = {
            'img-rf' : loss1,
            'rf-img' : loss2
        }
        
        for key , val in list(losses.items()) :
            if val == None :
                del losses[key]
        
        ##sum loss##
        sum_loss = None
        for i in losses :
            if sum_loss == None :
                sum_loss = loss_coef[i]*losses[i]
            else :
                sum_loss += loss_coef[i]*losses[i]
        
        return losses, sum_loss ##losses for log
        
def build_data2vec_Model(args):
    selfmodel = SelfsupervisedModel( embed_dim=args.hidden_dim, depth=4, num_heads=args.nheads,
        decoder_embed_dim=args.hidden_dim, decoder_depth=6, decoder_num_heads=8, use_feature = args.use_feature)
    
    return selfmodel
        
def build_data2vec_Criterion() :
    selfcriterion = selfsupervised_criterion()
    
    return selfcriterion