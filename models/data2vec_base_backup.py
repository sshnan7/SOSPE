# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
import argparse
import torch
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncodingPermute2D 
from timm.models.vision_transformer import PatchEmbed, Block
from .convnext import build_convbackbone, LayerNorm
from einops.einops import rearrange, repeat
from einops.layers.torch import Rearrange
from scipy.optimize import linear_sum_assignment
from torch import einsum, nn
import torch.nn.functional as F

#from util.pos_embed import get_1d_sincos_pos_embed


class Data2Vec_base(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, backbone = None, use_img_feat_enc = False, device = None, task = None, scale_level = 3):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        
        #num_patches = self.patch_embed.num_patches
        
        #################not using in C-Mae#####################
        #self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        #self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        
        #self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)
        self.backbone = backbone
        self.token_num = 256
        self.masked_token_num = 256//4*3
        self.vc_embed_size = 256
        self.img_token_num = 100
        self.use_img_feat_enc = use_img_feat_enc
        self.device = device
        self.batch_size = 64
        self.group_size = 4
        self.task = task
        self.decoder_stages = scale_level
        self.depth = depth #// (self.decoder_stages + 1)
        self.decoder_depth = decoder_depth
        
        self.pos_embed = PositionalEncoding1D(embed_dim)
        self.img_pos_embed = PositionalEncodingPermute2D(embed_dim)
        self.dec_pos_embed = PositionalEncoding1D(embed_dim)
        #self.pos_embed.requires_grad = False
        #networks = nn.ModuleList([])
        
        #-----------------------------------------------------------------
        #stage 1
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, activation="gelu")
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.depth)
        #self.transformer_encoder = transformer_encoder
        '''
        self.transformer_encoder = nn.ModuleList([
            #Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(self.depth)])
        self.norm = norm_layer(embed_dim)
        '''
        #networks.append(self.transformer_encoder)
        #networks.append(self.norm)
        
        self.single_modal_transformer_decoders = nn.ModuleList()
        #self.decoder_stages = 2
        for decoder_stage in range(self.decoder_stages) :
            decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, activation="gelu")
            #transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.decoder_depth)
            self.single_modal_transformer_decoders.append(nn.TransformerDecoder(decoder_layer, num_layers=self.decoder_depth))
        #networks.append(self.single_modal_transformer_decoders)
            
        self.vc_decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.GroupNorm(self.group_size, self.token_num),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim, bias=False)
        )
        #networks.append(self.vc_decoder)
        
        if self.use_img_feat_enc :
            self.img_feature_transformer_enc = nn.TransformerEncoder(encoder_layer, num_layers=self.depth)
            '''
            self.img_feature_transformer_enc = nn.ModuleList([
                #Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(2)])
            self.norm = norm_layer(embed_dim)
            '''
            #networks.append(self.img_feature_transformer_enc)
            #networks.append(self.norm)
        
        #--------------------------------------------
        #stage 2
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, activation="gelu")
        #transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.decoder_depth)
        self.multi_modal_transformer_decoders = nn.TransformerDecoder(decoder_layer, num_layers=self.decoder_depth)
        #networks.append(self.multi_modal_transformer_decoders)
        
        self.cls_MLP = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.GroupNorm(1, 1),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5),
            nn.Linear(embed_dim, embed_dim, bias=False),
        )
        
        self.vec_MLP = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.GroupNorm(self.group_size, self.token_num),
            nn.ReLU(inplace=False),
            nn.Linear(embed_dim, embed_dim, bias=False),
        )
        
        self.SimSiam_cls_MLP = nn.Sequential(
            #nn.Linear(embed_dim, embed_dim, bias=False),
            #nn.BatchNorm1d(embed_dim),
            #nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.GroupNorm(1, 1, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim, bias=False),
        )
        '''
        nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim, affine=False)
        )
        '''
        #networks.append(self.SimSiam_cls_MLP)
        
        self.SimSiam_vec_MLP = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.GroupNorm(self.group_size, self.token_num, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.GroupNorm(self.group_size, self.token_num, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim, bias=False),
        )
        #networks.append(self.SimSiam_vec_MLP)
        
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        #-----------------------------------------------------------------------
        #self.initialize_weights()
        '''
        print("\n\ninit weight")
        
        for network_block in networks :
            for param_tensor in network_block.state_dict():
                print(param_tensor)
            for param in network_block.named_parameters():
                #print(param[0])
                if 'weight' in param[0] and 'norm' not in param[0]:
                    #print(param)
                    #torch.nn.init.xavier_uniform_(param[1])
                    if isinstance(param[0], nn.Linear) :
                        print("catch")
                    print(param[0])
        '''
        #-----------------------------------------------------------------------
        
        dummy = torch.zeros((64, 1, 64, 768))
        img_Feat_dummy = torch.zeros((64, 256, 16, 16))
        self.check_dim(dummy, img_Feat_dummy)
    
    
    '''
    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        #pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        #self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        #decoder_pos_embed = get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        #self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        #w = self.patch_embed.proj.weight.data
        #torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        #torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    '''

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio, init_check = False):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
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
    
    def forward_backbone(self, x, init_check = False) :
        #backbone 
        if init_check :
            print("before embed :", x.shape)
            
        x = self.backbone(x)
        if init_check :
            print("after embed : {}".format(x.shape))
        
        return x
    
    def foward_embedding(self, x, mask_ratio, init_check = False) :
        # embed patches
        # add pos embed w/o cls token
        device = x.device
        for i in range(x.shape[1]) :
            pos_emb = self.pos_embed(x[:,i]).detach().to(device)
            x[:,i] = x[:,i] + pos_emb#[:, 1:, :]
        if init_check :
            print("pos embed sum : {} ".format(x.shape))
        
        # masking: length -> length * mask_ratio
        full_x = x.clone()
        masked_x, mask, ids_restore = self.random_masking(x[:,2], mask_ratio)
        if init_check :
            print("random masking :", masked_x.shape, masked_x.requires_grad)
            print()
        
        return full_x, masked_x, mask, ids_restore
    
    def forward_online_encoder(self, x, mask, init_check = False):
        # Input : embedded x
        # apply Transformer blocks
        if init_check :
            print("online encoder input x :", x.shape)
        
        '''
        for blk in self.transformer_encoder:
            x = blk(x)
        x = self.norm(x)
        '''
        x = self.transformer_encoder(x)
        
        if init_check :
            print("online transformer encoder finish x :", x.shape)
            
        #x = self.SimSiamMLP(x)
        
        #if init_check :
        #    print("online encoder simsiam finish x :", x.shape)
        #    print()

        return x

    def forward_online_transformer_decoder(self, x, memory, init_check=False) :
        for i in range(self.decoder_stages) :
            memory = self.single_modal_transformer_decoders[i](x[:,2-i], memory)
            if init_check :
                print("stage {} memory shape {}".format(i, memory.shape))
            

        return memory

    def forward_target_encoder(self, x, init_check = False):
        # Input : embedded x
        # apply Transformer blocks
        if init_check :
            print("target encoder x :", x.shape)
        '''
        for blk in self.transformer_encoder:
            x = blk(x)
        x = self.norm(x)
        '''
        x = self.transformer_encoder(x)
        if init_check :
            print("target encoder finish x :", x.shape)
        
        x = x.detach()

        return x

    def forward_target_transformer_decoder(self, x, memory, init_check=False) :
        for i in range(self.decoder_stages) :
            memory = self.single_modal_transformer_decoders[i](x[:,2-i], memory)
            if init_check :
                print("stage {} memory shape {}".format(i, memory.shape))
                
        memory = memory.detach()

        return memory

    def forward_vc_decoder(self, RF_vec, init_check=False) :
        x = self.vc_decoder(RF_vec)
        if init_check :
            print("VC shape ", x.shape)
        
        return x

    def forward_imgencoder(self, x, init_check = False):
        # Input :img feature x
        # apply Transformer blocks
        #print(x.shape)
        img_pos_emb = self.img_pos_embed(x).detach()
        device = x.device
        x = x + img_pos_emb.to(device)
        
        if init_check :
            print("img feat encoder input shape :", x.shape)
        
        B, C, h, w = x.shape
        
        x = rearrange(x, 'b ch h w -> b (h w) ch') ##tokenize
    
        #x = x.reshape(B, C, -1)
        if init_check :
            print("img feat encoder reshape shape :", x.shape)
        
        if self.use_img_feat_enc :
            x = self.img_feature_transformer_enc(x)
            '''
            for blk in self.img_feature_transformer_enc:
                x = blk(x)
            x = self.norm(x)
            '''
            if init_check :
                print("img feat transformer encoder shape :", x.shape)
        else :
            x = x.detach()
            
        if init_check :
            print("target encoder grad :", x.requires_grad)
            print()

        return x

    def forward_online_decoder(self, x, memory, init_check = False):
        b, seq, ch = x.shape
        if init_check :
            ("rf_vec shape : {}".format(x.shape))
            ("vc_vec shape : {}".format(memory.shape))
        
        cls_token = torch.zeros((b, 1, ch)).requires_grad_(requires_grad=True)
        
        if init_check == False :
            cls_token = cls_token.to(self.device)
            
        if init_check :
            print("cls token shape : {}".format(cls_token.shape))
            
        x = torch.cat((cls_token, x), 1)
        if init_check :
            print("full rf_vec shape : {}".format(x.shape))
            print("cls token grad", cls_token.requires_grad)
            
        dec_pos_enc = self.dec_pos_embed(x).detach()
        device = x.device
        x = x + dec_pos_enc.to(device)
        
        x = self.multi_modal_transformer_decoders(x, memory)
        if init_check :
            print("trans dec latent shape : {}".format(x.shape))
        
        x_cls = self.cls_MLP(x[:,:1])
        x_vec = self.vec_MLP(x[:,1:])
        
        x_cls = self.SimSiam_cls_MLP(x_cls)
        if init_check :
            print("cls token grad", x_cls.requires_grad)
            
        x_vec = self.SimSiam_vec_MLP(x_vec)
        if init_check :
            print("sim siam cls shape : {}".format(x_cls.shape))
            print("sim siam vec shape : {}".format(x_vec.shape))

        return x_cls, x_vec
        
    def forward_target_decoder(self, x, memory, init_check = False):
        b, seq, ch = x.shape
            
        cls_token = torch.zeros((b, 1, ch))
        
        if init_check == False :
            cls_token = cls_token.to(self.device)
            
        x = torch.cat((cls_token, x), 1)
            
        dec_pos_enc = self.dec_pos_embed(x).detach()
        device = x.device
        x = x + dec_pos_enc.to(device)
        
        x = self.multi_modal_transformer_decoders(x, memory)
        
        x_cls = self.cls_MLP(x[:,:1])
        x_vec = self.vec_MLP(x[:,1:])
        
        x_cls = x_cls.detach()
        x_vec = x_vec.detach()

        return x_cls, x_vec

    

    def forward(self, x, img_feat, mask_ratio=0.25, init_check = False):
        x = self.forward_backbone(x)
        
        ####stage 1####
        full_x, masked_x, mask, ids_restore = self.foward_embedding(x, mask_ratio)
        
        if self.task == 'selfsuperveised' :
            if masked_x.requires_grad == False :
                online_latent = self.forward_online_encoder(full_x[:,2], mask)
            else :
                online_latent = self.forward_online_encoder(masked_x, mask)
            target_latent = self.forward_target_encoder(full_x[:,2])
            online_RF_vec = self.forward_online_transformer_decoder(full_x, online_latent) ### RF vec
            target_RF_vec = self.forward_target_transformer_decoder(full_x, target_latent)
            VC_vec = self.forward_vc_decoder(online_RF_vec)
        else :
            online_latent = self.forward_online_encoder(full_x[:,2], mask)
            online_RF_vec = self.forward_online_transformer_decoder(full_x, online_latent) ### RF vec
                
            VC_vec = self.forward_vc_decoder(online_RF_vec)
        
        ####stage 2####
        if self.task == 'selfsuperveised' :
            img_feat_vec = self.forward_imgencoder(img_feat)
            online_cls, online_vec = self.forward_online_decoder(online_RF_vec, VC_vec)
            target_cls, target_vec = self.forward_target_decoder(target_RF_vec, img_feat_vec)
            online_outputs = {"online_vc": VC_vec, 
                              "online_RF_vec": online_RF_vec,
                              "online_latent_mse" : online_vec,
                              "online_latent_cls" : online_cls
                              }
            target_outputs = {"vc_gt": img_feat_vec, 
                              "RF_vec_gt": target_RF_vec,
                              "target_latent_mse" : target_vec,
                              "target_latent_cls" : target_cls
                              }
        else :
            online_outputs = {"online_vc": VC_vec, 
                              "online_RF_vec": online_RF_vec,
                              "online_latent_mse" : None,
                              "online_latent_cls" : None
                              }
            target_outputs = {"vc_gt": None, 
                              "RF_vec_gt": None,
                              "target_latent_mse" : None,
                              "target_latent_cls" : None
                              }
                    
        return online_outputs, target_outputs
        
    def check_dim(self, x, img_feat, mask_ratio=0.25) :
        print("\n\n ----------------check dimension of data2vec model--------------")
        print("initial grad : ", x.requires_grad)
        x = self.forward_backbone(x, init_check = True)
        full_x, masked_x, mask, ids_restore = self.foward_embedding(x, mask_ratio, init_check = True)
        if self.task == 'selfsuperveised' :
            online_latent = self.forward_online_encoder(masked_x, mask, init_check = True)
            target_latent = self.forward_target_encoder(full_x[:,2], init_check = True)
            online_RF_vec = self.forward_online_transformer_decoder(full_x, online_latent, init_check = True) ### RF vec
            target_RF_vec = self.forward_target_transformer_decoder(full_x, target_latent, init_check = True)
            print("stage 1\nonline gradient : {} \ntartget gradient : {}".format(online_RF_vec.requires_grad, target_RF_vec.requires_grad))
        else :
            online_latent = self.forward_online_encoder(full_x[:,2], mask, init_check = True)
            online_RF_vec = self.forward_online_transformer_decoder(full_x, online_latent, init_check = True)
            print("stage 1\nonline gradient : {} \n".format(online_RF_vec.requires_grad))
        VC_vec = self.forward_vc_decoder(online_RF_vec, init_check = True)
        if self.task == 'selfsuperveised' :
            img_feat_vec = self.forward_imgencoder(img_feat, init_check = True)
            online_cls, online_vec = self.forward_online_decoder(online_RF_vec, VC_vec, init_check = True)
            target_cls, target_vec = self.forward_target_decoder(target_RF_vec, img_feat_vec, init_check = True)
            print("stage 2\nonline gradient : {} \ntartget gradient : {}".format(online_RF_vec.requires_grad, target_RF_vec.requires_grad))
        #pred = self.forward_decoder(online_latent, ids_restore, init_check = True)  # [N, L, p*p*3]
        #loss = self.forward_loss(imgs, pred, mask, init_check = True)
        #return loss, pred, mask


class SetCriterion(nn.Module):
    def __init__(self, use_vc_loss = True, use_RF_vec_loss = False, final_loss_type = 'mse', device = None, vc_loss_coef = 1, 
                RF_vec_loss_coef = 1, final_loss_mse_coef = 1, final_loss_cls_coef = 1):
        super().__init__()
        
        self.temperature = nn.Parameter(torch.Tensor([1.]))
        self.device = device
        self.coefs = {
            'vc' : torch.tensor(vc_loss_coef).to(self.device),
            'RF_vec' : torch.tensor(RF_vec_loss_coef).to(self.device),
            'final_mse' : torch.tensor(final_loss_mse_coef).to(self.device),
            'final_cls' : torch.tensor(final_loss_cls_coef).to(self.device)
        }
        
        if use_vc_loss :
            self.vc_loss = nn.MSELoss(reduction='mean').to(device)
        else :
            self.vc_loss = None
            
        if use_RF_vec_loss :
            self.RF_vec_loss = nn.CosineSimilarity(dim=1, eps=1e-6).to(device) #nn.MSELoss(reduction='mean').to(device)
        else :
            self.RF_vec_loss = None
            
            self.final_loss_mse = nn.MSELoss(reduction='mean').to(device)
            self.final_loss_cls = nn.CrossEntropyLoss #nn.CosineSimilarity(dim=1, eps=1e-6).to(device)
        
        self.grad_check()
        
    def forward_vc_loss(self, online_vc, vc_gt) :
        loss = self.vc_loss(online_vc, vc_gt)
        
        return loss
    
    def forward_RF_vec_loss(self, online_RF_vec, RF_vec_gt) :
        loss = self.RF_vec_loss(online_RF_vec, RF_vec_gt)
        
        return loss
        
    def forward_final_loss_mse(self, online_latent, target_latent) :
        loss = self.final_loss_mse(online_latent, target_latent)
        
        return loss
    
    def forward_final_loss_cls(self, online_latent, target_latent) :
        batch_size = online_latent.shape[0]
        '''
        tau = 0.07
        for i in range(batch_size) :
            cos_sim = self.final_loss_cls(online_latent[i].clone().unsqueeze(0), target_latent)
            pos_pair = torch.exp(cos_sim[i].clone()/tau)
            neg_pairs = torch.sum(torch.exp(torch.cat((cos_sim[:i], cos_sim[i+1:]))/tau))
            temp_loss = -torch.log(pos_pair/(pos_pair+neg_pairs)).unsqueeze(0)
            if i == 0 :
                losses = temp_loss.clone()
            else :
                losses = torch.cat((losses, temp_loss.clone()))
        
        #loss = torch.tensor(1).to(self.device) - loss
        loss = torch.mean(losses)
        '''
        device = online_latent.device
        online_latent = online_latent.squeeze(1)
        target_latent = target_latent.squeeze(1)
        sim = einsum('i d, j d -> i j', online_latent, target_latent)
        sim = sim * self.temperature.exp()
        contrastive_labels = torch.arange(batch_size, device=device)
        
        ce = F.cross_entropy
        contrastive_loss = (ce(sim, contrastive_labels) + ce(sim.t(), contrastive_labels)) * 0.5
        #contrastive_loss = contrastive_loss
        return contrastive_loss
    
    def forward(self, online_outputs, target_outputs) :
        if self.vc_loss != None :
            vc_loss = self.forward_vc_loss(online_outputs["online_vc"], target_outputs["vc_gt"])
        else :
            vc_loss = None
            
        if self.RF_vec_loss != None :
            RF_vec_loss = self.forward_RF_vec_loss(online_outputs["online_RF_vec"], target_outputs["RF_vec_gt"])
        else :
            RF_vec_loss = None
        
            final_loss_mse = self.forward_final_loss_mse(online_outputs["online_latent_mse"], target_outputs["target_latent_mse"])
            final_loss_cls = self.forward_final_loss_cls(online_outputs["online_latent_cls"], target_outputs["target_latent_cls"])
            
        losses = {
            'vc' : vc_loss,
            'RF_vec' : RF_vec_loss,
            'final_mse' : final_loss_mse,
            'final_cls' : final_loss_cls
        }
        #print(losses['final_mse'])
        #print(losses)
        sum_loss = torch.zeros(1).to(self.device)
        
        for key , type_ck in list(losses.items()) :
            if type_ck == None :
                del losses[key]
        
        ##for log update##
        for i in losses :
            if losses[i] != None :
                #print(self.coefs[i].device)
                sum_loss += self.coefs[i]*losses[i]
        
        return losses, sum_loss ##losses for log
        
    def grad_check(self, vc_loss = torch.ones(1, requires_grad=True), RF_vec_loss = None, final_loss_mse = torch.ones(1, requires_grad=True), final_loss_cls = None) :
        print('\n\n---------------grad check for loss----------- losses has to be False, sum loss has to be True-------------------')
        losses = {
            'vc' : vc_loss,
            'RF_vec' : RF_vec_loss,
            'final_mse' : final_loss_mse,
            'final_cls' : final_loss_cls
        }
        
        sum_loss = torch.zeros(1).to(self.device)
        
        for i in losses :
            if losses[i] != None :
                losses[i] = losses[i].to(self.device)
                #print(self.coefs[i].device, losses[i].device)
                sum_loss += self.coefs[i]*losses[i]
        
        for i in losses :
            if losses[i] != None :
                losses[i] = losses[i].detach()
                
        print("losses grad :{}\nsum_loss grad :{}".format(losses['vc'].requires_grad, sum_loss.requires_grad))
        
class pose_model_petr(nn.Module):
    def __init__(self, embed_dim = 256, num_heads = 8, num_queries = 15, multi_modal_depth = 2, pose_dec_depth = 2,device = None, use_vc = False, feature_gt = False, use_only_gt = False):
        super().__init__()
        
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.device = device
        self.keypoint_num = 16
        self.group_size = 5
        self.use_vc = use_vc
        self.feature_gt = feature_gt
        self.use_only_gt = use_only_gt
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, activation="gelu")
        if self.use_vc :
            self.multimodal_decoder = nn.TransformerDecoder(decoder_layer, num_layers=multi_modal_depth)
            if self.feature_gt :
                self.feat_pos = PositionalEncoding1D(embed_dim)
        vc_encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, activation="gelu")
        self.vc_attention = nn.TransformerEncoder(vc_encoder_layer, num_layers=2)
        self.person_decoder = nn.TransformerDecoder(decoder_layer, num_layers=pose_dec_depth)
        self.pose_decoder = nn.TransformerDecoder(decoder_layer, num_layers=pose_dec_depth)
        self.pos_embed = PositionalEncoding1D(embed_dim)
        self.cls_MLP = nn.Sequential(
            nn.Linear(embed_dim, 1, bias=False),
            nn.Sigmoid(),
            #nn.BatchNorm1d(embed_dim),
            #nn.ReLU(inplace=True),
            #nn.Linear(embed_dim, embed_dim, bias=False),
            #nn.BatchNorm1d(embed_dim, affine=False)
        )
        self.pos_MLP = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.GroupNorm(self.group_size, self.num_queries),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.2),
            nn.Linear(embed_dim, self.keypoint_num*2, bias=False),
            nn.Sigmoid(),
        )
        
        if self.feature_gt :
            vc = torch.zeros((64, 17*16, 256), requires_grad=False)
        else :
            vc = torch.zeros((64, 17*16, self.embed_dim), requires_grad=False)
        memory = torch.zeros((64, 256, self.embed_dim), requires_grad=False)
        #vc = torch.zeros((64, 17*16, self.embed_dim), requires_grad=False)
        self.forward(memory, vc, init_check=True)
        
    def forward(self, memory, visual_clue, init_check=False) :
        if not self.use_only_gt :
            b, t, ch = memory.shape
            device = memory.device
            if init_check :
                print("memory shape : ", memory.shape)
        
        if self.use_vc :
            if self.feature_gt :
                vc = visual_clue
                vc_pos = self.feat_pos(vc)
                vc = vc + vc_pos
                vc = self.vc_attention(vc)
            if self.use_only_gt :
                memory = vc
            else :
                memory = self.multimodal_decoder(memory, vc)
        
        b, t, ch = memory.shape
        device = memory.device
        
        person_queries = torch.zeros((b, self.num_queries, ch), requires_grad=True).to(device)
        pos_encoding = self.pos_embed(person_queries).to(device)
        if init_check  == False :
            pos_encoding = pos_encoding.to(self.device)
            person_queries = person_queries.to(self.device)
        
        person_queries = person_queries + pos_encoding
        
        x = self.person_decoder(person_queries, memory)
        cls = self.cls_MLP(x)
        pos = self.pos_MLP(x)
        if init_check :
            print("cls shape : ", cls.shape)
            print("pos shape : ", pos.shape)
            print("grad : ", cls.requires_grad)
        
        out = {'pred_logits': cls, 'pred_keypoint': pos, 'vc' : vc}
            
        return out

class Pose_Criterion(nn.Module):
    def __init__(self, use_box_loss = False,  device = None, vc_coef = 0, keypoint_coef = 1, class_coef = 1, box_mse_coef = 1, matcher_keypoint_coef = 1, matcher_class_coef = 1, vc_learn = False):
        super().__init__()
        
        self.device = device
        self.match_cost_class = matcher_class_coef
        self.match_cost_keypoint = matcher_keypoint_coef
        
        self.cost_class = class_coef
        self.cost_keypoint = keypoint_coef
        self.cost_vc = vc_coef
        
        self.vc_learn = vc_learn
        if self.vc_learn :
           self.vc_loss = nn.MSELoss(reduction='mean').to(device)
        
        self.matchcoef = {
            'keypoint' : torch.tensor(matcher_keypoint_coef).to(self.device),
            'class' : torch.tensor(matcher_class_coef).to(self.device)
        }
        
        if use_box_loss :
            self.box_loss = nn.MSELoss(reduction='mean').to(device)
        else :
            self.box_loss = None
        
        self.class_loss = nn.BCELoss().to(device)
        self.keypoint_loss = nn.L1Loss(reduction='mean').to(device) #nn.MSELoss().to(device)#nn.L1Loss(reduction='mean').to(device)
        
        
        output_ex = []
        gt_ex = []
        outputs_class = torch.rand(64, 15, 1)
        outputs_keypoint = torch.rand(64, 15, 2*16)
        output_vc = torch.rand(64, 256, 256)
        out_ex = {'pred_logits': outputs_class, 'pred_keypoint': outputs_keypoint, 'vc' : output_vc}
        for i in range(32) :
            gt_class = torch.ones((3, 1))
            gt_keypoint = torch.rand(3, 2*16)
            gt_feat = torch.rand(256, 256)
            tmp_gt_ex = {'labels' : gt_class, 'cd' : gt_keypoint, 'features' : gt_feat}
            gt_ex.append(tmp_gt_ex)
        for i in range(16) :
            gt_class = torch.tensor([])
            gt_keypoint = torch.tensor([])
            tmp_gt_ex = {'labels' : gt_class, 'cd' : gt_keypoint, 'features' : gt_feat}
            gt_ex.append(tmp_gt_ex)
        for i in range(16) :
            gt_class = torch.ones((2, 1))
            gt_keypoint = torch.rand(2, 2*16)
            tmp_gt_ex = {'labels' : gt_class, 'cd' : gt_keypoint, 'features' : gt_feat}
            gt_ex.append(tmp_gt_ex)
        #self.matching_init_test(out_ex, gt_ex)
        self.forward(out_ex, gt_ex, init_check=True)
        
    def matching(self, outputs, gt, init_check=False) :
        batch_row = [] #row is output
        batch_col = [] #col is tgt
        neg_idx = []
        
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"]#.softmax(-1)  # [batch_size * num_queries, num_classes]
        out_keypoint = outputs["pred_keypoint"]  # [batch_size * num_queries, 2*keypoints]
        
        for i in range(bs) :
            temp_tgt_id = gt[i]["labels"]
            temp_tgt_id[:] = 0
            temp_tgt_keypoint = gt[i]["cd"]
            if temp_tgt_keypoint.shape[0] == 0 :
                batch_row.append(torch.tensor([]))
                batch_col.append(torch.tensor([]))
                continue
            
            temp_out_prob = outputs["pred_logits"][i]
            temp_out_keypoint = outputs["pred_keypoint"][i]
            
            temp_cost_class = 1-temp_out_prob
        
            # Compute the L1 cost between keypoint
            temp_cost_keypoint = torch.cdist(temp_out_keypoint, temp_tgt_keypoint, p=1)
                
            # Final cost matrix
            temp_C = self.match_cost_class * temp_cost_class + self.match_cost_keypoint * temp_cost_keypoint
            temp_C = temp_C.cpu().detach().numpy() #linear sum assignment can't work at torch
            #C = C.view(bs, num_queries, -1).cpu()
            
            row, col = linear_sum_assignment(temp_C)
            
            batch_row.append(torch.as_tensor(row, dtype=torch.int64))
            batch_col.append(torch.as_tensor(col, dtype=torch.int64))
        
        return batch_row, batch_col
    
    def forward(self, output, gt, init_check=False) :
        if init_check :
            print("------------------------check loss func works-----------------------------")
        batch_row, batch_col = self.matching(output, gt)
        
        bs, queries = output["pred_logits"].shape[:2]
        device = output["pred_logits"].device
        gt_class = torch.zeros((bs, queries, 1)).to(device)
        if init_check :
            print("gt class grad check :", gt_class.requires_grad)
        class_loss = None #torch.tensor(0)
        keypoint_loss = None #torch.tensor(0)
        vc_loss = None
        
        non_less_bs = 0
        first_gt_class = []
        for i in range(bs) :
            tmp_gt_class = gt_class[i].clone()
            gt_len = batch_row[i].shape[0]
            tmp_rows = batch_row[i]
            tmp_cols = batch_col[i]
            if tmp_rows.shape[0] == 0 :
                continue
            
            for j in range(gt_len) :
                tmp_gt_class[tmp_rows[j]] = 1
                if j == 0 :
                    batch_output_coor = output['pred_keypoint'][i][tmp_rows[j]]
                    batch_gt_coor = gt[i]["cd"][tmp_cols[j]]
                else :
                    batch_output_coor = torch.cat((batch_output_coor, output['pred_keypoint'][i][tmp_rows[j]]))
                    batch_gt_coor = torch.cat((batch_gt_coor, gt[i]["cd"][tmp_cols[j]]))
            
            if i == 0 :# for print
                first_gt_class.append(tmp_gt_class)
            
            if class_loss == None :
                class_loss = self.class_loss(output["pred_logits"][i], tmp_gt_class)
            else :
                class_loss = class_loss + self.class_loss(output["pred_logits"][i], tmp_gt_class)
            
            if keypoint_loss == None :
                keypoint_loss = self.keypoint_loss(batch_output_coor, batch_gt_coor)
            else :
                keypoint_loss = keypoint_loss + self.keypoint_loss(batch_output_coor, batch_gt_coor)
            non_less_bs += 1
            
            if self.vc_learn :
                tokens, emb = output['vc'][i].shape
                gt_feat = gt[i]["features"].view(tokens, emb)
                if vc_loss == None :
                    vc_loss = self.vc_loss(output['vc'][i], gt_feat)
                else :
                    vc_loss = vc_loss + self.vc_loss(output['vc'][i], gt_feat)
        
        if class_loss == None :
            class_loss = torch.tensor(0)
        if keypoint_loss == None :
            keypoint_loss = torch.tensor(0)
        if vc_loss == None :
            vc_loss = torch.tensor(0)
            
        losses = {
            'cls' : class_loss,
            'cd' : keypoint_loss,
            'vc' : vc_loss
        }
        
        if init_check :
            print("class grad {} // cd grad {}", class_loss.requires_grad, keypoint_loss.requires_grad)
            print(class_loss, keypoint_loss, vc_loss)
            
        sum_loss = self.cost_class*class_loss/non_less_bs + self.cost_keypoint*keypoint_loss/non_less_bs + self.cost_vc*vc_loss/non_less_bs
        
        return losses, sum_loss, first_gt_class
        


def build_model(args):   
    device = torch.device(args.device)

    convbackbone = build_convbackbone(args)

    model = Data2Vec_base(
        embed_dim=256, depth=args.enc_layers, num_heads=8,
        decoder_embed_dim=256, decoder_depth=args.dec_layers, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), backbone= convbackbone, use_img_feat_enc = args.use_img_feat_enc, device = device, task = args.task, scale_level = args.multi_scale)
    
    criterion = SetCriterion(use_vc_loss = True, use_RF_vec_loss = False, final_loss_type = 'mse', device = device, vc_loss_coef = 0, 
                RF_vec_loss_coef = 0, final_loss_mse_coef = 0, final_loss_cls_coef = 1)
    criterion.to(device)
    model.to(device)
    
    if args.task != 'selfsupervised' :
        pose_model = pose_model_petr(multi_modal_depth = args.cross_dec_layers, device = device, use_vc = args.use_vc, feature_gt = args.feature_gt, use_only_gt = args.use_only_gt)
        pose_model.to(device)
        if args.task == 'scratch' and args.use_vc:
            criterion = Pose_Criterion(vc_coef = args.feature_loss_coef, keypoint_coef = args.keypoint_loss_coef, class_coef = args.cls_loss_coef, matcher_keypoint_coef = args.set_cost_keypoint, matcher_class_coef = args.set_cost_class, vc_learn= True)
            if args.feature_gt :
                criterion = Pose_Criterion(vc_coef = 0, keypoint_coef = args.keypoint_loss_coef, class_coef = args.cls_loss_coef, matcher_keypoint_coef = args.set_cost_keypoint, matcher_class_coef = args.set_cost_class, vc_learn= True)
        else :
            criterion = Pose_Criterion(keypoint_coef = args.keypoint_loss_coef, class_coef = args.cls_loss_coef, matcher_keypoint_coef = args.set_cost_keypoint, matcher_class_coef = args.set_cost_class)
        criterion.to(device)
        
        return model, pose_model, criterion
        
    else :
        return model, criterion

'''
def get_args_parser():
    parser = argparse.ArgumentParser('self supervised learning testing', add_help=False)
    parser.add_argument('--device', default=0, type=float) # 0.0004
    parser.add_argument('--stack_num', default=1, type=int) # 0.0004
    parser.add_argument('--frame_skip', default=1, type=int) # 0.0004
    parser.add_argument('--res4dim', default=6, type=int) # 0.0004
    parser.add_argument('--drop_prob', default=0, type=float) # 0.0004
    parser.add_argument('--dropblock_prob', default=0, type=float) # 0.0004
    parser.add_argument('--drop_size', default=0, type=float) # 0.0004
    parser.add_argument('--num_txrx', default=8, type=float) # 0.0004
    parser.add_argument('--box_feature', default='x', type=str) # 0.0004
    parser.add_argument('--hidden_dim', default=256, type=int) # 0.0004
    
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser('self supervised learning testing', parents=[get_args_parser()])
    args = parser.parse_args()
    
    build_model(args)
'''