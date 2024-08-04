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
#from .convnext import build_convbackbone, LayerNorm
from .time_convnext import build_convbackbone, LayerNorm
#from .Light_conv import build_convbackbone, LayerNorm
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
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, backbone = None, use_img_feat_enc = False, device = None, task = None, 
                 scale_level = 3, attentional_pooling = True, scale_order = 'small_first', slow_time = 1):
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
        self.decoder_stages = scale_level - 1 # one scale use in encoder
        self.depth = depth #// (self.decoder_stages + 1)
        self.decoder_depth = decoder_depth
        self.scale_order = scale_order
        
        self.mask_pos = 3#10 #1,2,3
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.local_attention = True
        if self.local_attention :
            self.local_window_size = 4 #resol : 4m / rf_len * window_size
            
        self.using_scale = [2]#[0,1,2]
        
        self.attention_type = ["local", "global"] #["local", "local"] #["local", "global"]
        
        self.slow_time = slow_time
        self.rf_len = 768
        self.downsample = 3
        #token_lens = [128*slow_time, 64*slow_time, 32*slow_time]
        token_lens = [int(768/self.downsample*self.slow_time), int(768/self.downsample*self.slow_time), int(768/self.downsample*self.slow_time)]
        #token_lens = [64*slow_time, 16*slow_time, 4*slow_time]
        self.pos_embeds = nn.ParameterList() #nn.ModuleList()
        
        #each cnn scaled feature position emb
        for i in range(3) :
            pos_embedding = nn.Parameter(torch.zeros(1, token_lens[i], embed_dim))
            self.pos_embeds.append(pos_embedding)
        
        #VC position emb
        pos_embedding = nn.Parameter(torch.zeros(1, 256, embed_dim))
        self.pos_embeds.append(pos_embedding)
        
        #self.img_pos_embed = PositionalEncodingPermute2D(embed_dim)
        #self.dec_pos_embed = PositionalEncoding1D(embed_dim)
        self.attentional_pooling = attentional_pooling
        print("att pool : ", self.attentional_pooling)
        #self.pos_embed.requires_grad = False
        #networks = nn.ModuleList([])
        
        #-----------------------------------------------------------------
        #stage 1
        self.transformer_encoders = nn.ModuleList([])
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, batch_first=True, activation="gelu")
        #for i in range(len(token_lens)) :
        self.transformer_encoders.append(nn.TransformerEncoder(encoder_layer, num_layers=self.depth))
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
        for decoder_stage in range(len(self.using_scale)-1) : #
            decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, batch_first=True, activation="gelu")
            #transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.decoder_depth)
            self.single_modal_transformer_decoders.append(nn.TransformerDecoder(decoder_layer, num_layers=self.decoder_depth))
        #networks.append(self.single_modal_transformer_decoders)
        
        self.vc_query = nn.Parameter(torch.zeros(1, 16*16, embed_dim))
        self.VC_query_pos_embed = PositionalEncoding1D(embed_dim)
        vc_decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, batch_first=True, activation="gelu")
        self.vc_decoder = nn.TransformerDecoder(vc_decoder_layer, num_layers=self.decoder_depth)
        self.vc_proj = nn.Linear(embed_dim, 256, bias=False)
        
            #networks.append(self.img_feature_transformer_enc)
            #networks.append(self.norm)
        
        #--------------------------------------------
        #stage 2
        vc_encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, batch_first=True, activation="gelu")
        self.VC_self_attention = nn.TransformerEncoder(vc_encoder_layer, num_layers=self.depth)
        self.VC_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.VC_cls_MLP = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.img_feat_emb = nn.Linear(256, embed_dim, bias=False)
        self.img_feature_transformer_enc = nn.TransformerEncoder(vc_encoder_layer, num_layers=self.depth)
        self.img_feat_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.img_feat_cls_MLP = nn.Linear(embed_dim, embed_dim, bias=False)
        #networks.append(self.multi_modal_transformer_decoders)
        
        self.vec_MLP = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.GroupNorm(self.group_size, self.token_num),
            nn.ReLU(inplace=False),
            nn.Linear(embed_dim, embed_dim, bias=False),
        )
        
        final_decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, batch_first=True, activation="gelu")
        self.multi_modal_transformer_decoders = nn.TransformerDecoder(final_decoder_layer, num_layers=self.decoder_depth)
        self.final_proj = nn.Linear(embed_dim, 256, bias = False)
        
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
        
        dummy = torch.zeros((64, 64, self.slow_time, 768)) # batch channel slowtime fasttime
        img_Feat_dummy = torch.zeros((64, 256, 16, 16))
        self.forward(dummy, img_Feat_dummy, init_check = True)
    
    
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
    
    def forward_backbone(self, x, init_check = False) :
        #backbone 
        if init_check :
            print("before embed :", x.shape)
            
        x = self.backbone(x)
        if init_check :
            print("after embed x type : {}".format(type(x)))
            print("x len : {}".format(len(x)))
        
        return x
    
    def foward_embedding(self, x, mask_ratio, init_check = False) :
        # embed patches
        # add pos embed w/o cls token
        device = x[0].device
        if init_check :
            print("use attentional_pooling : ", self.attentional_pooling)
        if self.scale_order == 'big_first' :
            x.reverse()
        
        if self.attentional_pooling :
            for i in range(len(x)) :
                x[i] = x[i] + self.pos_embeds[i]#[:, 1:, :]
            if init_check :
                print("pos emb finish. big shape : {}".format(x[0].shape))
            # masking: length -> length * mask_ratio
            full_x = x.copy()
            
            if self.mask_pos < len(x) :
                masked_x, mask, ids_restore = self.random_masking(x[self.mask_pos], mask_ratio)
                full_x[self.mask_pos] = masked_x
                if init_check :
                    print("random masking :", masked_x.shape, masked_x.requires_grad)
                    print()
        else :
            b, _, ch = x[0].shape
            x_ = torch.cat((x[0].reshape(b, -1, ch), x[1].reshape(b, -1, ch)), axis = 1)
            full_x = torch.cat((x_, x[2].reshape(b, -1, ch)), axis = 1)
            #error
            pos_emb = self.pos_embed(full_x).detach().to(device)
            full_x = full_x + pos_emb#[:, 1:, :]
            if init_check :
                print(full_x.shape)
            masked_x = None
            mask = None
            ids_restore = None
        
        
        return full_x
    
    def forward_self_attention(self, x, init_check = False):
        # Input : embedded x
        # apply Transformer blocks
        if init_check :
            print("local attention first input x :", x[0].shape)
        
        attentioned_x = []
        real_b, real_token_num, emb = x[0].shape
        #local attention
        for i in range(len(self.using_scale)) :
            tmp_x = x[3 - (1 + i)].clone()
            b, token_len, emb = tmp_x.shape
            #for local attention
            if self.attention_type[0] == 'local' :
                tmp_x = tmp_x.view(-1, self.local_window_size ,emb)
            
            if i == 0 : #
                tmp_x = self.transformer_encoders[i](tmp_x)
                if init_check :
                    print("{} attention {} finish x : {}".format(self.attention_type[0],i, tmp_x.shape))

            if self.attention_type[1] != 'local' :
                tmp_x = tmp_x.view(real_b, real_token_num, emb)
            
            attentioned_x.append(tmp_x)
            attentioned_x.reverse()

        return attentioned_x

    def forward_online_transformer_decoder(self, x, init_check=False) :
        memory = x[-1] #deepest feature - first memory
        b, token_len, emb = memory.shape
        #cross attention
        for i in range(len(self.using_scale)-1) :
            tmp_x = x[len(x)-(i+1+1)]
            
            memory = self.single_modal_transformer_decoders[i](tmp_x, memory)
            if init_check :
                print("cross attention stage {} memory shape {}".format(i, memory.shape))
        
        if self.attention_type[1] == 'local' :
            memory = memory.view(-1, int(self.rf_len/self.downsample*self.slow_time), emb)
            
        return memory

    def forward_vc_decoder(self, RF_vec, init_check=False) :
        device = RF_vec.device
        b, t, ch = RF_vec.shape
        #vc_query = torch.zeros((b, 13*17, ch)).to(device)
        vc_query = repeat(self.vc_query, '1 len d -> b len d', b = b)
        vc_query_pos = self.pos_embeds[-1]
        vc_query = vc_query + vc_query_pos
        x = self.vc_decoder(vc_query, RF_vec)
        vc_emb = x.clone()
        x = self.vc_proj(x)
        if init_check :
            print("VC shape ", x.shape)
        
        return x, vc_emb

    def forward_imgencoder(self, x, mask_ratio = 0.25, init_check = False):
        # Input :img feature x
        # apply Transformer blocks
        #print(x.shape)
        device = x.device
        x = rearrange(x, 'b ch h w -> b (h w) ch') ##tokenize
        x = self.img_feat_emb(x) # 256 -> embsize
        imgfeat = x.clone()
        b, seq, ch = x.shape
        
        #masking imgfeat
        masked_imgfeat, mask, ids_restore = self.random_masking(x, mask_ratio)
        mask_tokens = self.mask_token.repeat(masked_imgfeat.shape[0], ids_restore.shape[1] - masked_imgfeat.shape[1], 1)
        #print(mask_tokens.shape)
        x = torch.cat([x, mask_tokens], dim=1)  # no cls token
        masked_imgfeat = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        
        img_pos_emb = self.pos_embeds[-1]
        masked_imgfeat = masked_imgfeat + img_pos_emb.to(device)
        
        #cls imgfeat embedding
        cls_token = repeat(self.img_feat_cls_token, '1 1 d -> b 1 d', b = b)
        
        imgfeat = torch.cat((cls_token, imgfeat), 1)
        imgfeatpos_emb = self.pos_embed(imgfeat).detach()
        imgfeat = imgfeat + imgfeatpos_emb
        
        if init_check :
            print("img feat encoder input shape :", masked_imgfeat.shape)
    
        #x = x.reshape(B, C, -1)
        if init_check :
            print("img feat encoder reshape shape :", masked_imgfeat.shape)
        
        imgfeat = self.img_feature_transformer_enc(imgfeat)
        
        imgfeat_cls = self.img_feat_cls_MLP(imgfeat[:,:1])
        
        if init_check :
            print("img feat cls embedding :", imgfeat_cls.shape)
            
        return imgfeat_cls, masked_imgfeat, mask

    def forward_online_vc_encoder(self, x, init_check = False):
        b, seq, ch = x.shape
        device = x.device
        if init_check :
            ("vc_vec shape : {}".format(x.shape))
        
        cls_token = repeat(self.VC_cls_token, '1 1 d -> b 1 d', b = b)
            
        if init_check :
            print("cls token shape : {}".format(cls_token.shape))
            
        x = torch.cat((cls_token, x), 1)
        if init_check :
            print("cls + vc_vec shape : {}".format(x.shape))
            print("cls token grad", cls_token.requires_grad)
            
        dec_pos_enc = self.dec_pos_embed(x).detach()
        device = x.device
        x = x + dec_pos_enc.to(device)
        
        x = self.VC_self_attention(x)
        
        x_cls = self.VC_cls_MLP(x[:,:1])
        x_vec = x[:,1:]
        if init_check :
            print("vc : {}".format(x_vec.shape))
            print("vc cls {}".format(x_cls.shape))
        
        return x_cls, x_vec
    
    def forward_multimodal_decoder(self, img_feat_vec, VC_vec, init_check = False) :
        final_vec = self.multi_modal_transformer_decoders(img_feat_vec, VC_vec)
        final_vec = self.final_proj(final_vec)
        if init_check :
            print("final vec : {}".format(final_vec.shape))
        
        return final_vec
    

    def forward(self, x, img_feat, mask_ratio=0.25, init_check = False):
        x = self.forward_backbone(x, init_check)
        
        ####stage 1#### making VC
        full_x = self.foward_embedding(x, mask_ratio, init_check)
        
        if self.task == 'selfsupervised' :
            online_latent = self.forward_self_attention(full_x, init_check) #smallest feat
            online_RF_vec = self.forward_online_transformer_decoder(online_latent, init_check) ### RF vec
            VC, vc_emb = self.forward_vc_decoder(online_RF_vec, init_check)
            
            
            img_feat_cls, imgfeat, imgfeat_mask = self.forward_imgencoder(img_feat, mask_ratio, init_check)
            VC_cls, VC_embed_vec = self.forward_online_vc_encoder(vc_emb, init_check)
            
            
            final_latent = self.forward_multimodal_decoder(imgfeat, VC_embed_vec, init_check)
            online_outputs = {"online_vc": VC, 
                              "VC_latent_cls" : VC_cls,
                              "imgfeat_latent_cls" : img_feat_cls,
                              "final_latent" : final_latent
                              }
            target_outputs = {"vc_gt": img_feat,
                              "imgfeat_mask" : imgfeat_mask
                              }
        else :
            if self.attentional_pooling :
                online_latent = self.forward_self_attention(full_x, init_check)
                online_RF_vec = self.forward_online_transformer_decoder(online_latent, init_check) ### RF vec
            else :
                online_RF_vec = self.forward_self_attention(full_x, init_check)
            VC, vc_emb = self.forward_vc_decoder(online_RF_vec, init_check)
            online_outputs = {"online_vc": VC, 
                              "online_RF_latent" : online_RF_vec,
                              "online_latent_mse" : None,
                              "online_latent_cls" : None
                              }
            target_outputs = {"vc_gt": None, 
                              }
        
                    
        return online_outputs, target_outputs


class SetCriterion(nn.Module):
    def __init__(self, use_vc_loss = True, use_RF_vec_loss = False, final_loss_type = 'mse', device = None, vc_loss_coef = 1, 
                RF_vec_loss_coef = 1, final_loss_mse_coef = 1, final_loss_cls_coef = 1):
        super().__init__()
        
        self.temperature = nn.Parameter(torch.Tensor([1.]))
        self.coefs = {
            'vc' : torch.tensor(vc_loss_coef),
            #'RF_vec' : torch.tensor(RF_vec_loss_coef).to(self.device),
            'final_mse' : torch.tensor(final_loss_mse_coef),
            'final_cls' : torch.tensor(final_loss_cls_coef)
        }
        
        self.vc_loss = nn.MSELoss(reduction='mean').to(device)
            
            
        self.final_loss_mse = nn.MSELoss(reduction='mean').to(device)
        self.final_loss_cls = nn.CrossEntropyLoss #nn.CosineSimilarity(dim=1, eps=1e-6).to(device)
        
        #self.forward()
        
    def forward_vc_loss(self, online_vc, vc_gt) :
        vc_gt = rearrange(vc_gt, 'b ch h w -> b (h w) ch')
        loss = self.vc_loss(online_vc, vc_gt)
        
        return loss
    
    def forward_RF_vec_loss(self, online_RF_vec, RF_vec_gt) :
        loss = self.RF_vec_loss(online_RF_vec, RF_vec_gt)
        
        return loss
        
    def forward_final_loss_mse(self, online_latent, vc_gt, mask) :
        vc_gt = rearrange(vc_gt, 'b ch h w -> b (h w) ch')
        loss = (online_latent - vc_gt) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        #print(mask.sum())
        loss = (loss * mask).sum() / mask.sum()
        #loss = self.final_loss_mse(online_latent, vc_gt)
        
        return loss
    
    def forward_final_loss_cls(self, online_latent, target_latent) :
        batch_size = online_latent.shape[0]
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
        device = online_outputs["online_vc"].device
        vc_loss = self.forward_vc_loss(online_outputs["online_vc"], target_outputs["vc_gt"])
        
        final_loss_mse = self.forward_final_loss_mse(online_outputs["final_latent"], target_outputs["vc_gt"], target_outputs["imgfeat_mask"])
        final_loss_cls = self.forward_final_loss_cls(online_outputs["VC_latent_cls"], online_outputs['imgfeat_latent_cls'])
            
        losses = {
            'vc' : vc_loss,
            'final_mse' : final_loss_mse,
            'final_cls' : final_loss_cls
        }
        #print(losses['final_mse'])
        #print(losses)
        sum_loss = torch.zeros(1).to(device)
        
        for key , type_ck in list(losses.items()) :
            if type_ck == None :
                del losses[key]
        
        ##for log update##
        for i in losses :
            if losses[i] != None :
                #print(self.coefs[i].device)
                sum_loss += self.coefs[i].to(device)*losses[i]
        
        return losses, sum_loss ##losses for log
        
class pose_model_petr(nn.Module):
    def __init__(self, embed_dim = 256, num_heads = 8, people_queries = 15, multi_modal_depth = 2, pose_dec_depth = 2,device = None, use_vc = False, feature_gt = False, use_only_gt = False):
        super().__init__()
        
        self.people_queries = people_queries
        self.embed_dim = embed_dim
        self.pose_embed_dim = self.embed_dim#*3
        self.num_heads = num_heads
        self.device = device
        self.keypoint_num = 16
        self.group_size = 8
        self.use_vc = use_vc
        self.feature_gt = feature_gt
        self.use_only_gt = use_only_gt
        self.use_vc = False
        
        if embed_dim != 256 and self.feature_gt:
            self.gt_MLP = nn.Linear(256, 256, bias=False)
        vc_encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=num_heads, dim_feedforward=self.embed_dim*4,  batch_first=True, activation="gelu")
        self.vc_attention = nn.TransformerEncoder(vc_encoder_layer, num_layers=2)
        person_decoder_layer = nn.TransformerDecoderLayer(d_model=self.embed_dim, nhead=num_heads, dim_feedforward=self.embed_dim*4, batch_first=True, activation="gelu")
        if not self.use_vc :
            self.multimodal_decoder = nn.TransformerDecoder(person_decoder_layer, num_layers=multi_modal_depth)
        
        self.person_queries = nn.Parameter(torch.zeros(1, 15, embed_dim))
        
        self.pos_embeds = nn.ParameterList()#nn.ModuleList()
        
        #vc
        pos_embedding = nn.Parameter(torch.zeros(1, 256, embed_dim))
        self.pos_embeds.append(pos_embedding)
        
        #person
        pos_embedding = nn.Parameter(torch.zeros(1, people_queries, embed_dim))
        self.pos_embeds.append(pos_embedding)
        
        #pos
        pos_embedding = nn.Parameter(torch.zeros(1, 1 + self.keypoint_num, embed_dim)) # 1 for classification
        self.pos_embeds.append(pos_embedding)
        
        self.person_decoder = nn.TransformerDecoder(person_decoder_layer, num_layers=pose_dec_depth)
        
        if self.embed_dim != self.pose_embed_dim :
            self.PtoP_MLP = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.pose_queries = nn.Parameter(torch.zeros(1, self.keypoint_num+1, embed_dim))
        pose_decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=16, dim_feedforward=embed_dim*4, batch_first=True, activation="gelu")
        self.pose_decoder = nn.TransformerDecoder(pose_decoder_layer, num_layers=1)

        self.cls_MLP = nn.Sequential(
            nn.Linear(embed_dim, 1, bias=False),
            #nn.GroupNorm(5, self.people_queries),
            #nn.ReLU(inplace=False),
            #nn.Dropout(p=0.2),
            #nn.Linear((embed_dim)//2, 1, bias=False),
            nn.Sigmoid(),
        )
        self.pose_MLP = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2, bias=False),
            nn.GroupNorm(self.group_size, self.keypoint_num*self.people_queries),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.2),
            #nn.Linear((self.pose_embed_dim+2)//2, (self.pose_embed_dim+2)//2, bias=False),
            #nn.GroupNorm(self.group_size, self.keypoint_num*self.people_queries),
            #nn.ReLU(inplace=False),
            #nn.Dropout(p=0.2),
            nn.Linear(embed_dim//2, 2, bias=False),
            #nn.Linear(embed_dim, 2, bias=False),
            nn.Sigmoid(),
        )
        
        if self.feature_gt :
            vc = torch.zeros((64, 17*16, embed_dim), requires_grad=False)
            RF = torch.zeros((64, 256, embed_dim), requires_grad=False)
        else :
            vc = torch.zeros((64, 17*16, embed_dim), requires_grad=False)
            RF = torch.zeros((64, 256, embed_dim), requires_grad=False)
        #vc = torch.zeros((64, 17*16, self.embed_dim), requires_grad=False)
        self.forward(vc,RF , init_check=True)
        
    def forward(self, visual_clue, RF_latent, init_check=False) :
        device = visual_clue.device
        vc = visual_clue
        
        if self.feature_gt :
            if self.embed_dim != 256 :
                vc = self.gt_MLP(vc)
            else :
                vc = visual_clue
            #vc_pos = self.pos(vc).detach().to(device)
            #vc = vc + vc_pos
            vc = self.vc_attention(vc)
        else :
            if self.use_vc :
                vc_pos = self.pos_embeds[0]
                vc = vc + vc_pos
                vc = self.vc_attention(vc)
                memory = vc
            else :
                memory = RF_latent
        
        b, t, ch = memory.shape

        person_queries = repeat(self.person_queries, '1 person_num d -> b person_num d', b = b)
        pos_encoding = self.pos_embeds[1]
        if init_check  == False :
            pos_encoding = pos_encoding.to(device)
            person_queries = person_queries.to(device)
        
        person_queries = person_queries + pos_encoding
        x = self.person_decoder(person_queries, memory)
        
        if self.embed_dim != self.pose_embed_dim :
            x = self.PtoP_MLP(x)
        if init_check :
            print("person embed", x.shape)
            #print("first person", x[:, 0].shape)
        
        batch, person_queries, embed = x.shape
        for person_num in range(person_queries) :
            pose_queries = self.pose_queries
            pose_queries = repeat(self.pose_queries, '1 pose_len d -> b pose_len d', b = b)
            pose_pos_encoding = self.pos_embeds[2]
            pose_queries = pose_queries + pose_pos_encoding
            temp_person = self.pose_decoder(pose_queries, x[:,person_num,:].unsqueeze(1)).unsqueeze(1)
            if person_num == 0 :
                pose_embeds = temp_person
            else :
                pose_embeds = torch.cat((pose_embeds, temp_person), 1)
        x = pose_embeds #batch, persons, score+keypoints, dim
        if init_check :
            print("cls embed", x[:,:,0,:].shape)
            print("pos embed", x[:,:,1:,:].shape)
        cls = self.cls_MLP(x[:,:,0,:])
        pose = self.pose_MLP(x[:,:,1:,:].reshape(b, self.people_queries*self.keypoint_num, embed))
        pose = pose.view(b, self.people_queries, self.keypoint_num*2)
        if init_check :
            print("cls shape : ", cls.shape)
            print("pos shape : ", pose.shape)
            print("grad : ", cls.requires_grad)
        
        out = {'pred_logits': cls, 'pred_keypoint': pose, 'vc' : visual_clue}
            
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
                    '''
                    batch_output_coor_wrist = output['pred_keypoint'][i][tmp_rows[j]][2*10:2*11]
                    batch_gt_coor_wrist = gt[i]["cd"][tmp_cols[j]][2*10:2*11]
                    
                    batch_output_coor_wrist = torch.cat((batch_output_coor_wrist, output['pred_keypoint'][i][tmp_rows[j]][2*15:2*16]))
                    batch_gt_coor_wrist = torch.cat((batch_gt_coor_wrist, gt[i]["cd"][tmp_cols[j]][2*15:2*16]))
                    '''
                else :
                    batch_output_coor = torch.cat((batch_output_coor, output['pred_keypoint'][i][tmp_rows[j]]))
                    batch_gt_coor = torch.cat((batch_gt_coor, gt[i]["cd"][tmp_cols[j]]))
                    '''
                    batch_output_coor_wrist = torch.cat((batch_output_coor_wrist, output['pred_keypoint'][i][tmp_rows[j]][2*10:2*11]))
                    batch_gt_coor_wrist = torch.cat((batch_gt_coor_wrist, gt[i]["cd"][tmp_cols[j]][2*10:2*11]))
                    
                    batch_output_coor_wrist = torch.cat((batch_output_coor_wrist, output['pred_keypoint'][i][tmp_rows[j]][2*15:2*16]))
                    batch_gt_coor_wrist = torch.cat((batch_gt_coor_wrist, gt[i]["cd"][tmp_cols[j]][2*15:2*16]))
                    '''
            
            if i == 0 :# for print
                first_gt_class.append(tmp_gt_class)
            
            if class_loss == None :
                class_loss = self.class_loss(output["pred_logits"][i], tmp_gt_class)
            else :
                class_loss = class_loss + self.class_loss(output["pred_logits"][i], tmp_gt_class)
            
            if keypoint_loss == None :
                keypoint_loss = self.keypoint_loss(batch_output_coor, batch_gt_coor)
                #wrist weight +
                #keypoint_loss = keypoint_loss + 15*self.keypoint_loss(batch_output_coor_wrist, batch_gt_coor_wrist)
            else :
                keypoint_loss = keypoint_loss + self.keypoint_loss(batch_output_coor, batch_gt_coor)
                #wrist weight +
                #keypoint_loss = keypoint_loss + 15*self.keypoint_loss(batch_output_coor_wrist, batch_gt_coor_wrist)
                
            non_less_bs += 1
            
            if self.vc_learn :
                tokens, emb = output['vc'][i].shape
                #print(output['vc'][i].shape)
                #print(gt[i]["features"].shape)
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
            'cls' : class_loss/non_less_bs,
            'cd' : keypoint_loss/non_less_bs,
            'vc' : vc_loss/non_less_bs
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
        embed_dim=args.hidden_dim, depth=args.enc_layers, num_heads=args.nheads,
        decoder_embed_dim=args.hidden_dim, decoder_depth=args.dec_layers, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), backbone= convbackbone, use_img_feat_enc = args.use_img_feat_enc, device = device, 
        task = args.task, scale_level = args.multi_scale, attentional_pooling = args.attentional_pooling, scale_order = args.scale_order, slow_time = args.slow_time)
    
    criterion = SetCriterion(use_vc_loss = True, use_RF_vec_loss = False, final_loss_type = 'mse', device = device, vc_loss_coef = 1, 
                RF_vec_loss_coef = 1, final_loss_cls_coef = 1)
    
    
    #model = nn.DataParallel(model, device_ids=[0, 1, 2])
    #criterion = nn.DataParallel(criterion, device_ids=[0, 1, 2])
    
    criterion.to(device)
    model.to(device)
    
    if args.task != 'selfsupervised' :
        pose_model = pose_model_petr(embed_dim = args.hidden_dim, multi_modal_depth = args.cross_dec_layers, device = device, use_vc = args.use_vc, feature_gt = args.feature_gt, use_only_gt = args.use_only_gt)
        #pose_model.to(device)
        #pose_model = nn.DataParallel(pose_model, device_ids=[0, 1, 2])
        pose_model.to(device)
        if args.task == 'scratch' and args.use_vc:
            criterion = Pose_Criterion(vc_coef = args.feature_loss_coef, keypoint_coef = args.keypoint_loss_coef, class_coef = args.cls_loss_coef, matcher_keypoint_coef = args.set_cost_keypoint, matcher_class_coef = args.set_cost_class, vc_learn= True)
            if args.feature_gt :
                criterion = Pose_Criterion(vc_coef = 0, keypoint_coef = args.keypoint_loss_coef, class_coef = args.cls_loss_coef, matcher_keypoint_coef = args.set_cost_keypoint, matcher_class_coef = args.set_cost_class, vc_learn= True)
        else :
            criterion = Pose_Criterion(vc_coef = args.feature_loss_coef, keypoint_coef = args.keypoint_loss_coef, class_coef = args.cls_loss_coef, matcher_keypoint_coef = args.set_cost_keypoint, matcher_class_coef = args.set_cost_class, vc_learn= True)
        #criterion = nn.DataParallel(criterion, device_ids=[0, 1, 2])
        
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