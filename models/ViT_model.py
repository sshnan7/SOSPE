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
from positional_encodings.torch_encodings import PositionalEncoding1D
from timm.models.vision_transformer import PatchEmbed, Block
from convnext import build_convbackbone, LayerNorm

#from util.pos_embed import get_1d_sincos_pos_embed


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, backbone = None):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        
        #num_patches = self.patch_embed.num_patches
        
        #################not using in C-Mae#####################
        #self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        #self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        
        #self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)
        self.backbone = backbone
        self.token_num = 768
        self.masked_token_num = 576
        self.img_token_num = 100
        
        self.pos_embed = PositionalEncoding1D(self.token_num)
        self.img_pos_embed = PositionalEncoding2D(self.img_token_num)
        #self.pos_embed.requires_grad = False
        
        self.online_encoder = nn.ModuleList([
            #Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        
        self.SimSiamMLP = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.BatchNorm1d(self.masked_token_num),
            nn.ReLU(inplace=True),
            #nn.Linear(embed_dim, embed_dim, bias=False),
            #nn.BatchNorm1d(embed_dim),
            #nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.BatchNorm1d(self.masked_token_num, affine=False)
        )
        
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        #self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_pos_embed = PositionalEncoding1D(self.token_num)
        self.decoder_pos_embed.requires_grad = False

        self.decoder_blocks = nn.ModuleList([
            #Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        #self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        self.decoder_pred = nn.Linear(decoder_embed_dim, embed_dim, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()
        
        dummy = torch.zeros((48, 1, 64, 768))
        self.check_dim(dummy)

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
    
    def foward_embedding(self, x, mask_ratio, init_check = False) :
        # embed patches
        if init_check :
            print("before embed :", x.shape)
            
        x = self.backbone(x)
        if init_check :
            print("after embed :", x.shape)

        # add pos embed w/o cls token
        pos_emb = self.pos_embed(x).detach()
        x = x + pos_emb#[:, 1:, :]
        if init_check :
            print("after pose embed :", x.shape)
        
        # masking: length -> length * mask_ratio
        full_x = x.clone()
        masked_x, mask, ids_restore = self.random_masking(x, mask_ratio)
        if init_check :
            print("random masking :", x.shape)
            print()
        
        return full_x, masked_x, mask, ids_restore
    
    def forward_online_encoder(self, x, mask, init_check = False):
        # Input : embedded x
        # apply Transformer blocks
        if init_check :
            print("online encoder x :", x.shape)
            
        for blk in self.online_encoder:
            x = blk(x)
        x = self.norm(x)
        if init_check :
            print("online transformer encoder finish x :", x.shape)
            
        x = self.SimSiamMLP(x)
        
        if init_check :
            print("online encoder simsiam finish x :", x.shape)
            print()

        return x

    def forward_target_encoder(self, x, init_check = False):
        # Input : embedded x
        # apply Transformer blocks
        if init_check :
            print("target encoder x :", x.shape)
        for blk in self.online_encoder:
            x = blk(x)
        x = self.norm(x)
        if init_check :
            print("target encoder finish x :", x.shape)
        
        x = x.detach()
        
        if init_check :
            print("target encoder grad :", x.requires_grad)
            print()

        return x

    def forward_target_imgencoder(self, x, init_check = False):
        # Input :img feature x
        # apply Transformer blocks
        img_pos_emb = self.img_pos_embed(x).detach()
        x = x + pos_emb#[:, 1:, :]
        
        if init_check :
            print("target img encoder x :", x.shape)
        
        B, C, h, w = x.shape
        
        x = x.reshape(B, C, -1)
        if init_check :
            print("target img encoder reshape x :", x.shape)
        
        for blk in self.online_encoder:
            x = blk(x)
        x = self.norm(x)
        if init_check :
            print("target img encoder finish x :", x.shape)
        
        x = x.detach()
        
        if init_check :
            print("target encoder grad :", x.requires_grad)
            print()

        return x

    def forward_decoder(self, x, ids_restore, init_check = False):
        # embed tokens
        if init_check :
            print("decoder stage :", x.shape)
        x = self.decoder_embed(x)
        if init_check :
            print("decoder embed :", x.shape)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        if init_check :
            print("decoder full :", x.shape)

        # add pos embed
        decoder_pos_embed = self.decoder_pos_embed(x).detach()
        x = x + decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)
        if init_check :
            print("decoder pred :", x.shape)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask, init_check = False):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, x, mask_ratio=0.25):
        full_x, masked_x, mask, ids_restore = self.foward_embedding(x, mask_ratio)
        online_latent = self.forward_online_encoder(masked_x, mask)
        target_latent = self.forward_target_encoder(full_x)
        pred = self.forward_decoder(online_latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask
        
    def check_dim(self, x, mask_ratio=0.25) :
        full_x, masked_x, mask, ids_restore = self.foward_embedding(x, mask_ratio, init_check = True)
        online_latent = self.forward_online_encoder(masked_x, mask, init_check = True)
        target_latent = self.forward_target_encoder(full_x, init_check = True)
        pred = self.forward_decoder(online_latent, ids_restore, init_check = True)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask, init_check = True)
        return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
    
    
def build_model(args):   
    device = torch.device(args.device)

    convbackbone = build_convbackbone(args)

    model = MaskedAutoencoderViT(
        embed_dim=256, depth=12, num_heads=8,
        decoder_embed_dim=256, decoder_depth=8, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), backbone= convbackbone)
    
    return model

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