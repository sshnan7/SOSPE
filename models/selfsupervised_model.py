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


class SelfsupervisedModel(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, embed_dim=1024, depth=24, num_heads=16, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16):
        super().__init__()
        

        self.depth = depth #// (self.decoder_stages + 1)
        self.decoder_depth = decoder_depth
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        #pos enc 1d fixed
        self.pos_enc_1d = PositionalEncoding1D(embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.img_feat_emb = nn.Linear(256, embed_dim, bias=False)
        imgfeat_encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, batch_first=True, activation="gelu")
        self.img_feature_transformer_enc = nn.TransformerEncoder(imgfeat_encoder_layer, num_layers=self.depth)
        self.img_feat_cls_MLP = nn.Linear(embed_dim, embed_dim, bias=False)
        imgfeat_dec_layer= nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, batch_first=True, activation="gelu")
        self.img_feature_transformer_dec = nn.TransformerDecoder(imgfeat_dec_layer, num_layers=self.depth)
        self.img_feat_dec_MLP = nn.Linear(embed_dim, 256, bias=False)
        
        rf_embed_dummy = torch.zeros((64, 256, 256)) # batch channel slowtime fasttime
        img_Feat_dummy = torch.zeros((64, 1, 256, 256))
        self.forward(rf_embed_dummy, img_Feat_dummy, init_check = True)
    
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
        
    def forward_imgencoder(self, x, mask_ratio, init_check = False):
        # Input :img feature x
        device = x.device
        imgfeat = x.clone()
        b, slowt, seq, ch = x.shape
        x = x.view(b, seq, ch)
        
        #masking imgfeat
        cls_token = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_token, x), 1)
        img_pos_emb = self.pos_enc_1d(x)
        x = x + img_pos_emb.to(device)
        
        cls_token = x[:, :1]
        masked_imgfeat, mask, ids_restore = self.random_masking(x[:, 1:], mask_ratio)
        masked_imgfeat = torch.cat((cls_token, masked_imgfeat), 1)
        
        if init_check :
            print("img feat encoder input shape :", masked_imgfeat.shape)
        
        x_attn = self.img_feature_transformer_enc(masked_imgfeat)
        
        imgfeat_cls = self.img_feat_cls_MLP(x_attn[:,:1])
        imgfeat_emb = x_attn[:,1:]
        
        #only use if forward imgdecoder not use
        imgfeat_emb = self.img_feat_dec_MLP(imgfeat_emb)
        
        if init_check :
            print("img feat attn :", imgfeat_emb.shape)
            
        return imgfeat_cls, imgfeat_emb, mask, ids_restore
        
    def forward_imgdecoder(self, masked_imgfeat, RF_latent, ids_restore, init_check = False):
        # Input :img feature x
        # apply Transformer blocks
        device = masked_imgfeat.device
        #x = self.img_feat_emb(x) # 256 -> embsize
        b, seq, ch = masked_imgfeat.shape
        
        #masked_imgfeat, mask, ids_restore = self.random_masking(masked_imgfeat, mask_ratio)
        
        mask_tokens = self.mask_token.repeat(masked_imgfeat.shape[0], ids_restore.shape[1] - masked_imgfeat.shape[1], 1)
        masked_img_feat_attn = torch.cat([masked_imgfeat, mask_tokens], dim=1)  # no cls token
        masked_img_feat_attn = torch.gather(masked_img_feat_attn, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, masked_imgfeat.shape[2]))  # unshuffle
        
        img_pos_emb = self.pos_enc_1d(masked_img_feat_attn)
        masked_img_feat_attn = masked_img_feat_attn + img_pos_emb.to(device)
        img_feat_dec_output = self.img_feature_transformer_dec(masked_img_feat_attn, RF_latent)
        img_feat_pred = self.img_feat_dec_MLP(img_feat_dec_output)
        
        if init_check :
            print("img feat dec output shape :", img_feat_dec_output.shape)
 
            
        return img_feat_pred, img_feat_dec_output

    def forward(self, rf_latent, img_feat, mask_ratio=0, init_check = False):
        img_feat_cls, imgfeat_emb, mask, ids_restore = self.forward_imgencoder(img_feat, mask_ratio, init_check)
        mask = None
        online_outputs = {
                          "imgfeat_cls" : img_feat_cls,
                          "imgfeat_pred" : imgfeat_emb,
                          "imgfeat_latent" : imgfeat_emb,
                          "mask" : mask
                          }
        '''
        #img_feat_pred, img_feat_latent = self.forward_imgdecoder(imgfeat_emb, rf_latent, ids_restore, init_check)
        online_outputs = {
                          "imgfeat_cls" : img_feat_cls,
                          "imgfeat_pred" : img_feat_pred,
                          "imgfeat_latent" : img_feat_latent,
                          "mask" : mask
                          }
        '''
        
                    
        return online_outputs
        

class selfsupervised_criterion(nn.Module):
    def __init__(self, imgfeat_loss_coef = 1, cls_loss_coef = 0.1):
        super().__init__()
        
        self.temperature = nn.Parameter(torch.Tensor([1.]))
        self.coefs = {
            'imgfeat' : torch.tensor(imgfeat_loss_coef),
            'cls' : torch.tensor(cls_loss_coef)
        }
        
        #self.imgfeat_loss = nn.MSELoss(reduction='mean').to(device)
        self.final_loss_cls = nn.CrossEntropyLoss #nn.CosineSimilarity(dim=1, eps=1e-6).to(device)
        
    def forward_imgfeat_loss(self, imgfeat_pred, imgfeat, mask) :
        #loss = self.imgfeat_loss(imgfeat_pred, imgfeat)
        if len(imgfeat.shape) == 4 :
            b, st, seq, ch = imgfeat.shape
        if len(imgfeat.shape) == 3 :
            b, seq, ch = imgfeat.shape
        imgfeat = imgfeat.view(b, seq, ch)
        loss = (imgfeat_pred - imgfeat) ** 2
        
        if mask != None :
            loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
            loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        
        else :
            loss = loss.mean()  # [N, L], mean loss per patch
        
        return loss
    
    def forward_cls_loss(self, online_latent, target_latent) :
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
    
    def forward(self, online_outputs, selfsupoutputs, target) :
        device = online_outputs["imgfeat_pred"].device
        imgfeat_loss = None
        imgfeat_loss = self.forward_imgfeat_loss(selfsupoutputs["imgfeat_pred"], online_outputs["online_vc"], selfsupoutputs["mask"])
        #imgfeat_loss = self.forward_imgfeat_loss(selfsupoutputs["imgfeat_pred"], target, selfsupoutputs["mask"])
        
        cls_loss = self.forward_cls_loss(selfsupoutputs["imgfeat_cls"], online_outputs['online_RF_cls'])
            
        losses = {
            'imgfeat' : imgfeat_loss,
            'cls' : cls_loss
        }
        #print(losses['final_mse'])
        #print(losses)
        sum_loss = None
        
        for key , type_ck in list(losses.items()) :
            if type_ck == None :
                del losses[key]
        
        ##for log update##
        for i in losses :
            if losses[i] != None :
                if sum_loss == None :
                    sum_loss = self.coefs[i].to(device)*losses[i]
                else :
                    sum_loss += self.coefs[i].to(device)*losses[i]
        
        return losses, sum_loss ##losses for log
        
def build_selfsupervisedModel(args):
    selfmodel = SelfsupervisedModel( embed_dim=args.hidden_dim, depth=args.enc_layers, num_heads=args.nheads,
        decoder_embed_dim=args.hidden_dim, decoder_depth=args.dec_layers, decoder_num_heads=8)
    
    return selfmodel
        
def build_selfsupervisedCriterion(imgfeat_loss_coef = 1, cls_loss_coef = 0.1) :
    selfcriterion = selfsupervised_criterion(imgfeat_loss_coef, cls_loss_coef)
    
    return selfcriterion