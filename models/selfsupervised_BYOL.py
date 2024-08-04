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
    def __init__(self, embed_dim=1024, depth=6, num_heads=16, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, shuffle = False):
        super().__init__()
        

        self.depth = depth #// (self.decoder_stages + 1)
        self.decoder_depth = decoder_depth
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        #pos enc 1d fixed
        self.pos_enc_1d = PositionalEncoding1D(embed_dim)
        
        #student model
        self.transformer_encoder = nn.ModuleList([])
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, batch_first=True, activation="gelu")
        for i in range(depth) :
            single_transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
            self.transformer_encoder.append(single_transformer_encoder)
        '''   
        self.simsiam_MLP = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4, bias=False),
            nn.LayerNorm(embed_dim*4, eps=1e-06),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim*4, embed_dim*4, bias=False),
            nn.LayerNorm(embed_dim*4, eps=1e-06),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim*4, embed_dim, bias=False)
        )
        '''
        self.simsiam_MLP = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.LayerNorm(embed_dim, eps=1e-06),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim, bias=False)
        )
        
        self.shuffle = shuffle
        
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
    
    def imgfeat_patchify(self, x) :
        b, slowt, seq, ch = x.shape
        x = x.view(b, seq, ch)
        
        return x
    
    def forward_transformer(self, x) :
        for block in self.transformer_encoder :
            x = block(x)
        return x

    def forward(self, rf_latent, img_feat, mask_ratio=0, init_check = False):
        img_feat = self.imgfeat_patchify(img_feat)
        
        student_pred1 = self.forward_transformer(img_feat) #x1
        with torch.no_grad():
            teacher_pred1 = self.forward_transformer(rf_latent) #x2
        
        student_pred2 = None
        teacher_pred2 = None
        if self.shuffle :
            student_pred2 = self.forward_transformer(rf_latent)
            with torch.no_grad():
                teacher_pred2 = self.forward_transformer(img_feat)
        
        student_pred1 = self.simsiam_MLP(student_pred1)
        if student_pred2 != None :
            student_pred2 = self.simsiam_MLP(student_pred2)
        
        outputs = {
                          "imgfeat_pred" : student_pred1,
                          "imgfeat_latent" : teacher_pred1,
                          "student_pred1" : student_pred1,
                          "student_pred2" : student_pred2,
                          "teacher_pred1" : teacher_pred1,
                          "teacher_pred2" : teacher_pred2,
                          }
        
        return outputs
        

class selfsupervised_criterion(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward_loss(self, x, y) :
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)
    
    def forward(self, outputs) :
        device = outputs["student_pred1"].device
        
        loss1 = self.forward_loss(outputs["student_pred1"], outputs["teacher_pred1"].detach())
        loss2 = None
        if outputs["student_pred2"] != None :
            loss2 = self.forward_loss(outputs["student_pred2"], outputs["teacher_pred2"].detach())
        
        losses = {
            'BYOL1' : loss1,
            'BYOL2' : loss2
        }
        
        for key , val in list(losses.items()) :
            if val == None :
                del losses[key]
            else :
                losses[key] = val.mean()
        
        ##sum loss##
        sum_loss = None
        for i in losses :
            if sum_loss == None :
                sum_loss = losses[i]
            else :
                sum_loss += losses[i]
        
        return losses, sum_loss ##losses for log
        
def build_BYOLModel(args):
    selfmodel = SelfsupervisedModel( embed_dim=args.hidden_dim, depth=6, num_heads=args.nheads,
        decoder_embed_dim=args.hidden_dim, decoder_depth=args.dec_layers, decoder_num_heads=8, shuffle= False)
    
    return selfmodel
        
def build_BYOLCriterion() :
    selfcriterion = selfsupervised_criterion()
    
    return selfcriterion