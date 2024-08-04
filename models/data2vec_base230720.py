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
from .Light_conv import build_lightbackbone
from .Light_vit import build_lightvitbackbone
from einops.einops import rearrange, repeat
from einops.layers.torch import Rearrange
from scipy.optimize import linear_sum_assignment
from torch import einsum, nn
import torch.nn.functional as F
from .selfsupervised_model import build_selfsupervisedModel, build_selfsupervisedCriterion
from .selfsupervised_BYOL import build_BYOLModel, build_BYOLCriterion
from .selfsupervised_data2vec import build_data2vec_Model, build_data2vec_Criterion

#from util.pos_embed import get_1d_sincos_pos_embed


class Data2Vec_base(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, use_feature = 'HR',
                 backbone = None):
        super().__init__()

        # --------------------------------------------------------------------------
        
        self.backbone = backbone
        self.depth = depth #// (self.decoder_stages + 1)
        self.decoder_depth = decoder_depth
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        #pos enc 1d fixed
        self.pos_enc_1d = PositionalEncoding1D(embed_dim)
        
        #-----------------------------------------------------------------
        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-06)
        
        #stage 1
        self.transformer_encoder = nn.ModuleList([])
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, batch_first=True, activation="gelu")
        for i in range(depth) :
            single_transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
            self.transformer_encoder.append(single_transformer_encoder)
        
        self.layernorm2 = nn.LayerNorm([256, depth, 256], eps=1e-06)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_feature == 'HR' :
            vc_size  =  16*16 
        elif use_feature == 'PETR' :
            vc_size  =  13*17
        self.vc_query = nn.Parameter(torch.zeros(1, vc_size, embed_dim))
        self.VC_query_pos_embed = PositionalEncoding1D(embed_dim)
        vc_decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, batch_first=True, activation="gelu")
        self.vc_decoder = nn.TransformerDecoder(vc_decoder_layer, num_layers=self.decoder_depth)
        self.vc_cls_proj = nn.Linear(embed_dim, 256, bias=False)
        self.vc_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.GELU(),
            nn.Linear(embed_dim, 256, bias=False)
            )
        
            #networks.append(self.img_feature_transformer_enc)
            #networks.append(self.norm)
        
        #--------------------------------------------
        
        #networks.append(self.SimSiam_vec_MLP)
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
        
        dummy = torch.zeros((64, 64, 768)) # batch channel slowtime fasttime
        img_Feat_dummy = torch.zeros((64, 1, vc_size, 256))
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
    
    def forward_backbone(self, x, init_check = False) :
        #backbone 
        if init_check :
            print("before embed :", x.shape)
            
        x = self.backbone(x)
        if init_check :
            print("after embed x type : {}".format(type(x)))
            print("x shape : {}".format(x.shape))
        
        return x
    
    def foward_embedding(self, x, init_check = False) :
        # patches norm
        x = self.layernorm1(x)
        # add pos embed w/o cls token
        device = x.device
        pos_enc = self.pos_enc_1d(x)
        x = x + pos_enc #[:, 1:, :]
        if init_check :
            print("pos emb finish. {}".format(x.shape))
            # masking: length -> length * mask_ratio
        full_x = x.clone()
        return full_x
    
    def forward_self_attention(self, x, init_check = False):
        # Input : embedded x
        # apply Transformer blocks
        b, token_num, emb = x.shape
        x = x.view(b, -1, emb) #patchfy
        for i in range(len(self.transformer_encoder)) :
            x = self.transformer_encoder[i](x)
            
            if i== 0 :
                outputs = x.clone().unsqueeze(3)
            else :
                outputs = torch.cat([outputs, x.clone().unsqueeze(2)], dim = 3)
        
        if init_check :
            print("attention finish x : {}".format(outputs.shape))
                
        x = torch.mean(outputs, dim = 3)
        
        if init_check :
            print("attention mean finish x : {}".format(x.shape))

        return x

    def forward_vc_decoder(self, RF_vec, init_check=False) :
        device = RF_vec.device
        b, t, ch = RF_vec.shape
        vcs = []
        vc_embs = []
        vc_clses = []
        #vc_query = torch.zeros((b, 13*17, ch)).to(device)
        vc_query = repeat(self.vc_query, '1 len d -> b len d', b = b)
        cls_token = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        vc_query = torch.cat((cls_token, vc_query), 1)
        
        vc_query_pos_enc = self.pos_enc_1d(vc_query)
        vc_query = vc_query + vc_query_pos_enc
        x = self.vc_decoder(vc_query, RF_vec)
        
        vc_emb = x[:, 1:].clone()
        vc_cls = self.vc_cls_proj(x[:, :1])
        vc_pred = self.vc_proj(x[:, 1:])
        
        if init_check :
            print("VC shape ", vc_pred.shape)
        
        return vc_pred, vc_emb, vc_cls
    
    

    def forward(self, x, img_feat, mask_ratio=0.5, init_check = False):
        x = self.forward_backbone(x, init_check)
        
        ############################ making VC ####################################
        full_x = self.foward_embedding(x, init_check)
        online_RF_vec = self.forward_self_attention(full_x, init_check)
        VC = None
        vc_clses = None
        VC, online_RF_vec, vc_clses = self.forward_vc_decoder(online_RF_vec, init_check)
        
        online_outputs = {"online_vc": VC, 
                          "online_RF_cls" : vc_clses,
                          "online_RF_latent" : online_RF_vec,
                          "imgfeat_pred" : None,
                          "imgfeat_latent" : None
                          }
        
        return online_outputs

        
class pose_model_petr(nn.Module):
    def __init__(self, embed_dim = 256, num_heads = 8, people_queries = 15, pose_dec_depth = 2, using_latent = 0, feature_gt = False):
        super().__init__()
        
        self.people_queries = people_queries
        self.embed_dim = embed_dim
        self.pose_embed_dim = self.embed_dim#*3
        self.num_heads = num_heads
        self.keypoint_num = 16
        self.group_size = 8
        self.using_latent = using_latent
        self.feature_gt = feature_gt
        
        
        
        #self.gt_MLP = nn.Linear(256, 256, bias=False)
        
        #pos encs
        self.pos_enc = PositionalEncoding1D(embed_dim)
        
        
        #vc self attention
        vc_encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=num_heads, dim_feedforward=self.embed_dim*4,  batch_first=True, activation="gelu")
        self.vc_attention = nn.TransformerEncoder(vc_encoder_layer, num_layers=2)
        
        
        #person stage
        #person queries
        self.person_queries = nn.Parameter(torch.zeros(1, people_queries, embed_dim))
        #person decoder
        person_decoder_layer = nn.TransformerDecoderLayer(d_model=self.embed_dim, nhead=num_heads, dim_feedforward=self.embed_dim*4, batch_first=True, activation="gelu")
        self.person_decoder = nn.TransformerDecoder(person_decoder_layer, num_layers=pose_dec_depth)
        
        
        #pose stage
        if self.embed_dim != self.pose_embed_dim :
            self.PtoP_MLP = nn.Linear(embed_dim, embed_dim, bias=False)
        #pose queries(cls + poses)
        self.pose_queries = nn.Parameter(torch.zeros(1, self.keypoint_num+1, embed_dim))
        #pose decoder
        pose_decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=16, dim_feedforward=embed_dim*4, batch_first=True, activation="gelu")
        self.pose_decoder = nn.TransformerDecoder(pose_decoder_layer, num_layers=1)
        #pose cls mlp
        self.cls_MLP = nn.Sequential(
            nn.Linear(embed_dim, 1, bias=False),
            nn.Sigmoid(),
        )
        #pose mlp
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
            nn.Sigmoid(),
        )
        
        if self.feature_gt :
            vc = torch.zeros((64, 17*16, embed_dim), requires_grad=False)
            RF = torch.zeros((64, 256, embed_dim), requires_grad=False)
        else :
            vc = torch.zeros((64, 17*16, embed_dim), requires_grad=False)
            RF = torch.zeros((64, 256, embed_dim), requires_grad=False)
        #vc = torch.zeros((64, 17*16, self.embed_dim), requires_grad=False)
        self.forward(vc, RF, vc, vc, init_check=True)
        
    def forward(self, visual_clue, RF_latent, imgfeat_pred, imgfeat_latent, former_person_queries = None ,init_check=False) :
        device = RF_latent.device
        vc = visual_clue
        
        if self.feature_gt :
            if self.embed_dim != 256 :
                memory = self.gt_MLP(vc)
            else :
                memory = visual_clue
            memory = self.vc_attention(memory)
        else :
            if self.using_latent == 0 : #use vc
                #vc_pos = self.pos_enc(vc)
                #vc = vc + vc_pos
                #vc = self.vc_attention(vc)
                memory = vc
            if self.using_latent == 1 : #use latent
                memory = RF_latent
            if self.using_latent == 2 : #use imgfeat_pred(only selfsupervised)
                memory = imgfeat_pred
            if self.using_latent == 3 : #use imgfeat_latent(only selfsupervised)
                memory = imgfeat_latent
                              
        memory_pos_enc = self.pos_enc(memory)
        memory = memory + memory_pos_enc.detach()
        #memory = self.vc_attention(memory)
        
        b, t, ch = memory.shape
        
        if former_person_queries == None :
            person_queries = repeat(self.person_queries, '1 person_num d -> b person_num d', b = b)
        else :
            person_queries = former_person_queries # batch, token_len, dim
        person_pos_enc = self.pos_enc(person_queries) #use fixed pos
        if init_check  == False :
            person_pos_enc = person_pos_enc.to(device)
            person_queries = person_queries.to(device)
        
        person_queries = person_queries + person_pos_enc
        x = self.person_decoder(person_queries, memory)
        embed_queries = x.clone()
        
        if self.embed_dim != self.pose_embed_dim :
            x = self.PtoP_MLP(x)
        if init_check :
            print("person embed", x.shape)
            print("pos encoding learn", person_pos_enc.requires_grad)
            #print("first person", x[:, 0].shape)
        
        batch, person_queries, embed = x.shape
        pose_queries = repeat(self.pose_queries, '1 pose_len d -> b pose_len d', b = b)
        pose_pos_enc = self.pos_enc(pose_queries)
        pose_queries = pose_queries + pose_pos_enc
        for person_num in range(person_queries) :
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
        
        out = {'pred_logits': cls, 'pred_keypoint': pose, 'vc' : visual_clue, 'embed_queries' : embed_queries}
            
        return out

class Pose_Criterion(nn.Module):
    def __init__(self, use_box_loss = False,  device = None, vc_coef = 0, keypoint_coef = 1, class_coef = 1, box_mse_coef = 1, matcher_keypoint_coef = 1, matcher_class_coef = 1, vc_learn = False, people_queries = 15):
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
                #print(output["pred_logits"][i].requires_grad)
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
            
            if self.vc_learn and output['vc'] != None:
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
    
    if args.backbone_type == 'convnext' :
        convbackbone = build_convbackbone(args)
    elif args.backbone_type == 'light_conv' :
        convbackbone = build_lightbackbone(args)
    elif args.backbone_type == 'light_vit' :
        convbackbone = build_lightvitbackbone(args)

    model = Data2Vec_base( embed_dim=args.hidden_dim, depth=args.enc_layers, num_heads=args.nheads,
        decoder_embed_dim=args.hidden_dim, decoder_depth=args.dec_layers, decoder_num_heads=8, use_feature = args.use_feature, backbone= convbackbone)
        
    model.to(device)
    
    selfmodel = None
    selfcriterion = None
    pose_model = None
    posecriterion = None
    
    if args.selfsupervised != None :
        print("self supervised training ver {}".format(args.selfsupervised))
    
    if args.selfsupervised == 'coca' :
        selfmodel = build_selfsupervisedModel(args)
        selfcriterion = build_selfsupervisedCriterion(imgfeat_loss_coef = 1, cls_loss_coef = 0.1)
        selfmodel.to(device)
        selfcriterion.to(device)
        
    if args.selfsupervised == 'BYOL' :
        selfmodel = build_BYOLModel(args)
        selfcriterion = build_BYOLCriterion()
        selfmodel.to(device)
        selfcriterion.to(device)
    
    if args.selfsupervised == 'data2vec' :
        selfmodel = build_data2vec_Model(args)
        selfcriterion = build_data2vec_Criterion()
        selfmodel.to(device)
        selfcriterion.to(device)
    
    if args.posemodel:
        pose_model = pose_model_petr(embed_dim = args.hidden_dim, using_latent = args.using_latent, feature_gt = args.feature_gt)
        pose_model.to(device)
        
        posecriterion = Pose_Criterion(vc_coef = args.feature_loss_coef, keypoint_coef = args.keypoint_loss_coef, class_coef = args.cls_loss_coef, matcher_keypoint_coef = args.set_cost_keypoint, matcher_class_coef = args.set_cost_class, vc_learn= args.vc_learn)
        if args.feature_gt :
            posecriterion = Pose_Criterion(vc_coef = 0, keypoint_coef = args.keypoint_loss_coef, class_coef = args.cls_loss_coef, matcher_keypoint_coef = args.set_cost_keypoint, matcher_class_coef = args.set_cost_class, vc_learn= False)
        else :
            posecriterion = Pose_Criterion(vc_coef = args.feature_loss_coef, keypoint_coef = args.keypoint_loss_coef, class_coef = args.cls_loss_coef, matcher_keypoint_coef = args.set_cost_keypoint, matcher_class_coef = args.set_cost_class, vc_learn= args.vc_learn)
        #criterion = nn.DataParallel(criterion, device_ids=[0, 1, 2])
        
        posecriterion.to(device)
        
        
    return model, selfmodel, selfcriterion, pose_model, posecriterion
        

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