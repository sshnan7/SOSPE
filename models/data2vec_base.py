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
from .convnext_ver2309 import build_convnext, LayerNorm
from .Light_conv import build_lightbackbone
from .Light_vit import build_lightvitbackbone
from .vit_231016 import build_vitbackbone, build_selfsupervised_criterion
#from .ema_teacher_model import build_teachermodel, build_selfsupervised_criterion

from einops.einops import rearrange, repeat
from einops.layers.torch import Rearrange
from scipy.optimize import linear_sum_assignment
from torch import einsum, nn
import torch.nn.functional as F
#from .selfsupervised_model import build_selfsupervisedModel, build_selfsupervisedCriterion
#from .selfsupervised_BYOL import build_BYOLModel, build_BYOLCriterion
#from .selfsupervised_data2vec import build_data2vec_Model, build_data2vec_Criterion

#from util.pos_embed import get_1d_sincos_pos_embed

        
class pose_model_petr(nn.Module):
    def __init__(self, argument, embed_dim = 256, num_heads = 8, people_queries = 15, pose_dec_depth = 2):
        super().__init__()
        
        self.people_queries = people_queries
        self.embed_dim = embed_dim
        self.pose_embed_dim = self.embed_dim #*3
        self.num_heads = num_heads
        self.keypoint_num = 16
        self.group_size = 8
        
        self.prob_func = nn.Sigmoid()
        
        #self.gt_MLP = nn.Linear(256, 256, bias=False)
        
        #positional encs
        self.pos_enc = PositionalEncoding1D(embed_dim)
        
        #person stage
        #person queries
        self.person_queries = nn.Parameter(torch.zeros(1, people_queries, embed_dim))
        #self.person_queries = torch.zeros(1, people_queries, embed_dim)
        #nn.init.trunc_normal_(self.person_queries, 0.02)
        
        #bounding box decoder
        #bbox_decoder_layer = nn.TransformerDecoderLayer(d_model=self.embed_dim, nhead=num_heads, dim_feedforward=self.embed_dim*4, batch_first=True, activation="gelu")
        #self.bbox_decoder = nn.TransformerDecoder(bbox_decoder_layer, num_layers=pose_dec_depth)
        
        #person decoder
        person_decoder_layer = nn.TransformerDecoderLayer(d_model=self.embed_dim, nhead=argument.nheads, dim_feedforward=self.embed_dim*4, batch_first=True, activation="gelu")
        self.person_decoder = nn.TransformerDecoder(person_decoder_layer, num_layers=pose_dec_depth)
        
        #if self.stage == 2 :
        #    self.pose_queries = nn.Parameter(torch.zeros(1, 16, embed_dim))
        #    pose_decoder_layer = nn.TransformerDecoderLayer(d_model=self.embed_dim, nhead=num_heads, dim_feedforward=self.embed_dim*4, batch_first=True, activation="gelu")
        #    self.pose_decoder = nn.TransformerDecoder(pose_decoder_layer, num_layers=pose_dec_depth)
            
        #cls stage
        self.cls_MLP = nn.Sequential(
            #nn.Linear(embed_dim, embed_dim, bias=False),
            #nn.LayerNorm(embed_dim),
            #nn.ReLU(inplace=False),
            nn.Linear(embed_dim, 1, bias=False),
            #nn.Sigmoid(),
        )
        #box stage
        self.box_MLP = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2, bias=False),
            nn.LayerNorm(embed_dim//2),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.2),
            nn.Linear(embed_dim//2, 4, bias=False),
        )
        
        #pose stage
        if self.embed_dim != self.pose_embed_dim :
            self.PtoP_MLP = nn.Linear(embed_dim, embed_dim, bias=False)
        
       
        self.pose_MLP = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2, bias=False),
            nn.LayerNorm(embed_dim//2),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.2),
            nn.Linear(embed_dim//2, (self.keypoint_num)*2, bias=False),
        )
            
        RF = torch.zeros((64, 256, embed_dim), requires_grad=False)
        self.forward(RF, init_check=True)
    
    def forward(self, RF_latent, former_person_queries = None ,init_check=False) :
        device = RF_latent.device
        memory = RF_latent
        
        b, t, ch = memory.shape
        
        if former_person_queries == None :
            person_queries = repeat(self.person_queries, '1 person_num d -> b person_num d', b = b)
            #person_queries = self.person_queries.expand(b, 15, -1)
        else :
            person_queries = former_person_queries # batch, token_len, dim
        
        #memory_pos_enc = self.pos_enc(memory) #use fixed pos
        #memory = memory + memory_pos_enc.detach()
        person_queries = person_queries.to(device)
        person_pos_enc = self.pos_enc(person_queries) #use fixed pos
        if init_check  == False :
            person_pos_enc = person_pos_enc.to(device)
            person_queries = person_queries.to(device)
        
        person_queries = person_queries + person_pos_enc.detach()
        x = self.person_decoder(person_queries, memory)
        embed_queries = x.clone()
        
        if self.embed_dim != self.pose_embed_dim :
            x = self.PtoP_MLP(x)
        if init_check :
            print("person embed", x.shape)
            print("pos encoding learn", person_pos_enc.requires_grad)
            #print("first person", x[:, 0].shape)
        
        cls = self.cls_MLP(x)
        cls = self.prob_func(cls) #cls_prob
        

        bbox = self.box_MLP(x)
        bbox = self.prob_func(bbox) #cd_prob
        pose = self.pose_MLP(x)
        pose = self.prob_func(pose) #pose_prob
        
        if init_check :
            print("bbox end", bbox.shape)
            print("pose end", pose.shape)
        #wrist = self.pose_MLP_hard(x)
        
        #pose = torch.cat((pose, wrist),2)
        
        
        '''
        batch, person_queries, embed = x.shape
        pose_queries = repeat(self.pose_queries, '1 pose_len d -> b pose_len d', b = b).to(device)
        pose_pos_enc = self.pos_enc(pose_queries)
        pose_queries = pose_queries + pose_pos_enc.to(device)
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
        '''
        
        if init_check :
            print("cls shape : ", cls.shape)
            print("pos shape : ", pose.shape)
            print("grad : ", cls.requires_grad)
        
        out = {'pred_logits': cls, 'pred_keypoint': pose, 'embed_queries' : embed_queries, 'bbox' : bbox}
            
        return out

class Pose_Criterion(nn.Module):
    def __init__(self, use_box_loss = False,  device = None, keypoint_coef = 1, class_coef = 1, bbox_coef = 1, matcher_keypoint_coef = 1, matcher_class_coef = 1, people_queries = 15):
        super().__init__()
        
        self.device = device
        
        self.cost_class = class_coef
        self.cost_keypoint = keypoint_coef
        self.cost_bbox = bbox_coef
        
        self.match_cost_class = matcher_class_coef
        self.match_cost_keypoint = matcher_keypoint_coef
        self.match_cost_bbox = matcher_keypoint_coef
        
        if self.cost_bbox == 0:
            self.match_cost_bbox = 0
        if self.cost_keypoint == 0 :
            self.match_cost_keypoint = 0
        
        if use_box_loss :
            self.box_loss = nn.MSELoss(reduction='mean').to(device)
        else :
            self.box_loss = None
        
        self.class_loss = nn.BCELoss().to(device) #nn.L1Loss(reduction='mean').to(device) #nn.BCELoss().to(device)
        self.keypoint_loss = nn.L1Loss(reduction='mean').to(device) #nn.MSELoss().to(device)#nn.L1Loss(reduction='mean').to(device)
        
        
        output_ex = []
        gt_ex = []
        outputs_class = torch.rand(64, 15, 1)
        outputs_keypoint = torch.rand(64, 15, 2*16)
        outputs_bbox = torch.rand(64, 15, 4)
        out_ex = {'pred_logits': outputs_class, 'pred_keypoint': outputs_keypoint, 'bbox' : outputs_bbox}
        feat = torch.rand(64, 256, 256)
        for i in range(32) :
            gt_class = torch.ones((3, 1))
            gt_keypoint = torch.rand(3, 2*16)
            gt_feat = torch.rand(256, 256)
            gt_bbox = torch.rand(3, 4)
            tmp_gt_ex = {'labels' : gt_class, 'cd' : gt_keypoint, 'features' : gt_feat, 'bbox' : gt_bbox}
            gt_ex.append(tmp_gt_ex)
        for i in range(16) :
            gt_class = torch.tensor([])
            gt_keypoint = torch.tensor([])
            gt_bbox = torch.tensor([])
            tmp_gt_ex = {'labels' : gt_class, 'cd' : gt_keypoint, 'features' : gt_feat, 'bbox' : gt_bbox}
            gt_ex.append(tmp_gt_ex)
        for i in range(16) :
            gt_class = torch.ones((2, 1))
            gt_keypoint = torch.rand(2, 2*16)
            gt_bbox = torch.rand(2, 4)
            tmp_gt_ex = {'labels' : gt_class, 'cd' : gt_keypoint, 'features' : gt_feat, 'bbox' : gt_bbox}
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
            temp_tgt_bbox = gt[i]["bbox"]
            if temp_tgt_keypoint.shape[0] == 0 :
                batch_row.append(torch.tensor([]))
                batch_col.append(torch.tensor([]))
                continue
            
            temp_out_prob = outputs["pred_logits"][i]
            temp_out_keypoint = outputs["pred_keypoint"][i]
            temp_out_bbox = outputs["bbox"][i]
            
            temp_cost_class = 1-temp_out_prob

            
            # Compute the L1 cost between keypoint
            temp_cost_keypoint = torch.cdist(temp_out_keypoint, temp_tgt_keypoint, p=1)
            temp_cost_bbox = torch.cdist(temp_out_bbox, temp_tgt_bbox, p=1)
                
            # Final cost matrix
            temp_C = self.match_cost_class * temp_cost_class + self.match_cost_keypoint * temp_cost_keypoint + self.match_cost_bbox * temp_cost_bbox
            temp_C = temp_C.detach().cpu().numpy() #linear sum assignment can't work at torch
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
                    batch_output_bbox = output['bbox'][i][tmp_rows[j]]
                    batch_gt_bbox = gt[i]["bbox"][tmp_cols[j]]
                    
                    '''
                    batch_output_coor_wrist = output['pred_keypoint'][i][tmp_rows[j]][2*10:2*11]
                    batch_gt_coor_wrist = gt[i]["cd"][tmp_cols[j]][2*10:2*11]
                    
                    batch_output_coor_wrist = torch.cat((batch_output_coor_wrist, output['pred_keypoint'][i][tmp_rows[j]][2*15:2*16]))
                    batch_gt_coor_wrist = torch.cat((batch_gt_coor_wrist, gt[i]["cd"][tmp_cols[j]][2*15:2*16]))
                    '''
                else :
                    batch_output_coor = torch.cat((batch_output_coor, output['pred_keypoint'][i][tmp_rows[j]]))
                    batch_gt_coor = torch.cat((batch_gt_coor, gt[i]["cd"][tmp_cols[j]]))
                    batch_output_bbox = torch.cat((batch_output_bbox, output['bbox'][i][tmp_rows[j]]))
                    batch_gt_bbox = torch.cat((batch_gt_bbox, gt[i]["bbox"][tmp_cols[j]]))
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
                bbox_loss = self.keypoint_loss(batch_output_bbox, batch_gt_bbox)
                #wrist weight +
                #keypoint_loss = keypoint_loss + 15*self.keypoint_loss(batch_output_coor_wrist, batch_gt_coor_wrist)
            else :
                keypoint_loss = keypoint_loss + self.keypoint_loss(batch_output_coor, batch_gt_coor)
                bbox_loss = bbox_loss + self.keypoint_loss(batch_output_bbox, batch_gt_bbox)
                #wrist weight +
                #keypoint_loss = keypoint_loss + 15*self.keypoint_loss(batch_output_coor_wrist, batch_gt_coor_wrist)
                
            non_less_bs += 1
        
        if class_loss == None :
            class_loss = torch.tensor(0)
        if keypoint_loss == None :
            keypoint_loss = torch.tensor(0)
        
        losses = {
            'cls' : class_loss/non_less_bs,
            'cd' : keypoint_loss/non_less_bs,
            'bbox' : bbox_loss/non_less_bs
        }
        
        if init_check :
            print("class grad {} // cd grad {}", class_loss.requires_grad, keypoint_loss.requires_grad)
            print(class_loss, keypoint_loss)
            
        sum_loss = self.cost_class*class_loss/non_less_bs + self.cost_keypoint*keypoint_loss/non_less_bs + self.cost_bbox*bbox_loss/non_less_bs 
        
        return losses, sum_loss, first_gt_class
        


def build_model(args):   
    device = torch.device(args.device)
    print("backbone type : ", args.backbone_type)
    '''
    if args.backbone_type == 'convnext' :
        convbackbone = build_convbackbone(args)
        vc_convbackbone = build_convbackbone(args)
    elif args.backbone_type == 'light_conv' :
        convbackbone = build_lightbackbone(args)
        vc_convbackbone = build_lightbackbone(args)
    elif args.backbone_type == 'light_vit' :
        convbackbone = build_lightvitbackbone(args)
        vc_convbackbone = build_lightvitbackbone(args)
    if args.backbone_type == 'convnext2d' :
        convbackbone = build_convnext(args)
    if args.backbone_type == 'vit' :
        convbackbone = build_vitbackbone(args)
    '''
    
    model = build_vitbackbone(args) #Encoder_model(backbone= convbackbone, selfsupervised = args.selfsupervised)
    model.to(device)
    
    selfmodel = None
    selfcriterion = None
    pose_model = None
    posecriterion = None
    '''
    if args.selfsupervised  :
        print("self supervised training ".format(args.selfsupervised))
        selfmodel = build_teachermodel(model)
        selfcriterion = build_selfsupervised_criterion()
        
        selfmodel.to(device)
        selfcriterion.to(device)
    '''
    if args.pretrain or args.TT :
        selfcriterion = build_selfsupervised_criterion(args)
        selfcriterion.to(device)
    
    if args.posemodel:
        pose_model = pose_model_petr(argument = args, embed_dim = args.hidden_dim, pose_dec_depth = args.dec_layers)
        posecriterion = Pose_Criterion(keypoint_coef = args.keypoint_loss_coef, class_coef = args.cls_loss_coef, bbox_coef = args.bbox_loss_coef, matcher_keypoint_coef = args.set_cost_keypoint, matcher_class_coef = args.set_cost_class)
        
        #criterion = nn.DataParallel(criterion, device_ids=[0, 1, 2])
        
        pose_model.to(device)
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