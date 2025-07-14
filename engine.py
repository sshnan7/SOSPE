# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

from operator import is_

from typing import Iterable
import time
from datetime import datetime
from pathlib import Path

import torch

import numpy as np

import util.misc as utils
from util import box_ops
from evaluate import pck_mAP,  pose_AP_mpii

def forward_model(samples, target, backbone_model = None, selfmodel = None, posemodel = None, precriterion = None, posecriterion = None, pretrain = False, downstream = False) :
    if pretrain :
        first_gt = None
        outputs = None
        if selfmodel != None :
            selfmodel_outputs = selfmodel(samples[:, :, 0])
            online_outputs, online_mask, emb_mask = backbone_model(samples[:, :, 0])#batch, ch, frames
            loss_dict, sum_loss = precriterion(selfmodel_outputs, online_outputs, online_mask, emb_mask)
        else :
            raw_pred, vec_pred, raw_target, vec_target, emb_mask = backbone_model(samples[:, :, 0])#batch, ch, frames
            loss_dict, sum_loss = precriterion(raw_pred, vec_pred, raw_target, vec_target, emb_mask)
            
    elif downstream :
        online_outputs = backbone_model(samples[:, :, 0])
        outputs = posemodel(online_outputs)
        loss_dict, sum_loss, first_gt = posecriterion(outputs, target[:, 0])
        
    else :
        online_outputs = backbone_model(samples[:, :, 0]) #batch, ch, frames
        outputs = posemodel(online_outputs)
        loss_dict, sum_loss, first_gt = posecriterion(outputs, target[:, 0])
    
    return loss_dict, sum_loss, first_gt, outputs

def train_one_epoch_downstream(model: torch.nn.Module, selfmodel : torch.nn.Module, selfcriterion : torch.nn.Module, posemodel : torch.nn.Module, posecriterion: torch.nn.Module,
                    data_loader: Iterable, batchsize, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                     pretrain = False, downstream = '', former_info = False, former_person_info = True):
    model.train()
        
    if not pretrain :
        posemodel.train()
        posecriterion.train()
    if selfmodel != None or pretrain:
        selfcriterion.train()
        if selfmodel != None :
            selfmodel.eval()
    if downstream :
        if selfmodel != None :
            selfmodel.eval()
            selfcriterion.eval()
        print("downstream : {}".format(downstream))
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    print_per_epoch = 3
    print_freq = 170000 //batchsize//print_per_epoch
    
    selfupdate_per_epoch = 10
    selfupdate_freq = 170000 //batchsize//selfupdate_per_epoch
    
    iterate = 0
    for samples, targets, cds, bboxs in metric_logger.log_every(data_loader, print_freq, header):
        iterate += 1
        samples = samples.to(device)
        pose_target = []
            
        for i in range(len(samples)): #len(target) : batch size
            t = samples[i]
            cd = cds[i]
            bbox = bboxs[i]
            target = targets[i] #.to(device)
            pose_target_frames = []

            #consider multi frame. Now len(cd) == 1
            for frame in range(len(cd)) :
                if frame == 0 :
                    pose_t = {'labels': target[frame].long().to(device), 'cd':cd[frame].to(device), 'bbox' : bbox[frame].to(device)}
                else :
                    pose_t = {'labels': target[frame].long().to(device), 'cd':cd[frame].to(device), 'bbox' : bbox[frame].to(device)}
                pose_target_frames.append(pose_t) # i batches, 3frames
            
            pose_target.append(pose_target_frames) # put batches
        pose_target = np.array(pose_target)

        loss_dict, sum_loss, first_gt, outputs = forward_model(samples = samples, target = pose_target, backbone_model = model, selfmodel = selfmodel, posemodel = posemodel, 
            precriterion = selfcriterion, posecriterion = posecriterion, pretrain = pretrain, downstream = downstream)
        

        if iterate % (print_freq//3) == 1 :
            if sum_loss != None :
                for loss_key in loss_dict :
                    print("{} : {}".format(loss_key, loss_dict[loss_key]))
            
        if iterate ==  1 and not pretrain :
            print(outputs['pred_logits'][0].view(-1))
            print(first_gt[0].view(-1))
            for loss_key in loss_dict :
                print("{} : {}".format(loss_key, loss_dict[loss_key]))
            
        
        frames = samples.shape[2]
        for frame in range(1, frames) :
            if former_info :
                online_outputs = model(samples[:, :, frame], former_data = online_outputs) 
            if posemodel != None :
                if former_person_info :
                    outputs = posemodel(online_outputs, outputs["embed_queries"])
                else :
                    outputs = posemodel(online_outputs)
                tmp_loss_dict, tmp_sum_loss, tmp_first_gt = posecriterion(outputs, pose_target[:, frame])
            sum_loss += tmp_sum_loss
        optimizer.zero_grad()
        sum_loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        if posemodel != None :
            metric_logger.update(loss=sum_loss, **loss_dict)
        if selfmodel != None or pretrain:
            metric_logger.update(loss=sum_loss, **loss_dict)
            
            if selfmodel != None :
                if iterate % selfupdate_freq == 0 : 
                    selfmodel.step(model)
                    print("self model update")
                    
        if selfmodel != None or posemodel != None or pretrain:
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    
    metric_logger.synchronize_between_processes()
    print("time: ", str(datetime.now()))
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    

@torch.no_grad()
def evaluate_downstream(model, selfmodel, selfcriterion, posemodel, posecriterion, data_loader, batchsize, device, output_dir, val_vis, img_dir, boxThrs, epoch=-1, test_type = 'b', gt_head = 'half', use_only_gt = False, pretrain = False, downstream = '', former_info = False, former_person_info = True):
    
    model.eval()
    
    if posemodel != None :
        posemodel.eval()
    if selfmodel != None :
        selfmodel.eval()
    
    output_folder = img_dir

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    ap_threshold_list = [0.5]
    pose_ap_results = {k:[] for k in ap_threshold_list}
    origin_size = torch.tensor([512, 512])
    
    print_per_epoch = 2
    print_freq = 3000 //batchsize//print_per_epoch
    
    iterate = 0

    for samples, ids, targets, cds, bboxs, imgs in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        pose_target = []
        iterate += 1
        best_ap = 0
        for i in range(len(samples)): #len(target) : batchi size
            t = samples[i]
            cd = cds[i]
            bbox = bboxs[i]
            idx = ids[i]
            img = imgs[i]
            target = targets[i]
            pose_target_frames = []
            for frame in range(len(cd)) :
                if frame == 0 :
                    pose_t = {'labels': target[frame].long().to(device), 'cd':cd[frame].to(device), 'bbox' : bbox[frame].to(device), 'f_id' : idx, 'img' : img, 'orig_size':origin_size.to(device), 'image_id':idx}
                else :
                    pose_t = {'labels': target[frame].long().to(device), 'cd':cd[frame].to(device), 'bbox' : bbox[frame].to(device), 'f_id' : idx, 'img' : img, 'orig_size':origin_size.to(device), 'image_id':idx}
                pose_target_frames.append(pose_t)

            pose_target.append(pose_target_frames)
        pose_target = np.array(pose_target)
        
        loss_dict, sum_loss, first_gt, outputs = forward_model(samples = samples, target = pose_target, backbone_model = model, selfmodel = selfmodel, posemodel = posemodel, 
            precriterion = selfcriterion, posecriterion = posecriterion, pretrain = pretrain, downstream = downstream)

    
        
        if iterate % print_freq == 1 :
            if sum_loss != None :
                for loss_key in loss_dict :
                    print("{} : {}".format(loss_key, loss_dict[loss_key]))
            
        if not pretrain :
            results = []
            batch_size, query_num = outputs['pred_keypoint'].shape[0], outputs['pred_keypoint'].shape[1]
            for batch in range(batch_size) :
                batch_output_keypoint = outputs['pred_keypoint'][batch]
                batch_output_keypoint = batch_output_keypoint.view(query_num, -1, 2)
                batch_output_score = outputs['pred_logits'][batch]
                results.append({'scores' : batch_output_score, 'keypoint' : batch_output_keypoint})
            
            if iterate == 1 :
                print(outputs['pred_logits'][0].view(-1))
                print(first_gt[0].view(-1))
                print_log = True
            else :
                print_log = False
                    
            
            for i in ap_threshold_list:
                
                mpii_result = pose_AP_mpii(pose_target[:, -1], results, imgs, IOUThreshold=i, \
                                                vis=val_vis, img_dir=output_folder, boxThrs=boxThrs, \
                                                test_type=test_type, gt_head = gt_head, video_num = iterate, best_iter_ap = best_ap, print_log = print_log
                )
                pose_ap_results[i].extend(mpii_result)

        
    if (epoch == -1 or epoch % 1 == 0) and posemodel!= None:
        ap_output_dir = Path(output_dir)
        if ap_output_dir != None:
            with (ap_output_dir / "mAP.txt").open("a") as f:
                for i in ap_threshold_list:
                    kpt = pck_mAP(pose_ap_results[i])
                    print(f"PCK {i} = TOT {kpt[8]} || HEAD {kpt[0]} | NECK {kpt[1]} | SHO {kpt[2]} | ELB {kpt[3]} | WRI {kpt[4]} | HIP {kpt[5]} | KNE {kpt[6]} | ANK {kpt[7]}")
                    if epoch != -1 : f.write(f"PCK {i} = TOT {kpt[8]} || HEAD {kpt[0]} | NECK {kpt[1]} | SHO {kpt[2]} | ELB {kpt[3]} | WRI {kpt[4]} | HIP {kpt[5]} | KNE {kpt[6]} | ANK {kpt[7]}\n")
        

        # reduce losses over all GPUs for logging purposes
        metric_logger.update(loss=sum_loss, **loss_dict)
    if selfmodel == None and posemodel == None:
        metric_logger.update(loss=sum_loss, **loss_dict)
                

    
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    return stats
