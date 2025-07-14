# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import util.misc as utils
from engine import train_one_epoch_downstream, evaluate_downstream
from models.data2vec_base import build_model
from uwb_dataset7 import UWBDataset, detection_collate, detection_collate_val
#from uwb_dataset6 import UWBDataset, detection_collate, detection_collate_val

from custom_lr_scheduler import CosineAnnealingWarmUpRestarts
from collections import defaultdict

#torch.backends.cudnn.deterministic = True #or you can use underline
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.benchmark = False
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float) # 0.0004
    parser.add_argument('--lr_min', default=2e-5, type=float) # 0.00002
    parser.add_argument('--batch_size', default=128, type=int) # 32
    parser.add_argument('--weight_decay', default=1e-4, type=float) # 0.0001(1e-4)
    parser.add_argument('--beta1', default=0.9, type=float) # 0.0001(1e-4)
    parser.add_argument('--beta2', default=0.999, type=float) # 0.0001(1e-4)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                       help='gradient clipping max norm')
                                            
    # * Backbone
    parser.add_argument('--backbone', type=str, default='convnext', choices=('convnext', 'mlpmixer'),
                        help="model_size")
    parser.add_argument('--backbone_type', type=str, default='vit', choices=('convnext', 'light_conv', 'light_vit', 'convnext2d', 'vit'),
                        help="model_type")
    parser.add_argument('--embtype', type=str, default='single', choices=('single', 'channel_wise'),
                        help="embtype")                    
    parser.add_argument('--model_scale', type=str, default='s', choices=('t', 's', 'm', 'l'),
                        help="model_size") 
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('none', 'sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--rf_interpretation', default='2d', type=str, choices=('2d', '1d'),
                        help="rf interpret 2d or 1d")
    
    # * Main model
    parser.add_argument('--in_chan', default=64, type=int, # 6
                        help="Number of signal encoder input channel 16 or 64")
    parser.add_argument('--enc_layers', default=2, type=int, # 6
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=2, type=int, # 6
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--stride', default=2, type=int, # 6
                        help="signal block stride")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--kernel', default=6, type=int,
                        help="Size of the signal kernel ")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float, #0.1
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=4, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=15, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--former_info', action='store_true', default=False, 
                        help="if use former info")
    parser.add_argument('--former_person_info', action='store_true', default=False, 
                        help="if use former person emb info")
    parser.add_argument('--use_feature', type=str, default='HR', choices=('HR', 'PETR'),
                        help="model_size") 
    parser.add_argument('--posemodel', action='store_true', default=False, 
                        help="if use posemodel")
    parser.add_argument('--selfmodel', action='store_true', default=False, 
                        help="if use selfmodel")
    parser.add_argument('--pretrain', action='store_true', default=False, 
                        help="if use selfsupervised")
    parser.add_argument('--group_idx', default=4, type=int, 
                        help="using group number")
    parser.add_argument('--tx_num_masking', default=0, type=int, 
                        help="number of tx masking")
    parser.add_argument('--rx_num_masking', default=0, type=int,  
                        help="number of rx masking")
    parser.add_argument('--mask_ratio', default=0., type=float, 
                        help="ratio of masking")
    parser.add_argument('--inv_mask_block', default=1, type=int,  
                        help="inverse mask block size")
    parser.add_argument('--alpha', default=1., type=float, 
                        help="ratio of pretrained model lr coef")
    parser.add_argument('--unfrozen_layer', default=0, type=int,  
                        help="unfrozen weights layer")
    parser.add_argument('--original_data2vec', action='store_true', default=False, 
                        help="if pretrain method original data2vec")
    parser.add_argument('--TT', action='store_true', default=False, 
                        help="if do test training(selfsupervised)")

    # Loss
    # * Hungarian Matcher hyperparam
    parser.add_argument('--set_cost_class', default=20, type=float, #20
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_keypoint', default=1, type=float, # origin : 1
                        help="L1 keypoint coefficient in the matching cost")
                        

    # * Loss coefficient
    parser.add_argument('--keypoint_loss_coef', default=50, type=float) # origin : 100

    parser.add_argument('--cls_loss_coef', default=1, type=float) # origin : 1
    parser.add_argument('--bbox_loss_coef', default=1, type=float) # origin : 1
    parser.add_argument('--raw_loss_coef', default=1, type=float) # origin : 1
    parser.add_argument('--vec_loss_coef', default=1, type=float) # origin : 1

    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--output_dir', default='./weights/ver4/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--fineresume', default='', help='resume finetuning model from checkpoint')
    parser.add_argument('--downstream', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain_model', default='', help='pretrained model')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=16, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    # rf dataset parameter
    parser.add_argument('--cutoff', type=int, default=256, #512,
            help='cut off the front of the input data,   --> length = 2048 - cutoff')  
    parser.add_argument('--stack_num', type=int, default=1,
                help = 'number of frame to stack')
    parser.add_argument('--frame_skip', type=int, default=1, # 1
                help = 'number of frame to skip when stacked')
    parser.add_argument('--stack_avg', type=int, default=64,
                help='use ewma adjust')
    parser.add_argument('--num_txrx', type=int, default=8,#4
                help='# of used tx & rx')
    parser.add_argument('--slow_time', type=int, default=1,#4
                help='# using how many stacked frame input')
    parser.add_argument('--gt_head', type=str, default='half', choices=('normal', 'half', 'coco'),
                        help="gt_head_location")
    
    # evaluate paramater
    parser.add_argument('--vis', action='store_true', default=False,
                help='visualize the image for debugging')
    parser.add_argument('--img_dir', default=None, type=str,
                        help='path where to save evaluation image')
    parser.add_argument('--box_threshold', default=0.5, type=float,
                        help="confidence threshold to print out")
    parser.add_argument('--test_dir', default='9,6,38', 
                        type=lambda s: [int(item) for item in s.split(',')])
    
    # freebies parameter
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                help="lr_scheduler (linear or cosine)")
    parser.add_argument('--drop_prob', default=0.3, type=float,
                        help="drop_path drop probability")
    parser.add_argument('--dropblock_prob', default=0.2, type=float,
                        help="dropblock drop probability")
    parser.add_argument('--drop_size', type=int, default=4,
                help = 'size of drop block')
    parser.add_argument('--model_debug', action='store_true', default=False,
                help='small test set for debug')
    parser.add_argument('--mixup_prob', default=0., type=float, #0.3
                        help="mixup_probability")
    parser.add_argument('--three', default=0.5, type=float,
                help='more three data')     

    return parser


def main(args):
    utils.init_distributed_mode(args)
   #print("git:\n  {}\n".format(utils.get_sha()))
    print("torch version ", torch.__version__)
    
    print("----------------pre train learning {}-------------------".format(args.pretrain))

    print(torch.backends.cudnn.benchmark)
    
    output_dir = Path(args.output_dir)
    if args.output_dir and not args.eval and utils.is_main_process():
        with (output_dir / "args.txt").open("a") as f:
            for arg in vars(args):
                f.write(arg + " = " + str(getattr(args, arg)) + "\n")

    device = torch.device(args.device)
    if args.TT :
        args.pretrain = True 
        print("------------------------Test training pretrain mode on--------------------")

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    
    model, selfmodel, selfcriterion, pose_model, posecriterion = build_model(args)

    model_without_ddp = model

    #not implemented cause we experiment on single gpu env
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    pose_model_without_ddp = pose_model
    selfmodel_without_ddp = selfmodel
    
    param_dicts = []

    
    n_parameters = 0
    n_pretrain_parameters = 0
    
    for n, p in model_without_ddp.named_parameters() :
        if args.pretrain_model or args.downstream :
            p.requires_grad = False
            #remember pretrain parms
            if 'transformer_encoder' in n :
                transformer_layer_num = n[len('transformer_encoder')+1:len('transformer_encoder')+2]
                if (4 - args.unfrozen_layer) <= int(transformer_layer_num) :
                    param_dicts += [{"params" : p, "alpha" : (args.alpha)**(4-int(transformer_layer_num))}]
                else :
                    param_dicts += [{"params" : p, "alpha" : 1}]
            
            else : 
                param_dicts += [{"params" : p, "alpha" : 1}]
            
        else :
            param_dicts += [{"params" : p, "alpha" : 1}]
        n_parameters += p.numel()
        
    print('number of learnable encoder params:', n_parameters)
    print('number of pretrain params:', n_pretrain_parameters)
    
            
            
    if pose_model != None :
        print("add pose model")
        for n, p in pose_model_without_ddp.named_parameters() :
            if p.requires_grad :
                param_dicts += [{"params" : p, "alpha" : 1}]
                n_parameters += p.numel()
        print('number of whole model learnable params:', n_parameters)
    
                
    if selfmodel != None :
        for n, p in selfmodel_without_ddp.named_parameters() :
            if p.requires_grad :
                param_dicts += [{"params" : p}]

    if args.lr_scheduler == 'linear' or args.TT:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=args.beta1, weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, 0.1) # after {lr_drop} epoch, lr drop to 1/10 
    elif args.lr_scheduler =='cosine':
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr_min, betas = (args.beta1, args.beta2), weight_decay=args.weight_decay)
        #optimizer = torch.optim.AdamW([{'params': param_dicts, 'lr': args.lr_min}], lr = args.lr_min, weight_decay=args.weight_decay)
        
        lr_scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=10, T_mult=1, eta_max=args.lr,  T_up=10, gamma=0.75)
    
    if args.pretrain_model :
        print("-------------------------------load pretrain encoder model----------------------------------------")
        print(args.pretrain_model)
        checkpoint = torch.load(args.pretrain_model, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if 'pose_model' in checkpoint :
            pose_model_without_ddp.load_state_dict(checkpoint['pose_model'])
    
    if args.downstream :
        print("downstream process")
        checkpoint = torch.load(args.downstream, map_location='cpu')
        print("-------------------------------load downstream encoder model----------------------------------------")
        model_without_ddp.load_state_dict(checkpoint['model'])
    
    if args.resume or args.fineresume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            if args.resume == '' :
                checkpoint = torch.load(args.fineresume, map_location='cpu')
            else :
                checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if args.posemodel :
            pose_model_without_ddp.load_state_dict(checkpoint['pose_model'])
        if not ((args.eval) or args.TT) and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
    
    
    # UWB Dataset
    dataset_train = UWBDataset(mode='train', args=args)
    dataset_val = UWBDataset(mode='test', args=args)
    
    data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, collate_fn=detection_collate, num_workers=args.num_workers, pin_memory=True)
    data_loader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, collate_fn=detection_collate_val, num_workers=args.num_workers, pin_memory=True)
    

    test_type = 'x' if args.test_dir[0] in [44, 441, 442, 443, 444, 45, 451, 452, 453, 454, 46, 47] else 'y' if (args.test_dir[0] > 100 and args.test_dir[0] < 200) else 'b' #in [101, 102, 103, 104, 105] else 'b'
    
    # Dataset Debug
    if args.eval:
        test_stats = evaluate_downstream(
                model_without_ddp, selfmodel_without_ddp, selfcriterion, pose_model_without_ddp, posecriterion, data_loader_val, args.batch_size, device, \
                args.output_dir, val_vis=args.vis, img_dir=args.img_dir, \
                            boxThrs=args.box_threshold, epoch=-1, \
                                        test_type = test_type, gt_head = args.gt_head, \
                                        pretrain = args.pretrain, downstream = args.downstream, former_info = args.former_info , former_person_info = args.former_person_info)
        return
    
    print("Start training")
    save_freq = 1
    if args.pretrain :
        save_freq = 5
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        
        if args.start_epoch > 0 or epoch == 1 :
            print("finetunning start")
            #pretrain parms
            for n, p in model_without_ddp.named_parameters()  :
                if args.pretrain_model or args.downstream :
                    if (4 - args.unfrozen_layer) <= -1 : #all weight
                        p.requires_grad = True
                        print(n, "requires grad True")
                    else :
                        if 'transformer_encoder' in n :
                            transformer_layer_num = n[len('transformer_encoder')+1:len('transformer_encoder')+2]
                            if (4 - args.unfrozen_layer) <= int(transformer_layer_num) :
                                p.requires_grad = True
                                print(n, "requires grad True")
                
                
        train_stats = train_one_epoch_downstream(
            model_without_ddp, selfmodel_without_ddp, selfcriterion, pose_model_without_ddp, posecriterion, data_loader_train, args.batch_size, optimizer, device, epoch,
            args.clip_max_norm, pretrain = (args.pretrain or args.TT), downstream = args.downstream, former_info = args.former_info, former_person_info = args.former_person_info)
            
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if (epoch + 1) % save_freq == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                if pose_model == None and selfmodel == None:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)
                elif selfmodel == None :
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'pose_model' : pose_model_without_ddp.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)
                elif pose_model == None :
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'self_model' : selfmodel_without_ddp.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)
                else:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'self_model' : selfmodel_without_ddp.state_dict(),
                        'pose_model' : pose_model_without_ddp.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)
        
        print("Former info", args.former_info)
        if args.TT :
            test_stats = evaluate_downstream(
                model_without_ddp, selfmodel_without_ddp, selfcriterion, pose_model_without_ddp, posecriterion, data_loader_val, args.batch_size, device, \
                args.output_dir, val_vis=args.vis, img_dir=args.img_dir, boxThrs=args.box_threshold, epoch=epoch, test_type = test_type, gt_head = args.gt_head, \
                pretrain = args.pretrain, downstream = args.downstream, former_info = args.former_info , former_person_info = args.former_person_info)
        else :
            test_stats = evaluate_downstream(
                model_without_ddp, selfmodel_without_ddp, selfcriterion, pose_model_without_ddp, posecriterion, data_loader_val, args.batch_size, device, \
                args.output_dir, val_vis=args.vis, img_dir=args.img_dir, boxThrs=args.box_threshold, epoch=epoch, test_type = test_type, gt_head = args.gt_head, \
                pretrain = args.pretrain, downstream = args.downstream, former_info = args.former_info , former_person_info = args.former_person_info)
        
        train_log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        test_log_stats = {**{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process() :
            with (output_dir / "train_log.txt").open("a") as f :
                f.write(json.dumps(train_log_stats) + "\n")
            with (output_dir / "test_log.txt").open("a") as f :
                f.write(json.dumps(test_log_stats) + "\n")
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('RFTR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
