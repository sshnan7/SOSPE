import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
#from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import os 
import glob
import numpy as np
import random
import queue

import time
import cv2
from scipy import signal
from einops import rearrange, reduce, repeat
from PIL import Image
from signal_processing import *

from collections import deque


class UWBDataset(Dataset):
    def __init__(self, args, mode='train'):
        
        '''
        dataset 처리
        
        mode - train : 학습을 위함.  rf, gt, img 다 있는 경우
                test : test를 위함. rf, gt, img 다 있는 경우 

        '''
        
        self.mode = mode
        self.load_cd = True #load bbox
        self.load_mask = True #False #True if args.pose is not None and mode != 'train' else False
        self.load_img = args.vis and mode != 'train'                                                                                                          

        
        self.is_ftr_normalize = False #True
        self.cutoff = args.cutoff
        
        self.print_once = True

        self.frame_skip = args.frame_skip
        self.slow_time = args.slow_time
        
        self.mixup_prob = args.mixup_prob
        self.mixup_alpha = 1.5

        self.model_debug = args.model_debug
        self.erase_size = 8

        self.three = args.three
        self.num_txrx = args.num_txrx
        
        self.stack_avg = args.stack_avg #True if args.stack_num == args.frame_skip else False #True
        self.store_frames = 64 
        if self.stack_avg > 0:
            outlier_by_ewma = self.stack_avg + 0
        
        
        data_path = '/data/nlos/save_data_ver6'
        
        data_path_list = glob.glob(data_path + '/*')
        data_path_list = sorted(data_path_list)
        
        rf_data = []  # rf data list
        target_list = []
        cd_list = []
        bbox_list = []

        img_list = []
        filename_list =[]
        
        if data_path == '/data/nlos/save_data_ver6' :
            three_dir = [17, 21, 22, 25, 26, 31, 32]
        else :
            three_dir = []
        three_list = []

        remove_dir = []
        
        print("start - data read ", mode)
        

        test_dir = args.test_dir
        
        if data_path == '/data/nlos/save_data_ver6' :
            valid_dir = [9, 6, 38, 37, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
            valid_dir += [0]
            if args.eval:
                valid_dir = list(range(1, 30))
            train_dir = ["A_one_1", "A_one_2", "A_one_3", "A_one_4", "A_one_5", "A_one_6",
                         "B_one_1", "B_one_2", "B_one_3", "B_one_4", "B_one_5",
                         "C_one_1", "C_one_2", "C_one_3", 
                         "D_one_1", "D_one_2", "D_one_3", "D_one_4",

                         "A_two_1", "A_two_2", "A_two_3", 
                         "B_two_1", "B_two_2", "B_two_3",
                         "C_two_1", "C_two_2",
                         "D_two_1", "D_two_2", "D_two_3", "D_two_4", 

                         "C_three_1", "C_three_2",
                         "D_three_1", "D_three_2"
                         
                         "C_four_1",
                         "D_four_1", "D_four_2",
                         ]
            inverse_dir = ["test_E", "test_F"]

            print("if TT", args.TT)
            if args.TT : #domain change
                train_dir = [41]
            

        if self.model_debug or args.eval:
            train_dir = [1, 2]
        
        train_outlier_idx = []
        test_outlier_idx = []

        if mode == 'train':
            print("train_dir = ", train_dir)
            print("valid_dir = ", valid_dir)
            print("remove_dir", remove_dir)
        

        dir_count = 0  # dicrectory index

        #frame_stack = deque(maxlen=self.stack_avg*self.frame_skip)
        not_stacked_list = []
        for tmp in train_dir :
            file  = data_path + "/" + tmp

            #rf list check
            rf_file_list = glob.glob(file + '/radar/*.npy')
            rf_file_list = sorted(rf_file_list)

            #coord gt list check
            if args.gt_head == 'normal' :
                cd_file_list = glob.glob(file + '/HEATMAP_COOR_mpii/*.npy')
            if args.gt_head == 'half' :
                cd_file_list = glob.glob(file + '/coord_mpii/*.npy') 
            if args.gt_head == 'coco' :
                cd_file_list = glob.glob(file + '/PETR_GT/*.npy')
            cd_file_list = sorted(cd_file_list)

            #bbox gt list check
            bbox_file_list = glob.glob(file + '/box_people/*.npy') 
            bbox_file_list = sorted(bbox_file_list)

            #img list check
            img_file_list = glob.glob(file + '/image/*.jpg')
            img_file_list = sorted(img_file_list)

            print('dir :{}'.format(file))
            print("# of rf : {}".format(rf_file_list))
            print("# of cd : {}".format(cd_file_list))
            print("# of bbox : {}".format(bbox_file_list))
            print("# of img : {}".format(bbox_file_list))

            #raw rf data to calculate mean rf
            mean_rf_list = deque(maxlen=self.store_frames)

            #rf processing
            for (file_idx_in_dir, rf) in enumerate(rf_file_list):
                #npy rf from str rf
                raw_rf_load = np.load(rf)

                #preprocessing part
                # error data
                if dir_count in [41, 42]:
                    #print(raw_rf_load.shape)
                    raw_rf_load[(6, 7), :, :] = raw_rf_load[(7, 6), :, :]
                    raw_rf_load[:, (6, 7), :] = raw_rf_load[:, (7, 6), :]
                # cut last some signals
                temp_raw_rf = raw_rf_load[:, :, self.cutoff:]
                temp_raw_rf = torch.tensor(temp_raw_rf).float()
                temp_raw_rf = rearrange(temp_raw_rf, 'tx rx len -> (tx rx) len')

                #store and update mean rf list at first some frames
                if file_idx_in_dir < self.store_frames :
                    mean_rf_list.append(temp_raw_rf)
                
                #put items to dataset
                else :
                    mean_rf = get_mean_rf(mean_rf_list, self.stack_avg)
                    mean_rf = mean_rf.float()
                    #subtract background signal
                    subtract_rf = temp_raw_rf.clone() - mean_rf.clone()
                    rf_data.append(subtract_rf)
                    mean_rf_list.append(temp_raw_rf)

                if file_idx_in_dir == len(rf_file_list) - 1  :
                    print('mean : ', torch.min(mean_rf), torch.max(mean_rf))
                    print("rf : ", torch.min(temp_raw_rf), torch.max(temp_raw_rf))
                    print("rf - mean : ", torch.min(subtract_rf), torch.max(subtract_rf))
            
            #cd gt processing
            for (file_idx_in_dir, coor_gt) in enumerate(cd_file_list):
                if file_idx_in_dir < self.store_frames :
                    continue
                else :
                    np_cd = np.load(coor_gt)
                    np_cd = np_cd[:,:,:2]
                    np_cd[:, :, 0] /= 640#704
                    np_cd[:, :, 1] /= 480#512
                    np_cd = np_cd.reshape(np_cd.shape[0], -1)
                    cd_list.append(np_cd)
                    
                    target_numpy = np.ones(np_cd.shape[0])
                    target_list.append(target_numpy)
                    num_people = np_cd.shape[0]
                    if num_people == 0 :
                        print("coordinate gt error no person! info : {}".format(coor_gt))

            #bbox gt processing
            if self.load_cd :
                for (file_idx_in_dir, bbox_gt) in enumerate(bbox_file_list):
                    if file_idx_in_dir < self.store_frames :
                        continue
                    else :
                        np_bbox = np.load(bbox_gt)
                        bbox_list.append(np_bbox[:, :4])
                        
            #img processing
            if self.load_img:
                for (file_idx_in_dir, img) in enumerate(img_file_list):
                    filename_list.append(f_name)
                    img_list.append(img)

        
        #print("not_stacked_list {}: {} ~ {}".format(len(not_stacked_list), not_stacked_list[0], not_stacked_list[-1]))
        #print(not_stacked_stdev_list)
        self.rf_data = rf_data
        self.cd_list = cd_list
        self.bbox_list = bbox_list
        self.filename_list = filename_list
        self.img_list = img_list
        self.target_list = target_list
        self.three_list = three_list
        
        
        print("rf data len : ", len(self.rf_data))
        print("pose data len : ", len(self.cd_list))
        print("bbox data len : ", len(self.bbox_list))
        print("img data len : ", len(self.img_list))

        self.three_len = len(self.three_list)        
        print("end - data read")
        print("size of dataset", len(self.rf_data))

    def __len__(self):    
        return len(self.rf_data)

    def __getitem__(self, idx):
        f_name = None

        #load rf
        rf = self.get_rf(idx)
        rf = torch.stack(rf)#.permute(1,0,2)

        #load coor gt
        if self.load_cd :
            tmp_cd = torch.FloatTensor(self.cd_list[idx]).clone()
            cd = [tmp_cd]
        else : # for inference
            cd = None
        
        #load bbox
        if self.load_cd :
            tmp_bbox = torch.FloatTensor(self.bbox_list[idx]).clone()
            bbox = [tmp_bbox]
        else :
            bbox = None
        
        #load image
        if self.load_img :
            img = self.img_list[idx]
            img = cv2.imread(img)
            f_name = self.filename_list[idx]
        else :
            img = None
            
        if self.load_cd :
            target = []
            tmp_target = self.target_list[idx ]
            tmp_target = torch.FloatTensor(tmp_target).clone()
            target.append(tmp_target)
            #target = torch.stack(target)
        else :
            target = None
            
        if self.mode=='train':
            return rf, target, cd, bbox

        else:
            return rf, idx, target, cd, bbox, img, f_name
            
       
    def get_rf(self, idx):
        rf = self.rf_data[idx]

        return rf

    def get_hm(self, idx):
        pose = self.hm_list[idx]
        pose = np.load(pose)        
        #pose = np.delete(pose,(1,2,3,4), axis=0)
        return pose


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and (lists of annotations, masks)

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list<tensor>, list<tensor>, list<int>) annotations for a given image are stacked
                on 0 dim. The output gt is a tuple of annotations and masks.
    """
    rfs = []
    targets = []
    cds = []
    bboxs = []
    #hms = []

    for sample in batch:
        #rf = torch.FloatTensor(sample[0]).clone()
        rf = sample[0].clone()
        #cd = torch.FloatTensor(sample[2]).clone() if sample[2] is not None else None
        #hm = torch.FloatTensor(sample[3]).clone() if sample[3] is not None else None
        cd = None
        target = None
        if sample[1] is not None :
            #cd = torch.FloatTensor(sample[3]).clone()
            #target = torch.FloatTensor(sample[2]).clone()
            cd = sample[2]
            target = sample[1]
            bbox = sample[3]
        rfs.append(rf)
        targets.append(target)
        #cds.append(cd)
        #hms.append(hm)

        cds.append(cd)
        bboxs.append(bbox)


    rfs = torch.stack(rfs)
    #if sample[2] is not None :
    #    cds = torch.stack(cds)
 
    return rfs, targets, cds, bboxs

def detection_collate_val(batch):
    rfs = []
    targets = []
    #masks = []
    cds =[]
    bboxs= []
    #hms = []
    ids = []
    imgs = []

    for sample in batch:
        #rf = torch.FloatTensor(sample[0]).clone()  
        rf = sample[0].clone()
        #cd = torch.FloatTensor(sample[3]).clone() if sample[3] is not None else None
        #hm = torch.FloatTensor(sample[4]).clone() if sample[4] is not None else None
        idx = sample[1]#(sample[1], sample[8])
        img = sample[5]
        cd = None
        target = None
        if sample[2] is not None :
            #cd = torch.FloatTensor(sample[4]).clone()
            #target = torch.FloatTensor(sample[3]).clone()
            cd = sample[3]
            bbox = sample[4]
            target = sample[2]

        rfs.append(rf)
        targets.append(target)
        ids.append(idx)
        cds.append(cd)
        bboxs.append(bbox)
        imgs.append(img)


    rfs = torch.stack(rfs)
    #if sample[3] is not None :
    #    cds = torch.stack(cds)

    return rfs, ids, targets, cds, bboxs, imgs


