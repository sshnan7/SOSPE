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
        self.load_cd = True
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
        
        self.load_cd = True
        
        
        data_path = '/data/nlos/save_data_ver6'
        
        data_path_list = glob.glob(data_path + '/*')
        data_path_list = sorted(data_path_list)
        
        rf_data = []  # rf data list
        train_idx_list = []
        raw_list = []
        target_list = []
        mask_list = []  # ground truth mask
        hm_list = []
        cd_list = []
        bbox_list = []

        img_list = []
        filename_list =[]
        
        if data_path == '/data/nlos/save_data_ver6' :
            three_dir = [17, 21, 22, 25, 26, 31, 32]
        else :
            three_dir = []
        three_list = []
        outlier_list = [] 
        remove_dir = []
        
        stdev_set = torch.Tensor([0.2509, 0.1846, 0.1719, 0.1611, 0.1305, 0.1215, 0.1212, 0.1206, 0.1832,
                    0.1733, 0.1568, 0.1437, 0.1350, 0.1257, 0.1235, 0.1152, 0.1508, 0.1354,
                    0.1302, 0.1232, 0.1714, 0.1520, 0.1465, 0.1344, 0.1224, 0.1151, 0.1157,
                    0.1088, 0.2353, 0.2023, 0.1768, 0.1491, 0.1661, 0.1809, 0.2009, 0.2127,
                    0.1280, 0.1239, 0.1076, 0.1111, 0.1495, 0.1419, 0.1480, 0.1538, 0.1353,
                    0.1292, 0.1221, 0.1219, 0.1285, 0.1218, 0.1268, 0.1245, 0.1471, 0.1462,
                    0.1559, 0.1642, 0.1151, 0.1165, 0.1297, 0.1242, 0.1915, 0.1954, 0.2059,
                    0.2514])
        mean_set = torch.Tensor([ 0.0078,  0.0077, -0.0199, -0.0172,  0.0112,  0.0115,  0.0035,  0.0187,
                  -0.0118,  0.0142,  0.0144,  0.0002,  0.0119,  0.0174, -0.0044, -0.0119,
                   0.0117,  0.0019, -0.0014,  0.0045,  0.0137,  0.0150, -0.0159, -0.0140,
                   0.0028, -0.0002,  0.0055,  0.0181,  0.0157,  0.0041, -0.0169, -0.0140,
                  -0.0151, -0.0183,  0.0017,  0.0046,  0.0179, -0.0031,  0.0066,  0.0101,
                   0.0038,  0.0099,  0.0069,  0.0020, -0.0053,  0.0092,  0.0076,  0.0066,
                   0.0109,  0.0002,  0.0085,  0.0090,  0.0102,  0.0021,  0.0088,  0.0149,
                   0.0185,  0.0046, -0.0102,  0.0117, -0.0161, -0.0169, -0.0156,  0.0021])
        
        print("start - data read ", mode)
        # 데이터셋 세부 내용은 비가시drive/미팅자료/8월 데이터 수집 참고
        test_dir = args.test_dir
        # 9 - B_one_1 3000
        # 6 - A_two_1 2100
        # 38 - test_C_four 2500
        # 26 - D_four_2 3000
        # 35 - D_two_3 2000
        # 37 - cloth 39 - 스티로폼 40 - Wall
        # 41 - E, 42 - F
        #valid_dir = [8, 17] #list(range(18, 30)) #[8, 17] 
        if data_path == '/data/nlos/save_data_ver6' :
            valid_dir = [9, 6, 38, 37, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
            valid_dir += [0]
            if args.eval:
                valid_dir = list(range(1, 30))
            train_dir = [x for x in list(range(43)) if x not in valid_dir] # 29
            print("if TT", args.TT)
            if args.TT : #domain change
                train_dir = [41]
            
        else :
            valid_dir = [1]
            #train_dir = [x for x in list(range(25)) if x not in valid_dir] # 29
            train_dir = [x for x in list(range(43)) if x not in valid_dir]
            #train_dir = [0,2,3,4, 5,6,7,8, 10, 11,12,14, 15, 16, 20] # 29
            #outlier_list = [4, 9, 15, 19, 20]
        #valid_dir = [37, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

        if self.model_debug or args.eval:
            train_dir = [1, 2]
        
        train_outlier_idx = []
        test_outlier_idx = []
        #test_outlier_idx = [0, 8835, 8835+12000, 8835+12000+13000]
        if mode == 'train':
            print("train_dir = ", train_dir)
            print("valid_dir = ", valid_dir)
            print("remove_dir", remove_dir)
        if mode == 'test':
            if self.model_debug:
                test_dir = [6] if not self.load_cd else [42]#[38] # 18, 22
            elif not args.eval  :
                if data_path == '/data/nlos/save_data_ver6' :
                    test_dir = [42]
                else :
                    test_dir = [30]
            
            if data_path == '/data/nlos/save_data_ver6' :
                test_set = args.test_dir[0]
                if test_set == 81:
                    test_dir = [9]
                elif test_set == 82:
                    test_dir = [6] 
                elif test_set == 83:
                    test_dir = [38]
                    outlier_list = list(range(2000, 4000))
                elif test_set == 84:
                    test_dir = [38]
                    outlier_list = list(range(2000))
                elif test_set == 85:
                    test_dir = [9, 6, 38]
                
                elif test_set == 61:
                    test_dir = [42]
                    outlier_list = list(range(1000,3001))
                elif test_set == 62:
                    test_dir = [42]
                    outlier_list = list(range(1000))
                    outlier_list += list(range(2000,3001))
                elif test_set == 63:
                    test_dir = [42]
                    outlier_list = list(range(1900-self.stack_avg-10))
                    outlier_list += list(range(2200, 2300-self.stack_avg-10))
                    outlier_list += list(range(2500,3001))
                    #outlier_list = list(range(2000))
                    #outlier_list += list(range(2600,3001))
                elif test_set == 64:
                    test_dir = [42]
                    outlier_list += list(range(2400-self.stack_avg-10))
                elif test_set == 65:
                    test_dir = [42]
    
                elif test_set == 66:
                    test_dir = [41]
                    outlier_list = []
                    outlier_list += list(range(200, 500-self.stack_avg))
                    outlier_list += list(range(600, 800-self.stack_avg))
                    outlier_list += list(range(1000,3001))
                    
                elif test_set == 67:
                    test_dir = [41]
                    outlier_list = list(range(1000))
                    outlier_list += list(range(2000,3001))
                elif test_set == 68:
                    test_dir = [41]
                    outlier_list = list(range(2000))
                    outlier_list += list(range(2600,3001))
                elif test_set == 69:
                    test_dir = [41]
                    outlier_list = list(range(2600))
                elif test_set == 70:
                    test_dir = [41]
                    outlier_list = []
                    outlier_list += list(range(200, 500-self.stack_avg))
                    outlier_list += list(range(600, 800-self.stack_avg))
                
                elif test_set == 71:
                    test_dir = [37]
                elif test_set == 371:
                    test_dir = [37]
                    outlier_list = []
                    outlier_list += list(range(1000, 2000))
                elif test_set == 372:
                    test_dir = [37]
                    outlier_list = []
                    outlier_list += list(range(1000))
                
                elif test_set == 72:
                    test_dir = [39]
                elif test_set == 391:
                    test_dir = [39]
                    outlier_list = []
                    outlier_list += list(range(1000, 2000))
                elif test_set == 392:
                    test_dir = [39]
                    outlier_list = []
                    outlier_list += list(range(1000))
                
                elif test_set == 73:
                    test_dir = [44]
                elif test_set == 441:
                    test_dir = [44]
                    outlier_list = []
                    outlier_list += list(range(620, 2000))
                elif test_set == 442:
                    test_dir = [44]
                    outlier_list = []
                    outlier_list += list(range(620))
                    outlier_list += list(range(1080, 2000))
                elif test_set == 443:
                    test_dir = [44]
                    outlier_list = []
                    outlier_list += list(range(1080))
                    outlier_list += list(range(1680, 2000))
                elif test_set == 444:
                    test_dir = [44]
                    outlier_list = []
                    outlier_list += list(range(1680))
                
                elif test_set ==74:
                    test_dir = [45]
                elif test_set == 451:
                    test_dir = [45]
                    outlier_list = []
                    outlier_list += list(range(600, 2000))
                elif test_set == 452:
                    test_dir = [45]
                    outlier_list = []
                    outlier_list += list(range(600))
                    outlier_list += list(range(1200, 2000))
                elif test_set == 453:
                    test_dir = [45]
                    outlier_list = []
                    outlier_list += list(range(1200))
                    outlier_list += list(range(1540, 2000))
                elif test_set == 454:
                    test_dir = [45]
                    outlier_list = []
                    outlier_list += list(range(1540))
                
                elif test_set == 47:
                    test_dir = [47]
                    outlier_list = []
                elif test_set == 471:
                    test_dir = [47]
                    outlier_list = []
                
                elif test_set == 102:
                    test_dir = [48]
                    outlier_list = list(range(300-self.stack_avg-10))
                    outlier_list += list(range(600, 700-self.stack_avg-10))
                    outlier_list += list(range(1000,2001))
                    #outlier_list = list(range(700,2001))
                elif test_set == 101:
                    test_dir = [48]
                    outlier_list = list(range(1000-self.stack_avg-10))
                    outlier_list += list(range(1200, 1300-self.stack_avg-10))
                    outlier_list += list(range(1500,2001))
                    #outlier_list = list(range(700))
                    #outlier_list += list(range(1400,2001))
                elif test_set == 103:
                    test_dir = [48]
                    outlier_list = list(range(1500-self.stack_avg-10))
                    #outlier_list = list(range(1400))
                elif test_set == 104:
                    test_dir = [51]
                elif test_set == 105:
                    test_dir = [48, 51]
                    #test_dir = [48, 49, 50, 51]
                    #outlier_list = list(range(300-self.stack_avg-10))
                    #outlier_list += list(range(1000, 1100-self.stack_avg-10))
                    #outlier_list += list(range(1200, 1300-self.stack_avg-10))
                
                elif test_set == 110:
                    test_dir = [49]
                elif test_set == 111:
                    test_dir = [49]
                    outlier_list = list(range(400, 1000))
                elif test_set == 112:
                    test_dir = [49]
                    outlier_list = list(range(400))
                    outlier_list += list(range(750, 1000))
                elif test_set == 113:
                    test_dir = [49]
                    outlier_list = list(range(750))
                    
                elif test_set == 120:
                    test_dir = [50]
                elif test_set == 121:
                    test_dir = [50]
                    outlier_list = list(range(400, 1000))
                elif test_set == 122:
                    test_dir = [50]
                    outlier_list = list(range(400))
                    outlier_list += list(range(800, 1000))
                elif test_set == 123:
                    test_dir = [50]
                    outlier_list = list(range(800))
                
                if args.TT or args.pretrain_model : #domain change
                    test_dir = [41]
                    
            else :
                test_num = 3
                if test_num == 1 :
                    test_dir = [1]
                if test_num == 2 :
                    test_dir = [13]
                if test_num == 3 :
                    test_dir = [9]
                if test_num == 4 :
                    test_dir = [18]

            print("test_dir = ", test_dir)
        
        print("outlier_list = ", len(outlier_list))

        dir_count = 0  # dicrectory index

        rf_index = -1  # rf data index
        target_index = -1
        rf_frame = -1 # index that saved in raw_list

        mask_index = -1
        hm_index = -1
        cd_index = -1
        bbox_index = -1

        img_index = -1
        filename_index = -1

        #frame_stack = deque(maxlen=self.stack_avg*self.frame_skip)
        not_stacked_list = []
        for file in data_path_list:
            if dir_count in remove_dir:
                dir_count += 1
                continue

            if mode == 'train' and dir_count not in train_dir:
                dir_count += 1
                continue
            elif mode == 'test' and dir_count not in test_dir:
                dir_count += 1
                continue

            if os.path.isdir(file) is True:
                # 각 폴더 안의 npy 데이터
                if data_path == '/data/nlos/save_data_ver6' :
                    rf_file_list = glob.glob(file + '/radar/*.npy')
                else :
                    rf_file_list = glob.glob(file + '/RADAR/*.npy')
                rf_file_list = sorted(rf_file_list)
                print('\n\n\tdir_count:', dir_count,'dir(raw):', file)
                print('\t# of data :', len(rf_file_list), "rf_idx =", rf_index)
                if dir_count in three_dir:
                    print("\t************")
                dir_rf_index = -1
                mean_rf_list = deque(maxlen=self.store_frames)
                hilbert_stacks = deque(maxlen=self.store_frames)
                #process_mode = 'and'#'mean'
                
                for (idx_by_folder, rf) in enumerate(rf_file_list):
                    rf_index += 1
                    if dir_count in three_dir:
                        three_list.append(rf_index)
                        
                    if rf_index in outlier_list:
                        if mode=='train' and rf_index in train_outlier_idx:
                            print("train_outlier_idx = ", rf_index, rf)
                        
                        if mode=='test' and rf_index in test_outlier_idx:
                            print("test_outlier_idx = ", rf_index, rf)
                        mean_rf_list.clear()
                        dir_rf_index = -1
                        continue
                   
                    dir_rf_index += 1
                    raw_rf_load = np.load(rf)
                    
                    if data_path == '/data/nlos/save_data_ver6' :
                        if dir_count in [41, 42]:
                            #print(raw_rf_load.shape)
                            raw_rf_load[(6, 7), :, :] = raw_rf_load[(7, 6), :, :]
                            raw_rf_load[:, (6, 7), :] = raw_rf_load[:, (7, 6), :]
                    if data_path == '/data/nlos/save_data_ver6' :
                        temp_raw_rf = raw_rf_load[:, :, self.cutoff:]
                        #print(temp_raw_rf.shape)
                    else :
                        temp_raw_rf = raw_rf_load[:, :, self.cutoff:-self.cutoff]
                                                   
                    temp_raw_rf = torch.tensor(temp_raw_rf).float()
                    temp_raw_rf = rearrange(temp_raw_rf, 'tx rx len -> (tx rx) len')
                    #temp_raw_rf = torch.flatten(temp_raw_rf, 0, 1)
                    
                    
                    if len(mean_rf_list) >= self.store_frames :
                    #if len(mean_rf_list) >= self.store_frames*self.frame_skip :
                        mean_mode = 'normal'
                        
                        mean_rf = get_mean_rf(mean_rf_list, self.stack_avg)
                        mean_rf = mean_rf.float()
                        #mean_rf = mean_rf[:,:, self.cutoff:]
                        #mean_rf = torch.flatten(mean_rf, 0, 1)
                        #subtract_rf = make_hilbert_signal(temp_raw_rf.clone()) - make_hilbert_signal(mean_rf.clone())
                        subtract_rf = temp_raw_rf.clone() - mean_rf.clone()
                        
                        
                        rf_data.append(subtract_rf)
                        if idx_by_folder >= self.store_frames*self.frame_skip + (self.slow_time-1) :
                            train_idx_list.append(len(rf_data)-1)
                        
                        if dir_rf_index == 1000:
                            print("dir_rf_index {} max, min, mean(raw) = ".format(dir_rf_index), np.max(raw_rf_load), np.min(raw_rf_load), np.mean(raw_rf_load))
                            print("dir_rf_index {} max, min, mean(input) = ".format(dir_rf_index), torch.max(subtract_rf), torch.min(subtract_rf), torch.mean(subtract_rf), subtract_rf.shape)
                        rf_frame +=1
                    
                    mean_rf_list.append(temp_raw_rf)
                    #print(len(mean_rf_list))
                    #sum_rf_list = get_mean_rf(temp_raw_rf, sum_rf_list, temp_idx = idx_by_folder, avg_size = self.stack_avg, jump_size = self.frame_skip, mode = 'continuous')
                    #frame_stack.append(rf_frame)

                    
                    if idx_by_folder>6000 and idx_by_folder % (self.stack_avg*(self.frame_skip)*10) == 0 :
                            print('mean : ', torch.min(mean_rf), torch.max(mean_rf))
                            print("rf : ", torch.min(temp_raw_rf), torch.max(temp_raw_rf))
                            print("rf - mean : ", torch.min(subtract_rf), torch.max(subtract_rf))
                #print(not_stacked_stdev_list)


###############keypoint coordinate#########################
                if self.load_cd:
                    if data_path == '/data/nlos/save_data_ver6' :
                        if args.gt_head == 'normal' :
                            cd_file_list = glob.glob(file + '/HEATMAP_COOR_mpii/*.npy')
                        if args.gt_head == 'half' :
                            cd_file_list = glob.glob(file + '/coord_mpii/*.npy') 
                        if args.gt_head == 'coco' :
                            cd_file_list = glob.glob(file + '/PETR_GT/*.npy')
                            mask_file_list = glob.glob(file + '/mask/*.npy')
                            mask_file_list = sorted(mask_file_list)
                    else :
                        if args.gt_head == 'normal' :
                            cd_file_list = glob.glob(file + '/HEATMAP_COOR_mpii/*.npy')
                        if args.gt_head == 'half' :
                            cd_file_list = glob.glob(file + '/coord_mpii/*.npy') 
                        if args.gt_head == 'coco' :
                            cd_file_list = glob.glob(file + '/PETR_GT/*.npy') 
                            mask_file_list = glob.glob(file + '/seg_sum/*.npy')
                            mask_file_list = sorted(mask_file_list)
                        
                    
                    cd_file_list = sorted(cd_file_list)
                    print('\n\tdir(pose_cd):', file, '\t# of data :', len(cd_file_list))
                    
                    if args.gt_head == 'coco' :#coco
                        for idx_by_folder, (cd, seg_sum) in enumerate(zip(cd_file_list, mask_file_list)):
                            cd_index += 1
                            if cd_index in outlier_list or cd_index in not_stacked_list:
                                mean_cd_list.clear()
                                continue
                            np_cd = np.load(cd)
                            np_seg = np.load(seg_sum)
                            #print(np_seg)
                            num_people = np_cd.shape[0]
                            if num_people == 0 :
                                print(cd, np_cd)
                            
                            #make segment value
                            seg_numpy_init = np.zeros(num_people)
                            if num_people != 0 :
                                seg_numpy_init[:] = np_seg/num_people
                            #print(np_cd.shape[0], seg_numpy_init.shape[0])
                            
                            #make target
                            target_numpy = np.ones(num_people)#사람 수만큼 1 있는 np
                            
                            if len(mean_cd_list) < self.store_frames*self.frame_skip :
                                mean_cd_list.append(cd)
                                
                                continue     
                            cd_list.append(np_cd)
                            mask_list.append(seg_numpy_init)
                            target_list.append(target_numpy)
                            #cd_list.append(visible_nodes)
    
                            if cd_index %10000 == 1000:
                                print("cd_shape ", cd, np_cd.shape)#, np_cd[0][0]) #, visible_nodes[0])
                                        
                    else : #not coco
                        mean_cd_list = deque(maxlen=self.store_frames)
                        for idx_by_folder, cd in enumerate(cd_file_list):
                            cd_index += 1
                            if cd_index in outlier_list or cd_index in not_stacked_list:
                                mean_cd_list.clear()
                                continue
                            if len(mean_cd_list) < self.store_frames*self.frame_skip :
                                mean_cd_list.append(cd) 
                                continue
                            np_cd = np.load(cd)
                            np_cd = np_cd[:,:,:2]
                            np_cd[:, :, 0] /= 640#704
                            np_cd[:, :, 1] /= 480#512
                            np_cd = np_cd.reshape(np_cd.shape[0], -1)
                            cd_list.append(np_cd)
                            
                            target_numpy = np.ones(np_cd.shape[0])
                            target_list.append(target_numpy)
                            num_people = np_cd.shape[0]
                            if num_people == 0 :
                                print(cd, np_cd)
    
                            if cd_index %10000 == 1000:
                                print("cd_shape ", cd, np_cd.shape)#, np_cd[0][0]) #, visible_nodes[0])
                                        
                            #cd_list.append(np_cd)

###############bbox coordinate#########################
                if self.load_cd:
                    if data_path == '/data/nlos/save_data_ver6' :
                        bbox_file_list = glob.glob(file + '/box_people/*.npy') 
                    mean_cd_list = deque(maxlen=self.store_frames)
                    bbox_file_list = sorted(bbox_file_list)
                    print('\n\tdir(bbox_cd):', file, '\t# of data :', len(bbox_file_list))
                            
                    for idx_by_folder, cd in enumerate(bbox_file_list):
                        bbox_index += 1
                        if bbox_index in outlier_list or bbox_index in not_stacked_list:
                            mean_cd_list.clear()
                            continue
                        if len(mean_cd_list) < self.store_frames*self.frame_skip :
                            mean_cd_list.append(cd)
                            continue
                        np_cd = np.load(cd)
                        #print(np_cd[:,:4])
                                                        
                        bbox_list.append(np_cd[:, :4])

                        if bbox_index %10000 == 1000:
                            print("bbox_shape ", cd, np_cd.shape)#, np_cd[0][0]) #, visible_nodes[0])
                   
############### img ########################
                if self.load_img:
                    if data_path == '/data/nlos/save_data_ver6' :
                        img_file_list = glob.glob(file + '/image/*.jpg')
                    else :
                        img_file_list = glob.glob(file + '/IMAGE/*.jpg')
                    img_file_list = sorted(img_file_list)
                    mean_file_list = deque(maxlen=self.store_frames)
                    print('\n\tdir(img):', file, '\t# of data :', len(img_file_list))

                    for idx_by_folder, img in enumerate(img_file_list):
                        img_index += 1
                        filename_index += 1 
                        f_name = '{}/IMAGE/{}.npy'.format(file, img.split('/')[-1].split('.')[0])
                        if img_index in outlier_list or img_index in not_stacked_list:
                            mean_file_list.clear()
                            continue
                        
                        #if idx_by_folder in outlier_list or idx_by_folder - (self.store_frames*self.frame_skip) < 0:
                        if len(mean_file_list)< self.store_frames*self.frame_skip :
                            mean_file_list.append(f_name)
                            continue   
                        #temp_img = cv2.imread(img)
                        #img_list.append(temp_img)
                    
                        #print(f_name)
                        
                        filename_list.append(f_name)
                        img_list.append(img)

                        if img_index %10000 == 1000:
                            print(f"img_index {img_index} img_shape {cv2.imread(img).shape}")

            dir_count += 1

        
        #print("not_stacked_list {}: {} ~ {}".format(len(not_stacked_list), not_stacked_list[0], not_stacked_list[-1]))
        #print(not_stacked_stdev_list)
        self.train_idx_list = train_idx_list
        self.rf_data = rf_data
        self.raw_list = raw_list
        self.cd_list = cd_list
        self.bbox_list = bbox_list
        self.filename_list = filename_list
        self.img_list = img_list
        self.target_list = target_list
        self.three_list = three_list
        
        #self.human_index = human_index
        #print(f"rf\t{len(rf_data)}/{outlier_by_ewma}\t raw\t{len(raw_list)}/{rf_frame}\t target\t{len(target_list)}")
        #print(f"3~4 list\t{len(three_list)}" )
        print("train idxes len : ", len(self.train_idx_list))
        print("rf data len : ", len(self.rf_data))
        print("pose data len : ", len(self.cd_list))
        print("bbox data len : ", len(self.bbox_list))
        print("img data len : ", len(self.img_list))

        self.three_len = len(self.three_list)        
        print("end - data read")
        print("size of dataset", len(self.rf_data))

    def __len__(self):    
        return len(self.train_idx_list)

    def __getitem__(self, idx):
        
        if random.random() < self.three and self.mode == 'train' and self.three_len>0:
            idx = self.three_list[idx%len(self.three_list)]

        f_name = None
        
        idx = self.train_idx_list[idx] #get nex idx from idx_train_list

        slow_time_stack_rf = []
        for temp_time in range(self.slow_time) :
            slow_time_stack_rf.append(self.get_rf(idx-((self.slow_time -1) - temp_time)))
        rf = torch.stack(slow_time_stack_rf).permute(1,0,2) #ch, slowtime, fasttime
        #print(rf.shape)
        
        if self.load_cd :
            cd = []
            for i in range(self.slow_time) :
                tmp_cd = self.cd_list[idx - (self.slow_time - (1 + i))]
                tmp_cd = torch.FloatTensor(tmp_cd).clone()
                cd.append(tmp_cd)
            
        else :
            cd = None
        
        if self.load_cd :
            bbox = []
            for i in range(self.slow_time) :
                tmp_bbox = self.bbox_list[idx - (self.slow_time - (1 + i))]
                tmp_bbox = torch.FloatTensor(tmp_bbox).clone()
                bbox.append(tmp_bbox)
            
        else :
            bbox = None
        
        if self.load_img :
            img = self.img_list[idx]
            img = cv2.imread(img)
            f_name = self.filename_list[idx]
        else :
            img = None
            
        if self.load_cd :
            target = []
            for i in range(self.slow_time) :
                tmp_target = self.target_list[idx - (self.slow_time - (1 + i))]
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


