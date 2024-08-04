import re
import matplotlib.pyplot as plt
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from collections import Counter
import torch
from PIL import Image
#from util.misc import nested_tensor_from_tensor_list
#from util import box_ops
#from models.rftr_pose import get_max_preds
from visualize import imshow_keypoints
from einops import rearrange, repeat
import copy

COLORS = ((244,  67,  54),
          (233,  30,  99),
          (156,  39, 176),
          (103,  58, 183),
          ( 63,  81, 181),
          ( 33, 150, 243),
          (  3, 169, 244),
          (  0, 188, 212),
          (  0, 150, 136),
          ( 76, 175,  80),
          (139, 195,  74),
          (205, 220,  57),
          (255, 235,  59),
          (255, 193,   7),
          (255, 152,   0),
          (255,  87,  34),
          (121,  85,  72),
          (158, 158, 158),
          ( 96, 125, 139))

COCO_CLASSES = ('person')

palette = np.array([[255, 128, 0], 
                        [255, 153, 51], [255, 178, 102], [230, 230, 0], 
                        [255, 153, 255], [153, 204, 255], [255, 102, 255], 
                        [255, 51, 255], 
                        [102, 178, 255],
                        [51, 153, 255], 
                        [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255],
                        [255, 0, 0], [255, 255, 255]])


coco_skeleton = [[11, 9], [9, 7], [12, 10], [10, 8], 
            [7, 8], [1, 7], [2, 8], 
            [1, 2], [1, 3], [2, 4], [3, 5], [4,6]]

'''mpii_skeleton = [[8,9], 
            [11,12], [11, 10], [2, 1], [1, 0],
            [13, 14], [14, 15], [3, 4], [4, 5],
            [8, 7], [7, 6], [6, 2], [6, 3], [8, 12], [8,13]]
            #[15,16]]'''
mpii_skeleton = [[8,9], 
            [11,12], [11, 10], [2, 1], [1, 0],
            [13, 14], [14, 15], [3, 4], [4, 5],
             [12, 2], [13, 3], [6, 2], [6, 3], [8, 12], [8,13]]

# 0 주황 7 핑크 9 blue 16 green
'''mpii_pose_link_color = palette[[ 9, 
                            0, 0, 0, 0,
                            7, 7, 7, 7, 
                            9, 9, 0, 7, 0, 7,
]]'''
mpii_pose_link_color = palette[[ 9, 
                            0, 0, 7, 7,
                            0, 0, 7, 7, 
                            0, 0, 7, 7, 0, 0,
]]

'''mpii_pose_kpt_color = palette[[
    0, 0, 0, 7, 7,
    7, 9, 9, 9, 9, 
    0, 0, 0, 7, 7, 7
]]'''
mpii_pose_kpt_color = palette[[
    7, 7, 7, 7, 7,
    7, 7, 0, 0, 9, 
    0, 0, 0, 0, 0, 0
]]

'''coco_pose_kpt_color = [(236, 6, 124), (236, 6, 124), (236, 6, 124),
                         (236, 6, 124), (236, 6, 124), (169, 209, 142),
                         (255, 255, 0), (169, 209, 142), (255, 255, 0),
                         (169, 209, 142), (255, 255, 0), (0, 176, 240),
                         (252, 176, 243), (0, 176, 240), (252, 176, 243),
                         (0, 176, 240), (252, 176, 243)]'''

'''coco_edges = [[0, 1], [0, 2], [1, 3], [2, 4],  # head 
[5, 7], [7, 9], [6, 8], [8, 10],  # arms
[5, 6], [5, 11],[6, 12],  # body
[11, 13], [13, 15], [12, 14], [14, 16]]  # legs
coco_edge_color = [(236, 6, 124), (236, 6, 124), (236, 6, 124), (236, 6, 124), (169, 209, 142),\
                  (169, 209, 142), (255, 255, 0), (255, 255, 0), (255, 102, 0),\
                  (0, 176, 240), (252, 176, 243), (0, 176, 240), (0, 176, 240),\
                  (252, 176, 243), (252, 176, 243)]'''

def pckh(pred, target):
    keypoints = [ "rankle", "rknee", "rhips", "lhips", "lknee", "lankle", \
                'chip', 'cshld', 'neck', 'head', \
                 'rwrist', 'relbow', 'rshld', 'lshld', 'lelbow', 'lwrist'
    ]
    num_joint = pred.shape[0]
    true_detect = np.zeros((4, num_joint))
    whole_count = np.zeros((4, num_joint))
    thr = [0.1, 0.2, 0.3, 0.5]
    
    #print(pred, target)
    if target[8][2] >= 0.7 and target[9][2] >= 0.7: 
        head_size = np.linalg.norm(target[8][:2] - target[9][:2])
    else:
        return true_detect, whole_count

    for j in range(num_joint):
        if target[j][2] < 0.7: # invisible
            continue
        dist = np.linalg.norm(target[j][:2] - pred[j][:2])
        for t in range(len(thr)):
            whole_count[t][j] += 1
            if dist <= thr[t] * head_size:
                true_detect[t][j] += 1

    return true_detect, whole_count

def pckh_thr(pred, target, Thr):
    num_joint = pred.shape[0]
    true_detect = np.zeros(num_joint)
    whole_count = np.zeros(num_joint)

    
    #print(pred, target)
    if target[8][2] >= 0.7 and target[9][2] >= 0.7: 
        head_size = np.linalg.norm(target[2] - target[7])
    else:
        return true_detect, whole_count

    for j in range(num_joint):
        if target[j][2] < 0.7: # invisible
            continue
        dist = np.linalg.norm(target[j] - pred[j])

        whole_count[j] += 1
        if dist <= Thr * head_size:
            true_detect[j] += 1

    return true_detect, whole_count

def getArea(box):
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)


def getUnionAreas(boxA, boxB, interArea=None):
    area_A = getArea(boxA)
    area_B = getArea(boxB)
    
    if interArea is None:
        interArea = getIntersectionArea(boxA, boxB)
        
    return float(area_A + area_B - interArea)

def getIntersectionArea(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # intersection area
    return (xB - xA + 1) * (yB - yA + 1)

# boxA = (Ax1,Ay1,Ax2,Ay2)
# boxB = (Bx1,By1,Bx2,By2)
def boxesIntersect(boxA, boxB):
    if boxA[0] > boxB[2]:
        return False  # boxA is right of boxB
    if boxB[0] > boxA[2]:
        return False  # boxA is left of boxB
    if boxA[3] < boxB[1]:
        return False  # boxA is above boxB
    if boxA[1] > boxB[3]:
        return False  # boxA is below boxB
    return True

def iou(boxA, boxB):
    # if boxes dont intersect
    if boxesIntersect(boxA, boxB) is False:
        return 0
    interArea = getIntersectionArea(boxA, boxB)
    union = getUnionAreas(boxA, boxB, interArea=interArea)
    
    # intersection over union
    result = interArea / union
    #assert result >= 0, "{} {} {}".format(result, interArea, union)
    if result >= 0:
        return result
    else:
        return 0


def calculateAveragePrecision(rec, prec):
    
    mrec = [0] + [e for e in rec] + [1]
    mpre = [0] + [e for e in prec] + [0]
    #print(mrec, mpre)
    for i in range(len(mpre)-1, 0, -1):
        mpre[i-1] = max(mpre[i-1], mpre[i])

    ii = []

    for i in range(len(mrec)-1):
        if mrec[1:][i] != mrec[0:-1][i]:
            ii.append(i+1)

    ap = 0
    for i in ii:
        ap = ap + np.sum((mrec[i] - mrec[i-1]) * mpre[i])
    
    return [ap, mpre[0:len(mpre)-1], mrec[0:len(mpre)-1], ii]


def ElevenPointInterpolatedAP(rec, prec):

    mrec = [e for e in rec]
    mpre = [e for e in prec]

    recallValues = np.linspace(0, 1, 11)
    recallValues = list(recallValues[::-1])
    rhoInterp, recallValid = [], []

    for r in recallValues:
        argGreaterRecalls = np.argwhere(mrec[:] >= r)
        pmax = 0

        if argGreaterRecalls.size != 0:
            pmax = max(mpre[argGreaterRecalls.min():])

        recallValid.append(r)
        rhoInterp.append(pmax)

    ap = sum(rhoInterp) / 11

    return [ap, rhoInterp, recallValues, None]



def pose_AP_mpii(targets, results, imgs=None,  
        IOUThreshold = 0.5, method = 'AP', vis=False, 
        img_dir=None, boxThrs=0.5, 
        test_type='b', gt_head = 'normal', video_num = None, best_iter_ap = None, print_log = False):
    
    frame_array = []
    
    if img_dir is not None:
        os.makedirs(img_dir, exist_ok=True)
    #print(targets, results)
    detections, groundtruths, classes = [], [], []
    batch_size = len(targets)
    #print(batch_size)
    result = []
    
    # ground truth data
    for i in range(batch_size):
        # Ground Truth Data
        img_size = targets[i]['orig_size']
        ori_size = img_size.int().cpu().numpy()
        num_obj = targets[i]['labels'].shape[0]
        img_id = targets[i]['image_id']
            
        #bbox = targets[i]['boxes']
        #boxes = box_ops.box_cxcywh_to_xyxy(bbox)
        #mask_size = targets[i]['mask_size']

        poses = targets[i]['cd']
        #print(poses.shape)
        #print(poses)
        poses = poses.view(poses.shape[0], -1, 2).cpu()
        #result_window = np.array([ori_size[0], ori_size[1], 1])
        result_window = np.array([ori_size[0], ori_size[1]])
        gt_poses = np.copy(poses) * result_window
        gt_poses_score = np.ones((gt_poses.shape[0], gt_poses.shape[1], 1)) #score for plot
        gt_poses = np.concatenate((gt_poses,gt_poses_score), axis = 2)
        #print(gt_poses)

        img_h, img_w = img_size
        #boxes = torch.mul(boxes, img_h)

        if vis and imgs is not None:
            img = imgs[i]
            gt_img = cv2.resize(img, (ori_size[1], ori_size[0]), interpolation=cv2.INTER_AREA)
            pred_img = gt_img.copy()
            blank_img = np.zeros((ori_size[1], ori_size[0], 3), dtype = np.uint8)
            #print("blank_img ", blank_img.shape)
           
        gt_pose_to_draw = []
        for j in range(num_obj):
            label, conf = 0, 1
            #x1, y1, x2, y2 = boxes[j]
            gt_pose_to_draw.append(gt_poses[j])
            #print(gt_poses[j].shape)
            #box_info = [img_id, label, conf, (x1.item(), y1.item(), x2.item(), y2.item())]
            pose_info = [img_id, gt_poses[j]]#, None, (x1.item(), y1.item(), x2.item(), y2.item())]

            groundtruths.append(pose_info)
            if label not in classes:
                classes.append(label)
                
        #####gt pose 그리기#####
        if vis and i == 0: #첫 batch만 그리기
            imshow_keypoints(gt_img, gt_pose_to_draw, mpii_skeleton, kpt_score_thr=0, \
                     pose_kpt_color=mpii_pose_kpt_color, pose_link_color=mpii_pose_link_color, radius=4, thickness=2)

        # Prediction Data
        pred = results[i]
        pred_score = pred['scores']
        #print(pred_score)
        #pred_label = pred['labels']
        result_window = np.array([ori_size[0], ori_size[1]])
        pred_keypoint = np.copy(pred['keypoint'].cpu())*result_window
        #pred_keypoint_score = np.ones((pred_keypoint.shape[0], pred_keypoint.shape[1], 1)) #score for plot
        #pred_keypoint = np.concatenate((pred_keypoint,pred_keypoint_score), axis = 2)
        #pred_boxes = pred['boxes']

        #res_pose_ = res_pose[i]#.cpu().numpy()
        pose_to_draw = []
        num_queries = pred_keypoint.shape[0]
        for q in range(num_queries):
            label, conf = 0, pred_score[q].item()
            #x1, y1, x2, y2 = pred_boxes[q]
            if test_type =='x':
                #x1, y1, x2, y2 = x1+44, y1+20, x2+38, y2-10
                pred_keypoint[q][:,0] += 41 #41 ## x coor
                pred_keypoint[q][:,1] += 5 #5 ## y coor
            elif test_type =='y':
                #x1, y1, x2, y2 = x1+47, y1+5, x2+38, y2-5
                #x1, y1, x2, y2 = x1+49, y1, x2+40, y2-4
                pred_keypoint[q][:,0] += 45 #45
                pred_keypoint[q][:,1] -= 2 # -2
                
            #box_info = [img_id, label, conf, (x1.item(), y1.item(), x2.item(), y2.item())]
            pose_info = [img_id, pred_keypoint[q], conf]
            #print(res_pose_[q])
            if pred_score[q] > boxThrs:
                tmp_keypoint = pred_keypoint[q].copy()
                tmp_keypoint_score = np.ones((tmp_keypoint.shape[0], 1)) #score for plot
                tmp_keypoint = np.concatenate((tmp_keypoint,tmp_keypoint_score), axis = 1)
                pose_to_draw.append(tmp_keypoint)
                detections.append(pose_info.copy())
            if label not in classes:
                classes.append(label)
        
        if vis and i == 0:
            #imshow_keypoints(pred_img, pose_to_draw, skeleton, kpt_score_thr=0, \
            imshow_keypoints(blank_img, pose_to_draw, mpii_skeleton, kpt_score_thr=0, \
                     pose_kpt_color=mpii_pose_kpt_color, pose_link_color=mpii_pose_link_color, radius=4, thickness=2)
            res = np.concatenate((pred_img, blank_img, gt_img), axis=1)
            #video
            #video_res = np.concatenate((pred_img, blank_img), axis=1)
            #reswid, resheight, resch = video_res.shape
            #frame_array.append(video_res.copy())
            #cv2.imwrite(os.path.join(img_dir, str()) +'_{}_pose_num{}.png'.format(img_id, num_obj), video_res)
            cv2.imwrite(os.path.join(img_dir, str()) +'_{}_pose_num{}.png'.format(img_id, num_obj), res)
    
    #print(os.path.join(img_dir, str()) +'{}_pose.mp4v'.format(video_num))
    
    img_key = []
    kpt_ap = []
    '''
    for tmp_ in detections :
        if tmp_[0] not in img_key :
            img_key.append(tmp_[0])

    for tmp_img_key in img_key :
    
    dects = [dect for dect in detections if dect[0] == tmp_img_key] #detections
    gts = [gt for gt in groundtruths if gt[0] == tmp_img_key] #groundtruths
    '''
    dects = detections
    gts = groundtruths
    
    npos = len(gts)
    #print(npos)
    dects = sorted(dects, key = lambda conf : conf[2], reverse=True) #image 순서 뒤바뀜conf 순
    #print(dects, "\n")
    for kpt in range(16):

        TP = np.zeros(len(dects))
        FP = np.zeros(len(dects))

        det = Counter(cc[0] for cc in gts) #image id 에 gt object 몇개있는지 count

        for key, val in det.items(): #key = img id, val = num_people in key img
            det[key] = np.zeros(val)

        for d in range(len(dects)): #모든 유효한 사람들 len(dects)
            gt = []
            for tmp_gt in gts :
                if ((tmp_gt[0] == dects[d][0])) :
                    gt.append(tmp_gt)
            #gt = [gt for gt in gts if gt[0] == dects[d][0]] #img id가 같은 gt들 list화
            dist_min = 10000
            gt_head_size = 0  
            
            np_pred = dects[d][1] # keypoint
                
            for j in range(len(gt)): 
                np_gt = gt[j][1]
                #print(np_pred.shape, np_gt.shape)
                '''
                if np_gt[8][2] >= 0.5 and np_gt[9][2] >= 0.5: 
                    if gt_head == 'half' :
                        head_size = np.linalg.norm(np_gt[7][:2] - np_gt[9][:2]) * 2
                    else :
                        head_size = np.linalg.norm(np_gt[7][:2] - np_gt[9][:2])
                else:
                    continue
                '''
                head_size = np.linalg.norm(np_gt[7][:2] - np_gt[9][:2]) * 2
                #if np_gt[kpt][2] < 0.7: # invisible
                #    continue
                dist = np.linalg.norm(np_gt[kpt][:2] - np_pred[kpt][:2])

                if dist < dist_min:
                    dist_min = dist
                    gt_head_size = head_size 
                    jmax = j

            if dist_min <= IOUThreshold * gt_head_size:
                if det[dects[d][0]][jmax] == 0:
                    TP[d] = 1
                    det[dects[d][0]][jmax] = 1
                else:
                    FP[d] = 1
            else:
                FP[d] = 1
        
        acc_FP = np.cumsum(FP)
        acc_TP = np.cumsum(TP)
        #print(acc_TP, acc_FP)
        rec = acc_TP / npos
        '''
        if print_log :
          print(rec)
          print(acc_FP)
          '''
        prec = np.divide(acc_TP, (acc_FP + acc_TP))
        if method == "AP":
            [ap, mpre, mrec, ii] = calculateAveragePrecision(rec, prec)
        else:
            [ap, mpre, mrec, _] = ElevenPointInterpolatedAP(rec, prec)
        #print(ap)
        kpt_ap.append(ap)
        #print(ap)
        
        #ap_mean = np.mean(np.array(ap))
        #if ap_mean > best_iter_ap :
        #    out = cv2.VideoWriter(os.path.join(img_dir, str()) +'best_pose.avi'.format(), cv2.VideoWriter_fourcc(*'MJPG'), 3, (resheight, reswid))
        
        #    for i in range(len(frame_array)):
                # writing to a image array
                #cv2.imshow('Frame', frame_array[i])
        #        out.write(frame_array[i])
        #    out.release()
    
    #print(kpt_ap)
    r = {
        'class' : 0,
        'AP' : kpt_ap,
    }
    #print(r)
    result.append(r)

    return result

def oks_iou(gt, pred, gt_mask_size, window_size,sigmas=None, in_vis_thre=None) :
    if not isinstance(sigmas, np.ndarray):
        sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
        equal_sigmas = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) / 17.0
    
    vars = (2*sigmas) ** 2
    equal_vars = (2*equal_sigmas) ** 2
    k = len(sigmas)
    gt_mask_size = gt_mask_size*window_size[0]*window_size[1]
    #gt = gt/window_size
    #pred = pred/window_size
    xg = gt[:, 0]; yg = gt[:, 1]; 
    #vg = gt[:,2]
    xd = pred[:, 0]; yd = pred[:, 1]
    #print(xg, xd)
    #print(yg)
    #print(yd)
    #print(xg, xd)
    #k1 = np.count_nonzero(vg > 0.5) #sigma로 대체
    k1 = np.count_nonzero(sigmas) #sigma로 대체
    
    #bb = gt_box
    #w, h = bb[2] - bb[0], bb[3] - bb[1]
    #x0 = bb[0] - w; x1 = bb[0] + w*2
    #y0 = bb[1] - h; y1 = bb[1] + h*2
    
    dx = xd - xg
    dy = yd - yg
    

    #print(gt_mask_size, gt_mask_size + np.spacing(1))
    e = (dx ** 2 + dy ** 2) / vars / (gt_mask_size + np.spacing(1)) / 2
    equal_e = (dx ** 2 + dy ** 2) / equal_vars / (gt_mask_size + np.spacing(1)) / 2
    
    keypoint_ious = np.exp(-e)
    equal_keypoint_ious = np.exp(-equal_e)
    #keypoints_ious = np.where(k1>0,keypoint_ious, 0)

    ious = np.sum(keypoint_ious) / e.shape[0] if e.shape[0] != 0 else 0.0
    #print(ious)
    return ious, equal_keypoint_ious

# Cacluate Average Precision (AP) and visualize results
def pose_AP(targets, results, imgs=None,  
        IOUThreshold = 0.5, method = 'AP', vis=False, 
        img_dir=None, boxThrs=0.5, dr_size=256, pose_method='simdr', 
        test_type='b'):
    
    if img_dir is not None:
        os.makedirs(img_dir, exist_ok=True)
    #print(targets, results)
    detections, groundtruths, classes = [], [], []
    batch_size = len(targets)
    result = []
    
    # ground truth data
    for i in range(batch_size):
        # Ground Truth Data
        img_size = targets[i]['orig_size']
        ori_size = img_size.int().cpu().numpy()
        num_obj = targets[i]['labels'].shape[0]
        img_id = targets[i]['image_id']
            
        #bbox = targets[i]['boxes']
        #boxes = box_ops.box_cxcywh_to_xyxy(bbox)
        gt_mask_size = targets[i]['mask_size']
        '''
        if pose_method == 'hm':
            poses = targets[i]['hm']
            pose_hm = poses.cpu().numpy()
            #print(pose_hm.shape)
            num_people = pose_hm.shape[0]
            num_joints =  pose_hm.shape[1]
            preds, maxval = get_max_preds(pose_hm)
            gt_poses = np.ones([num_people, num_joints, 3])

            gt_poses[:, :, 0] = preds[:, :, 0] * 8
            gt_poses[:, :, 1] = preds[:, :, 1] * 8
            gt_poses[:, :, 2] = maxval[:, :, 0]
        else:
        '''
        poses = targets[i]['cd']
        poses = poses.view(poses.shape[0], -1, 3).cpu()
        result_window = np.array([ori_size[0], ori_size[1], 1])
        gt_poses = np.copy(poses) * result_window

        img_h, img_w = img_size
        #boxes = torch.mul(boxes, img_h)

        if vis and imgs is not None:
            img = imgs[i]
            gt_img = cv2.resize(img, (ori_size[1], ori_size[0]), interpolation=cv2.INTER_AREA)
            pred_img = gt_img.copy()
            blank_img = np.zeros((ori_size[1], ori_size[0], 3), dtype = np.uint8)
            #print("blank_img ", blank_img.shape)
           
        gt_pose_to_draw = []
        for j in range(num_obj):
            #label, conf = targets[i]['labels'][j].item(), 1
            label, conf = 0, 1
            #x1, y1, x2, y2 = boxes[j]
            
            gt_pose_to_draw.append(gt_poses[j])
            #print(gt_poses[j].shape)
            
            #x1, y1, x2, y2 = box_ops.sanitize_all_coordinates(targets[i]['boxes'][j][0], targets[i]['boxes'][j][1], targets[i]['boxes'][j][2], targets[i]['boxes'][j][3], img_size[0])
            #box_info = [img_id, label, conf, (x1.item(), y1.item(), x2.item(), y2.item())]
            #print(img_id, gt_poses.shape, gt_mask_size.shape)
            pose_info = [img_id, gt_poses[j], gt_mask_size[j]]#, (x1.item(), y1.item(), x2.item(), y2.item())]

            groundtruths.append(pose_info)
            if label not in classes:
                classes.append(label)
        #gt batch중 처음만 그리기
        if vis and i == 0:
            imshow_keypoints(gt_img, gt_pose_to_draw, coco_edges, kpt_score_thr=0, \
                     pose_kpt_color=coco_pose_kpt_color, pose_link_color=coco_edge_color, radius=4, thickness=2)
            #print(gt_poses)

        # Prediction Data
        pred = results[i]
        pred_score = pred['scores']
        #pred_label = pred['labels']
        result_window = np.array([ori_size[0], ori_size[1]])
        pred_keypoint = np.copy(pred['keypoint'].cpu())*result_window
        pred_kp_score = np.ones((pred_keypoint.shape[0], pred_keypoint.shape[1], 1))
        pred_keypoint = np.append(pred_keypoint, pred_kp_score, axis = 2)
        #pred_indices = pred['indices']

        #res_pose_ = res_pose[i]#.cpu().numpy()
        pose_to_draw = []
        num_queries = pred_keypoint.shape[0]
        for q in range(num_queries):
            label, conf = 0, pred_score[q].item()
            #x1, y1, x2, y2 = pred_boxes[q]
            
            #index = pred_indices[q].int().item()
            #print(res_pose_[index].shape)
            if test_type =='x':
                #x1, y1, x2, y2 = x1+44, y1+20, x2+38, y2-10
                pred_keypoint[q][:,0] += 41 ## x coor
                pred_keypoint[q][:,1] += 5 ## y coor
            elif test_type =='y':
                ##x1, y1, x2, y2 = x1+47, y1+5, x2+38, y2-5
                #x1, y1, x2, y2 = x1+49, y1, x2+40, y2-4
                pred_keypoint[q][:,0] += 45
                pred_keypoint[q][:,1] -= 2
                

            #box_info = [img_id, label, conf, (x1.item(), y1.item(), x2.item(), y2.item())]
            pose_info = [img_id, pred_keypoint[q], conf]
            #print(res_pose_[q])
            if pred_score[q] > boxThrs:
                pose_to_draw.append(pred_keypoint[q])
                detections.append(pose_info)
            if label not in classes:
                classes.append(label)
                    
        if vis and i == 0:
            #print(pred_keypoint)
            #imshow_keypoints(pred_img, pose_to_draw, skeleton, kpt_score_thr=0, \
            imshow_keypoints(blank_img, pose_to_draw, coco_edges, kpt_score_thr=0, \
                     pose_kpt_color=coco_pose_kpt_color, pose_link_color=coco_edge_color, radius=4, thickness=2)
            res = np.concatenate((pred_img, blank_img, gt_img), axis=1)
            cv2.imwrite(os.path.join(img_dir, str()) +'_{}_pose_num{}.png'.format(img_id, num_obj), res)

    dects = detections
    gts = groundtruths
    
    npos = len(gts)
    dects = sorted(dects, key = lambda dects : dects[2], reverse=True) #dects[2] = conf
    
    
    TP = np.zeros(len(dects))
    FP = np.zeros(len(dects))

    det = Counter(cc[0] for cc in gts)

    for key, val in det.items(): # key : img id
        det[key] = np.zeros(val)

    #print(len(dects))
    avg_iou = 0
    avg_kpiou = np.zeros((17))
    for d in range(len(dects)):
        gt = [gt for gt in gts if gt[0] == dects[d][0]]
        iouMax = 0
        kpiouMax = 0 #None

        for j in range(len(gt)):
            np_pred = dects[d][1]
            np_gt = gt[j][1]
            gt_mask_size = gt[j][2]
            #print(np_gt.shape)
            #print(np_pred.shape)
            #gt_box = gt[j][3]
            
            iou1, kp_iou = oks_iou(np_gt, np_pred, gt_mask_size, np.array([ori_size[0], ori_size[1], 1]))
            #iou1, kp_iou = oks_iou(np_gt, np_gt, gt_mask_size, np.array([ori_size[0], ori_size[1], 1]))
            
            #iou1 = iou(dects[d][3], gt[j][3])
            if iou1 > iouMax:
                #print(iou1)
                iouMax = iou1
                kpiouMax = kp_iou
                jmax = j

        avg_iou += iouMax
        #print(kpiouMax)
        avg_kpiou += kpiouMax

        if iouMax >= IOUThreshold:
            if det[dects[d][0]][jmax] == 0:
                TP[d] = 1
                det[dects[d][0]][jmax] = 1
            else:
                FP[d] = 1
        else:
            FP[d] = 1
    
    acc_FP = np.cumsum(FP)
    acc_TP = np.cumsum(TP)
    #print(acc_TP, acc_FP)
    rec = acc_TP / npos
    prec = np.divide(acc_TP, (acc_FP + acc_TP))
    if method == "AP":
        [ap, mpre, mrec, ii] = calculateAveragePrecision(rec, prec)
    else:
        [ap, mpre, mrec, _] = ElevenPointInterpolatedAP(rec, prec)
        
        
    if len(dects) > 0 :
        r = {
            'class' : 0,
            'precision' : prec,
            'recall' : rec,
            'AP' : ap,
            'iou' : avg_iou / len(dects),
            'kpiou' : avg_kpiou / len(dects),
            'interpolated precision' : mpre,
            'interpolated recall' : mrec,
            'total positives' : npos,
            'total TP' : np.sum(TP),
            'total FP' : np.sum(FP)
        }
        #print(r)
        result.append(r)

    return result

# 0 : rank 1: rknee 2: rhips
# 3 : lhips 4: knee 5: lank
# 6: chip 7: chest 8: neck 9: head
# 10 : rwrist 11: relbow 12: rshld
# 13 : lshd 14: lelbow 15: lwrist

def mPCK(result):
    true_detect = np.zeros((4,16))
    whole_count = np.zeros((4,16))
    for r in result:
        true_detect += r['true_detect']
        whole_count += r['whole_count']
    
    pck_res = true_detect / whole_count * 100
    
    kpt_result = np.zeros((4, 9))
    
    # head, neck, shld, elbow, wrist, 
    # hip, knee, ankle, total
    kpt_result[:, 0] = pck_res[:, 9] # HEAD
    kpt_result[:, 1] = pck_res[:, 8] # NECK
    #kpt_result[:, 2] = (pck_res[:, 12] + pck_res[:, 13]) /2  # SHO
    kpt_result[:, 2] = (pck_res[:, 12] + pck_res[:, 7] + pck_res[:, 13]) / 3  # SHO
    kpt_result[:, 3] = (pck_res[:, 11] + pck_res[:, 14]) /2 # ELB
    kpt_result[:, 4] = (pck_res[:, 10] + pck_res[:, 15]) /2 # WRI
    #kpt_result[:, 5] = (pck_res[:, 2] + pck_res[:, 3]) /2 # HIP
    kpt_result[:, 5] = (pck_res[:, 2] + pck_res[:, 3]+ pck_res[:, 6]) /3 # HIP
    kpt_result[:, 6] = (pck_res[:, 1] + pck_res[:, 4]) /2 # KNE
    kpt_result[:, 7] = (pck_res[:, 0] + pck_res[:, 5]) /2 # ANK
    kpt_result[:, 8] = np.average(pck_res[:, :8], axis=1) # TOT

    #print(kpt_result)
    return pck_res, kpt_result

def mAP(result):
    ap = 0
    for r in result:
        ap += r['AP']

    #print("mAP len(result) = ", len(result))
    mAP = ap / len(result) if len(result) != 0 else 0.
    return mAP

def pck_mAP(result):
    #print(result[-1])
    kpt_result = np.zeros(9)
    ap = np.zeros(16)
    #ap = [0.] * 16
    for r in result:
        ap += np.array(r['AP'])
        #for k in range(16): #k for keypoints
        #    ap[k] += r['AP'][k]
    
    
    if len(result) != 0:
        #for k in range(16):
        #    ap[k] = ap[k] / len(result)
        ap = ap / len(result)
        
    print(ap)
    # head, neck, shld, elbow, wrist, 
    # hip, knee, ankle, total
    kpt_result[0] = ap[9] # HEAD
    kpt_result[1] = ap[8] # NECK
    #kpt_result[2] = (ap[12] + ap[13]) /2  # SHO
    kpt_result[2] = (ap[12] + ap[7] + ap[13]) / 3  # SHO
    kpt_result[3] = (ap[11] + ap[14]) /2 # ELB
    kpt_result[4] = (ap[10] + ap[15]) /2 # WRI
    #kpt_result[5] = (ap[2] + ap[3]) /2 # HIP
    kpt_result[5] = (ap[2] + ap[3]+ ap[6]) /3 # HIP
    kpt_result[6] = (ap[1] + ap[4]) /2 # KNE
    kpt_result[7] = (ap[0] + ap[5]) /2 # ANK
    kpt_result[8] = np.average(kpt_result[:8]) # TOT
    #print(ap)
    return kpt_result #ap

def mIOU(result):
    iou = 0
    for r in result:
        iou += r['iou']

    mIOU = iou / len(result) if len(result) != 0 else 0.
    return mIOU

def mkpIOU(result):
    iou = np.zeros((17))
    for r in result:
        iou += r['kpiou']

    mIOU = iou / len(result) if len(result) != 0 else 0.
    return mIOU

def class_ap(result, c):
    ap = 0
    k = 0
    for r in result:
        if r['class'] == c:
            ap += r['AP']
            k +=1
    
    if k == 0:
        mAP = 0
    else:
        mAP = ap / k

    return mAP


