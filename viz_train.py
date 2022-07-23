import os
import sys
import time
import logging
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

import cv2
import torch
import torch.utils.data
    
from lib.averageMeter import AverageMeters
from lib.logger import colorlogger
from lib.timer import Timers
from lib.averageMeter import AverageMeters
# from lib.torch_utils import adjust_learning_rate
from lib.viz_utils import draw

# from modeling.build_model import Pose2Seg
from datasets.CocoDatasetInfo import CocoDatasetInfo, annToMask
from datasets.OCP import OCP_Dataset
# from test import test

import random 
random.seed(88)
np.random.seed(88)
torch.manual_seed(88)


NAME = "release_base"

# Set `LOG_DIR` and `SNAPSHOT_DIR`
def setup_logdir():
    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()) 
    LOGDIR = os.path.join('logs', '%s_%s'%(NAME, timestamp))
    SNAPSHOTDIR = os.path.join('snapshot', '%s_%s'%(NAME, timestamp))
    if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)
    if not os.path.exists(SNAPSHOTDIR):
        os.makedirs(SNAPSHOTDIR)
    return LOGDIR, SNAPSHOTDIR
LOGDIR, SNAPSHOTDIR = setup_logdir()

# Set logging 
logger = colorlogger(log_dir=LOGDIR, log_name='train_logs.txt')

# Set Global Timer
timers = Timers()

# Set Global AverageMeter
averMeters = AverageMeters()

outdir = Path('./viz/')


def train(dataloader, epoch, iteration):
    # switch to train mode
    # model.train()
    
    averMeters.clear()
    end = time.time()
    for i, inputs in enumerate(dataloader): 
        averMeters['data_time'].update(time.time() - end)
        iteration += 1
        
        # forward
        input_dict = inputs
        print(i)
        # print(len(input_dict['batchimgs']))
        # print(input_dict['batchimgs'][0].shape)
        # print(len(input_dict['batchkpts']))
        # print(input_dict['batchkpts'][0].shape)
        # print(len(input_dict['batchmasks']))
        # print(input_dict['batchmasks'][0].shape)

        img = input_dict['batchimgs'][0]
        kpts = input_dict['batchkpts'][0]
        masks = input_dict['batchmasks'][0]

        out_img_path = outdir / f'{i}.jpg'
        cv2.imwrite(str(out_img_path), img)

        draw(
            img, 
            masks, 
            kpts, 
            out_img_path,
        )


        # import pdb;pdb.set_trace()
        
        # measure elapsed time
        averMeters['batch_time'].update(time.time() - end)
        end = time.time()
        
        if i % 10 == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  .format(
                      epoch, i, len(dataloader), 
                      batch_time=averMeters['batch_time'], data_time=averMeters['data_time'],
                 ))
        
    return iteration

class Dataset():
    def __init__(self):
        ImageRoot = './data/coco2017/train2017'
        AnnoFile = './data/coco2017/annotations/person_keypoints_train2017_pose2seg.json'
        self.datainfos = CocoDatasetInfo(ImageRoot, AnnoFile, onlyperson=True, loadimg=True)
    
    def __len__(self):
        return len(self.datainfos)
    
    def __getitem__(self, idx):
        rawdata = self.datainfos[idx]
        img = rawdata['data']
        image_id = rawdata['id']
        
        height, width = img.shape[0:2]
        gt_kpts = np.float32(rawdata['gt_keypoints']).transpose(0, 2, 1) # (N, 17, 3)
        gt_segms = rawdata['segms']
        gt_masks = np.array([annToMask(segm, height, width) for segm in gt_segms])
    
        return {'img': img, 'kpts': gt_kpts, 'masks': gt_masks}
        
    def collate_fn(self, batch):
        batchimgs = [data['img'] for data in batch]
        batchkpts = [data['kpts'] for data in batch]
        batchmasks = [data['masks'] for data in batch]
        return {'batchimgs': batchimgs, 'batchkpts': batchkpts, 'batchmasks':batchmasks}
        
if __name__=='__main__':
    logger.info('===========> loading model <===========')
    # model = Pose2Seg().cuda()
    #model.init("")
    # model.train()
    
    logger.info('===========> loading data <===========')
    datasetTrain = OCP_Dataset(
        imageroot='./data/coco2017/val2017',
        annofile='./data/coco2017/annotations/person_keypoints_val2017_pose2seg.json',
    )
    dataloaderTrain = torch.utils.data.DataLoader(datasetTrain, batch_size=1, shuffle=True,
                                                   num_workers=1, pin_memory=False,
                                                   collate_fn=datasetTrain.collate_fn)


    iteration = 0
    epoch = 0
    while iteration < 14150*1:
        logger.info('===========>   training    <===========')
        iteration = train(dataloaderTrain, epoch, iteration)
        epoch += 1
            