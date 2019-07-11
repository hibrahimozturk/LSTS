# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np
import os
import glob
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import utils


class DHF1KDataset(Dataset):

    def __init__(self, data_dir, target_dir, img_width, img_height, is_fixation=False, small_part = 1):
        self.rgb_dir      = data_dir
        self.smap_dir      = os.path.join(target_dir, 'map')
        self.fixation_dir  = os.path.join(target_dir, 'fixation')

        self.img_width     = img_width
        self.img_height    = img_height
        self.is_fixation   = is_fixation
        
        self.rgb_list = list(self.listdir_nohidden_ext( self.rgb_dir, '*.npy'))
        self.smap_list = list(self.listdir_nohidden_ext( self.smap_dir, '*.png'))
        self.fixation_list = list(self.listdir_nohidden_ext( self.fixation_dir, '*.png'))
        self.small_part = small_part
        
        self.rgb_list, self.smap_list, self.fixation_list= self.__list_match__()
        
        
    def __list_match__(self):
         
        data_list_wo_ext = [x.split('.')[0] for x in self.rgb_list]
        smap_list_wo_ext = [x.split('.')[0] for x in self.smap_list]
        fixation_list_wo_ext = [x.split('.')[0] for x in self.fixation_list]
         
        intersect = set(data_list_wo_ext).intersection(set(smap_list_wo_ext))
        intersect = set(fixation_list_wo_ext).intersection(intersect)

 
        rgb_list = [x+'.npy' for x in intersect]
        rgb_list.sort()
        smap_list = [x+'.png' for x in intersect]
        smap_list.sort()
        fix_list = [x+'.png' for x in intersect]
        fix_list.sort()
         
        return rgb_list, smap_list, fix_list
     
    def __len__(self):
        return int(len(self.rgb_list)/self.small_part)

    def __getitem__(self, idx):
        
        data = np.load(os.path.join(self.rgb_dir, self.rgb_list[idx]))
        data = np.squeeze(data, axis=0)
        
        #TODO: resize'lar olmadan nasıl oluyor bakılıcak, yavaşlatıyor olabilir
        fixation_target = cv2.imread(os.path.join(self.fixation_dir , self.fixation_list[idx]))
        fixation_target = cv2.cvtColor(fixation_target, cv2.COLOR_BGR2GRAY)
        fixation_target = cv2.resize(fixation_target, (self.img_width,self.img_height))
        fixation_target = np.array(fixation_target, dtype='float32')
        fixation_target = np.interp(fixation_target, (fixation_target.min(), fixation_target.max()), (0, 1))
        fixation_target = fixation_target >= 0.5
        fixation_target = fixation_target.astype(np.float32)
        fixation_target = np.expand_dims(fixation_target, axis=0)
        
        smap_target = cv2.imread(os.path.join(self.smap_dir , self.smap_list[idx]))
        smap_target = cv2.cvtColor(smap_target, cv2.COLOR_BGR2GRAY)
        smap_target = cv2.resize(smap_target, (self.img_width,self.img_height))
        smap_target = np.array(smap_target, dtype='float32')
        smap_target = np.interp(smap_target, (smap_target.min(), smap_target.max()), (0, 1))
        smap_target = np.expand_dims(smap_target, axis=0)

        data = torch.from_numpy(data.copy()).float()
        fixation_target = torch.from_numpy(fixation_target.copy()).float()
        smap_target = torch.from_numpy(smap_target.copy()).float()

        sample = {'X': data, 'fixation_Y': fixation_target , 'smap_Y': smap_target, 'name':self.rgb_list[idx]}

        return sample
    

    def listdir_nohidden_ext(self, AllVideos_Path, ext='*_C.txt'):  # To ignore hidden files
        file_dir_extension = os.path.join(AllVideos_Path, ext)
        for f in glob.glob(file_dir_extension):
            filename = f.split('/')[-1]
            if not filename.startswith('.'):
                yield os.path.basename(f)


class DHF1KDualDataset(Dataset):

    def __init__(self, rgb_dir, flow_dir, target_dir, img_width,
                         img_height, is_fixation=False, small_part = 1):
        self.rgb_dir      = rgb_dir
        self.flow_dir     = flow_dir

        self.smap_dir      = os.path.join(target_dir, 'map')
        self.fixation_dir  = os.path.join(target_dir, 'fixation')

        self.img_width     = img_width
        self.img_height    = img_height
        self.is_fixation   = is_fixation
        
        self.rgb_list = list(self.listdir_nohidden_ext( self.rgb_dir, '*.npy'))
        self.flow_list = list(self.listdir_nohidden_ext( self.flow_dir, '*.npy'))

        self.smap_list = list(self.listdir_nohidden_ext( self.smap_dir, '*.png'))
        self.fixation_list = list(self.listdir_nohidden_ext( self.fixation_dir, '*.png'))
        self.small_part = small_part
        
        self.rgb_list, self.flow_list, self.smap_list, self.fixation_list = self.__list_match__()
        
        
    def __list_match__(self):
         
        intersect = set(self.flow_list).intersection(set(self.rgb_list))
 
        rgb_list = [x for x in intersect]
        rgb_list.sort()
        flow_list = [x for x in intersect]
        flow_list.sort()
        
        rgb_list_wo_ext = [x.split('.')[0] for x in self.rgb_list]
        flow_list_wo_ext = [x.split('.')[0] for x in self.flow_list]
        smap_list_wo_ext = [x.split('.')[0] for x in self.smap_list]
        fixation_list_wo_ext = [x.split('.')[0] for x in self.fixation_list]

        intersect = set(rgb_list_wo_ext).intersection(set(flow_list_wo_ext))
        intersect = set(smap_list_wo_ext).intersection(intersect)
        intersect = set(fixation_list_wo_ext).intersection(intersect)


        rgb_list = [x+'.npy' for x in intersect]
        rgb_list.sort()
        flow_list = [x+'.npy' for x in intersect]
        flow_list.sort()
        smap_list = [x+'.png' for x in intersect]
        smap_list.sort()
        fix_list = [x+'.png' for x in intersect]
        fix_list.sort()
         

        return rgb_list, flow_list, smap_list, fix_list
     
    def __len__(self):
        return int(len(self.rgb_list)/self.small_part)

    def __getitem__(self, idx):
        
        rgb_feature = np.load(os.path.join(self.rgb_dir, self.rgb_list[idx]))
        rgb_feature = np.squeeze(rgb_feature, axis=0)
        flow_feature = np.load(os.path.join(self.flow_dir, self.flow_list[idx]))
        flow_feature = np.squeeze(flow_feature, axis=0)
        
        feature = np.concatenate((rgb_feature, flow_feature))
        
        #TODO: resize'lar olmadan nasıl oluyor bakılıcak, yavaşlatıyor olabilir
        fixation_target = cv2.imread(os.path.join(self.fixation_dir , self.fixation_list[idx]))
        fixation_target = cv2.cvtColor(fixation_target, cv2.COLOR_BGR2GRAY)
#         fixation_target = cv2.resize(fixation_target, (self.img_width,self.img_height))
        fixation_target = np.array(fixation_target, dtype='float32')
        fixation_target = np.interp(fixation_target, (fixation_target.min(), fixation_target.max()), (0, 1))
        fixation_target = fixation_target >= 0.5
        fixation_target = fixation_target.astype(np.float32)
        fixation_target = np.expand_dims(fixation_target, axis=0)
        
        smap_target = cv2.imread(os.path.join(self.smap_dir , self.smap_list[idx]))
        smap_target = cv2.cvtColor(smap_target, cv2.COLOR_BGR2GRAY)
#         smap_target = cv2.resize(smap_target, (self.img_width,self.img_height))
        smap_target = np.array(smap_target, dtype='float32')
        smap_target = np.interp(smap_target, (smap_target.min(), smap_target.max()), (0, 1))
        smap_target = np.expand_dims(smap_target, axis=0)

        feature = torch.from_numpy(feature.copy()).float()
        fixation_target = torch.from_numpy(fixation_target.copy()).float()
        smap_target = torch.from_numpy(smap_target.copy()).float()

        sample = {'X': feature, 'fixation_Y': fixation_target , 'smap_Y': smap_target, 'name':self.rgb_list[idx]}

        return sample
    

    def listdir_nohidden_ext(self, AllVideos_Path, ext='*_C.txt'):  # To ignore hidden files
        file_dir_extension = os.path.join(AllVideos_Path, ext)
        for f in glob.glob(file_dir_extension):
            filename = f.split('/')[-1]
            if not filename.startswith('.'):
                yield os.path.basename(f)

if __name__ == "__main__":
    
    os.environ["CUDA_VISIBLE_DEVICES"]= '4'

    
    train_data = DHF1KDualDataset('../../../dataset/DHF1K/train/data_fix', 
                              '../../../dataset/DHF1K/train/flows_fix', 
                              '../../../dataset/DHF1K/train/target', 640, 320)

    # show a batch
    batch_size = 4
    for i in range(batch_size):
        sample = train_data[i]
        print(i, sample['X'].size(), sample['name'].size())

    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=4)

    for i, batch in enumerate(dataloader):
        print(i, batch['X'].size(), batch['name'].size())
    
