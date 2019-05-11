#!/usr/bin/env python
# coding: utf-8

# In[75]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import os
from torch.utils.data import Dataset,DataLoader
import numpy as np
import cv2
import torch
from torchvision import transforms


# In[77]:


class FacialDetectionDataset(Dataset):
    def __init__(self,csv_file,root_dir,transform=None):
        self.key_pts_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.key_pts_frame)
    
    def __getitem__(self,idx):
        image_name = self.key_pts_frame.iloc[idx,0]
        image = mpimage.imread(os.path.join(self.root_dir,image_name))
        
        if image.shape[2] == 4:
            image = image[:,:,0:3]
            
        key_pts = self.key_pts_frame.iloc[idx,1:].as_matrix()
        key_pts = key_pts.astype('float').reshape(-1,2)
        
        sample = {'image':image,'key_pts':key_pts}
        
        if self.transform is not None:
            sample = self.transform(sample)
            
        return sample


# In[79]:


class Normalise(object):
    def __call__(self,sample):
        image,key_pts = sample['image'],sample['key_pts']
        
        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)
        
        image_copy = cv2.cvtColor(image_copy,cv2.COLOR_RGB2GRAY)
        
        image_copy = image_copy/255.0
        image_copy = image_copy.reshape(1,image_copy.shape[0],image_copy.shape[1])
        
        key_pts_copy = (key_pts_copy - 100)/50
        
        return {'image':image_copy,'key_pts':key_pts_copy}


# In[80]:


class Rescale(object):
    def __init__(self,output_size):
        assert isinstance(output_size,(int,tuple))
        self.output_size = output_size
        
    def __call__(self,sample):
        image , key_pts = sample['image'],sample['key_pts']
        
        h,w = image.shape[:2]
        
        if isinstance(self.output_size,int):
            if h>w:
                new_h , new_w = self.output_size * h/w , self.output_size
            else:
                new_h , new_w = self.output_size , self.output_size * w/h
        else:
            new_h , new_w = self.output_size
            
        new_h , new_w = int(new_h) , int(new_w)
        
        img = cv2.resize(image,(new_h,new_w))
        
        key_pts = key_pts * [new_w/w , new_h/h]
        
        return {'image':img,'key_pts':key_pts}
    


# In[81]:


class RandomCrop(object):
    def __init__(self,output_size):
        assert isinstance(output_size,(int,tuple))
        if isinstance(output_size , int):
            self.output_size = (output_size,output_size)
        else:
            assert len(output_size)==2
            self.output_size = output_size
            
    def __call__(self,sample):
        image , key_pts = sample['image'],sample['key_pts']
        
        h , w = image.shape[:2]
        
        new_h , new_w = self.output_size
        
        top = np.random.randint(0,h - new_h)
        left = np.random.randint(0,w - new_w)
        
        image = image[top:top+new_h , left:left+new_w]
        key_pts = key_pts - [left,top]
        
        return {'image':image,'key_pts':key_pts}


# In[82]:


class ToTensor(object):
    def __call__(self,sample):
        image , key_pts = sample['image'],sample['key_pts']
        
        if image.shape == 2:
            image = image.reshape(image.shape[0],image.shape[1],1)
            
        return {'image':torch.from_numpy(image),'key_pts':torch.from_numpy(key_pts)}


# In[ ]:




