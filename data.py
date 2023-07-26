import numpy as np
import cv2 as cv
import os
from skimage.util import img_as_float
from skimage import io
from skimage.segmentation import slic
import torch
import torchvision.transforms.functional as F


# edge map
def produce_edge_map(img_path, save_path, th1=100, th2=100):
    for file_name in os.listdir(img_path):
        img = cv.imread(os.path.join(img_path,file_name), cv.IMREAD_GRAYSCALE)
        assert img is not None, "file could not be read, check with os.path.exists()"
        edges = cv.Canny(img,th1,th2)
        cv.imwrite(os.path.join(save_path,file_name),edges)

# color map
def produce_color_map(img_path, save_path, n_segments=400, sigma=5):
    for file_name in os.listdir(img_path):
        image=img_as_float(io.imread(os.path.join(img_path,file_name)))
        segments=slic(image, n_segments=n_segments, sigma=sigma)
        mi=np.min(segments.reshape(-1))
        ma=np.max(segments.reshape(-1))
        image=image.transpose((2,0,1))
        for i in range(mi,ma+1):
            mask=(segments==i)
            region=image[:,mask]
            color=np.mean(region,axis=1).reshape(3,1,1)
            mask=mask.astype(np.int)
            image=np.expand_dims(mask,0).repeat(3,axis=0)*color+image*(1-mask)
        image=image.transpose((1,2,0))
        io.imsave(os.path.join(save_path,file_name),image)

# dataloader for extractor
class Dataloader_Ext(torch.utils.data.Dataset):
    def __init__(self, source_img_path, edge_path, color_path, nums=None, size=(256,256)):
        self.size=size
        self.source_img_path, self.edge_path, self.color_path = source_img_path, edge_path, color_path
        self.file_names=os.listdir(source_img_path)
        if nums is not None and nums<len(self.file_names):
            self.file_names=self.file_names[:nums]

    def __getitem__(self, idx):
        img = cv.resize(cv.imread(os.path.join(self.source_img_path, self.file_names[idx]), cv.IMREAD_COLOR),self.size)/255
        color = cv.resize(cv.imread(os.path.join(self.color_path, self.file_names[idx]), cv.IMREAD_COLOR),self.size)/255
        edge = np.expand_dims(cv.resize(cv.imread(os.path.join(self.edge_path, self.file_names[idx]), cv.IMREAD_GRAYSCALE),self.size),axis=0)/255
        img = img.transpose((2,0,1))
        color = color.transpose((2,0,1))
        return img.astype(np.float32), edge.astype(np.float32), color.astype(np.float32)

    def __len__(self):
        return len(self.file_names)

class Dataloader_EBRA(torch.utils.data.Dataset):
    def __init__(self, source_img_path, nums=None, size=(256,256), k=50):
        self.size=size
        self.k = k
        self.source_img_path = source_img_path
        self.file_names=os.listdir(source_img_path)
        if nums is not None and nums<len(self.file_names):
            self.file_names=self.file_names[:nums]

    def __getitem__(self, idx):
        img = cv.resize(cv.imread(os.path.join(self.source_img_path, self.file_names[idx]), cv.IMREAD_COLOR),self.size)/255
        img = img.transpose((2,0,1))
        bbox_list=[]
        # mask 
        coordinate=np.random.randint(0,128-self.k,2)
        x=coordinate[0]
        y=coordinate[1]
        mask=torch.zeros(1,256,256)
        mask[:,x:x+self.k,y:y+self.k]=1
        mask[:,x+128:x+self.k+128,y:y+self.k]=1
        mask[:,x:x+self.k,y+128:y+self.k+128]=1
        mask[:,x+128:x+self.k+128,y+128:y+self.k+128]=1

        # box list
        bbox_list=[[x,y,self.k,self.k],
                   [x+128,y,self.k,self.k],
                   [x,y+128,self.k,self.k],
                   [x+128,y+128,self.k,self.k]]
       
        return img.astype(np.float32),mask.float(),bbox_list
    # bbox_list
    def __len__(self):
        return len(self.file_names)


    
