from __future__ import division
###################################################
#
#   Script to pre-process the original imgs
#
##################################################


import numpy as np
from PIL import Image
import cv2
# from help_functions import *
import tensorflow as tf


#My pre processing (use for both training and testing!)
def my_PreProc(data,mode="train"):
    assert(len(data.shape)==4)
    assert (data.shape[1]==3)  #Use the original images
    #black-white conversion
    train_imgs = rgb2gray(data)
    #my preprocessing:
    train_imgs = dataset_normalized(train_imgs)
    train_imgs = clahe_equalized(train_imgs)
    train_imgs = adjust_gamma(train_imgs, 1.2)
    train_imgs = train_imgs/255.  #reduce to 0-1 range
    return train_imgs


#============================================================
#========= PRE PROCESSING FUNCTIONS ========================#
#============================================================

#==== histogram equalization
def histo_equalized(imgs):
    # assert (len(imgs.shape)==4)  #4D arrays
    # assert (imgs.shape[1]==1)  #check the channel is 1
    imgs_equalized = np.empty(imgs.shape)
    # for i in range(imgs.shape[0]):
        # imgs_equalized[i,0] = cv2.equalizeHist(np.array(imgs[i,0], dtype = np.uint8))
    imgs_equalized[:,:] = cv2.equalizeHist(np.array(imgs[:,:], dtype = np.uint8))
    return imgs_equalized


# CLAHE (Contrast Limited Adaptive Histogram Equalization)
#adaptive histogram equalization is used. In this, image is divided into small blocks called "tiles" (tileSize is 8x8 by default in OpenCV). Then each of these blocks are histogram equalized as usual. So in a small area, histogram would confine to a small region (unless there is noise). If noise is there, it will be amplified. To avoid this, contrast limiting is applied. If any histogram bin is above the specified contrast limit (by default 40 in OpenCV), those pixels are clipped and distributed uniformly to other bins before applying histogram equalization. After equalization, to remove artifacts in tile borders, bilinear interpolation is applied
#采用自适应直方图均衡化。在这种情况下，图像被划分为称为“tiles”的小块(在OpenCV中，tileSize默认为8x8)。然后像往常一样对每个块进行直方图均衡化。所以在一个小的区域内，直方图会局限在一个小的区域内(除非有噪声)。如果有噪音，它就会被放大。为了避免这种情况，应用了对比度限制。如果任何直方图bin高于指定的对比度限制(OpenCV中默认为40)，那么在应用直方图均衡化之前，这些像素将被裁剪并均匀分布到其他bin。均衡后，为了去除平铺边界中的伪影，应用双线性插值
def clahe_equalized(imgs):
    # assert (len(imgs.shape)==4)  #4D arrays
    # assert (imgs.shape[1]==1)  #check the channel is 1
    #create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=10.0,tileGridSize=(8,8))
    imgs_equalized = np.empty(imgs.shape)
    # for i in range(imgs.shape[0]):
    #     imgs_equalized[i,0] = clahe.apply(np.array(imgs[i,0],dtype = np.uint8))
    imgs_equalized[:,:] = clahe.apply(np.array(imgs[:,:], dtype = np.uint8))
    return imgs_equalized


# ===== normalize over the dataset
def dataset_normalized(imgs):
    # print(imgs.shape)
    # assert (len(imgs.shape)==3)  #4D arrays
    # assert (imgs.shape[0]==1)  #check the channel is 1

    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)

    # tmp = 3*imgs_std
    # imgs = imgs-imgs_mean
    # imgs[imgs > tmp] = tmp
    # imgs[imgs < -tmp] = -tmp
    # imgs_normalized = imgs / (3 * imgs_std)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    imgs_normalized = ((imgs_normalized - np.min(imgs_normalized)) / (np.max(imgs_normalized)-np.min(imgs_normalized)))*255
    return imgs_normalized
    # assert(len(imgs.shape)==4)
    # assert(imgs.shape[1]==1)
    # imgs_normalized = np.empty(imgs.shape)
    # imgs_std = np.std(imgs)
    # imgs_mean = np.mean(imgs)
    # imgs_normalized = (imgs-imgs_mean)/imgs_std
    # for i in range(imgs.shape[0]):
    #     imgs_normalized[i] = ((imgs_normalized[i]-np.min(imgs_normalized[i]))/ (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    # return imgs_normalized


def adjust_gamma(imgs, gamma=0.6):
    # assert (len(imgs.shape)==4)  #4D arrays
    # assert (imgs.shape[1]==1)  #check the channel is 1
    # build a lookup table mapping the pixel values [0, 255] to构建一个将像素值[0,255]映射到的查找表
    # their adjusted gamma values调整后的伽马值
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table使用查找表应用gamma校正
    new_imgs = np.empty(imgs.shape)
    # for i in range(imgs.shape[0]):
    #     new_imgs[i,0] = cv2.LUT(np.array(imgs[i,0], dtype = np.uint8), table)
    new_imgs[:,:] = cv2.LUT(np.array(imgs[:,:], dtype = np.uint8), table)
    return new_imgs

def fs(data):#腐蚀操作
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 椭圆结构    
    erosion = cv2.erode(data,kernel)
    return erosion

def pz(data):#膨胀操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 椭圆结构    
    dilation = cv2.dilate(data,kernel)
    return dilation

def rgb2gray(rgb):
    # assert (len(rgb.shape)==4)  #4D arrays
    # assert (rgb.shape[1]==3)
    # bn_imgs = rgb[:,0,:,:]*0.299 + rgb[:,1,:,:]*0.587 + rgb[:,2,:,:]*0.114
    # # bn_imgs = rgb[:,0,:,:]s
    # bn_imgs = np.reshape(bn_imgs,(rgb.shape[0],1,rgb.shape[2],rgb.shape[3]))
    bn_imgs = rgb[:,:,0]*0.299 + rgb[:,:,1]*0.587 + rgb[:,:,2]*0.114
    # bn_imgs = rgb[:,:,0]
    bn_imgs = np.reshape(bn_imgs,(rgb.shape[0],rgb.shape[1],1))
    return bn_imgs

#将眼球外的部分像素置为0,定位眼球部分
def qb(imgs,border_masks):
    print("imgs："+str(imgs.shape)+"GT："+str(border_masks.shape))
    # assert (len(imgs.shape)==4 and len(GrountTruth.shape)==4)  #4D arrays
    # assert (imgs.shape[0]==GrountTruth.shape[0])
    imgs[border_masks == 0] = 0
    return imgs