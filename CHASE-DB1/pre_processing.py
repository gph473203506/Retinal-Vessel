from __future__ import division

# 对原始图片进行预处理

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
from PIL import Image
import cv2

from help_functions import *


# 我的预处理(用于培训和测试!)
def my_PreProc(data):
    assert(len(data.shape)==4)
    assert(data.shape[1]==3)#RGB图片
    train_imgs = rgb2gray(data)#将RGB图片转换为二值图片
    #我的预处理
    train_imgs = dataset_normalized(train_imgs)
    train_imgs = clahe_equalized(train_imgs)
    train_imgs = adjust_gamma(train_imgs,1.2)
    train_imgs = train_imgs/255#将图片像素范围变到0-1
    return train_imgs


#============================================================
#======================== 预处理函数 ========================#
#============================================================

#直方图均衡化
def histo_equalized(imgs):
    assert(len(imgs.shape)==4)
    assert(imgs.shape[1]==1)
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = cv2.equalizeHist(np.array(imgs[i,0],dtype=np.uint8))
    return imgs_equalized


#CLAHE(对比度限制的自适应直方图均衡化)
#采用自适应直方图均衡化。
# 在这种情况下，图像被划分为称为“tiles”的小块(在OpenCV中，tileSize默认为8x8)。然后像往常一样对每个块进行直方图均衡化。所以在一个小的区域内，直方图会局限在一个小的区域内(除非有噪声)。如果有噪音，它就会被放大。为了避免这种情况，应用了对比度限制。如果任何直方图bin高于指定的对比度限制(OpenCV中默认为40)，那么在应用直方图均衡化之前，这些像素将被裁剪并均匀分布到其他bin。均衡后，为了去除平铺边界中的伪影，应用双线性插值
def clahe_equalized(imgs):
    assert(len(imgs.shape)==4)
    assert(imgs.shape[1]==1)
    # 创建CLAHE对象(参数是可选的)。
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = clahe.apply(np.array(imgs[i,0],dtype = np.uint8))
    return imgs_equalized


# 对数据集进行规范化
def dataset_normalized(imgs):
    assert(len(imgs.shape)==4)
    assert(imgs.shape[1]==1)
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i]-np.min(imgs_normalized[i]))/ (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    return imgs_normalized


#调整伽马值
def adjust_gamma(imgs,gamma=1.0):
    assert(len(imgs.shape)==4)
    assert(imgs.shape[1]==1)
    # 构建一个将像素值[0,255]映射到的查找表
    # 调整后的伽马值
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0,256)]).astype("uint8")
    # 使用查找表应用gamma校正
    new_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        new_imgs[i,0] = cv2.LUT(np.array(imgs[i,0],dtype = np.uint8),table)
    return new_imgs