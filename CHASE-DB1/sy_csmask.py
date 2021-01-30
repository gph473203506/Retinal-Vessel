from __future__ import division
#==========================================================
#
#  This prepare the hdf5 datasets of the DRIVE database
#
#============================================================
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)
import os
import h5py
import numpy as np
from PIL import Image
from sy_pre_processing import clahe_equalized,dataset_normalized,rgb2gray,fs,pz,adjust_gamma,qb
from matplotlib import pyplot as plt
from colorsys import rgb_to_yiq


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
def write_hdf5(arr,outfile):
  with h5py.File(outfile,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)


#------------Path of the images --------------------------------------------------------------
#train
original_imgs_train = "./CHASE_DB1/training/images/"
#test
original_imgs_test = "./CHASE_DB1/test/images/"
#---------------------------------------------------------------------------------------------

Nimgs = 14
channels = 1
height = 960
width = 999
dataset_path = "./CHASE_DB1/training/mask/"
dataset_path1 = "./CHASE_DB1/test/mask/"

P = 1

def RGB_TO_YIQ(img):
    x=np.array([0.299,0.587,0.114,0.596,-0.274,-0.322,0.211,-0.523,0.312],dtype = np.float)
    x=x.reshape(3,3)
    z=np.empty((img.shape[0],img.shape[1],1))    
    for h in range(img.shape[0]):
        for w in range(img.shape[1]):
            R=img[h,w,0]
            G=img[h,w,1]
            B=img[h,w,2]
            
            Y,I,Q=rgb_to_yiq(R,G,B)
            if Y>=0.3:
                z[h,w,0]=1
            else:
                z[h,w,0]=0
            # img[h,w,:]=jg[:]
    return z

def get_datasets(imgs_dir,train_test="null"):
# 遍历图片，imgs_dir是图片的地址，groundTruth_dir是GT地址，borderMasks_dir是背景地址
# 返回的imgs, groundTruth, border_masks分别对应图片、GT和背景   
    Nimgs = 14
    channels = 1
    imgs = np.empty((Nimgs,height,width,channels))    
    border_masks_test = np.empty((Nimgs,height,width))
    # border_masks = np.empty((Nimgs,height,width))    
    for path,subdirs,files in os.walk(imgs_dir): #list all files, directories in the path
        files.sort(key=lambda subdirs: int(subdirs[6:8]))
        for i in range(len(files)):
            img = Image.open(imgs_dir+files[i])
            img = np.asarray(img)
            img_YIQ = RGB_TO_YIQ(img)
            img_YIQ = img_YIQ.reshape(img_YIQ.shape[0],img_YIQ.shape[1])
            border_masks_test[i]=img_YIQ
    return border_masks_test

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
#getting the training datasets
border_masks_test = get_datasets(original_imgs_train,"train")
print(border_masks_test.shape)
for i in range(border_masks_test.shape[0]):
    B = border_masks_test[i,:,:]
    plt.imsave(dataset_path+str(i)+'.png',np.squeeze(B),cmap='gray')
# print ("saving train datasets")
# write_hdf5(border_masks_train,dataset_path + "DRIVE_dataset_borderMasks_train.hdf5")

# #getting the testing datasets
border_masks_test = get_datasets(original_imgs_test,"test")
for i in range(border_masks_test.shape[0]):
    B = border_masks_test[i,:,:]
    plt.imsave(dataset_path1+str(i)+'.png',np.squeeze(B),cmap='gray')
# print ("saving test datasets")
# write_hdf5(border_masks_test,dataset_path + "DRIVE_dataset_borderMasks_test.hdf5")