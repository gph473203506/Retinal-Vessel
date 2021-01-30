from __future__ import division
#==========================================================
#
#  This prepare the hdf5 datasets of the DRIVE database
#
#============================================================
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
import os
import h5py
import numpy as np
from PIL import Image
from sy_pre_processing import clahe_equalized,dataset_normalized,rgb2gray,fs,pz,adjust_gamma


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def write_hdf5(arr,outfile):
  with h5py.File(outfile,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)


#------------Path of the images --------------------------------------------------------------
#train
original_imgs_train = "./CHASE_DB1/training/images/"
borderMasks_imgs_train = "./CHASE_DB1/training/mask/"
groundTruth_imgs_train = "./CHASE_DB1/training/1st_manual/"
#test
original_imgs_test = "./CHASE_DB1/test/images/"
borderMasks_imgs_test = "./CHASE_DB1/test/mask/"
groundTruth_imgs_test = "./CHASE_DB1/test/1st_manual/"
#---------------------------------------------------------------------------------------------

Nimgs = 14
channels = 1
height = 960
width = 999
dataset_path = "./CHASE_DB1_datasets_training_testing/"

P = 1

def get_datasets(imgs_dir,groundTruth_dir,borderMasks_dir,train_test="null"):
# 遍历图片，imgs_dir是图片的地址，groundTruth_dir是GT地址，borderMasks_dir是背景地址
# 返回的imgs, groundTruth, border_masks分别对应图片、GT和背景
    if train_test == "train":
        Nimgs = 1680
        channels = 1
        imgs = np.empty((Nimgs,height,width,1))
        groundTruth = np.empty((Nimgs,height,width))
        # border_masks = np.empty((Nimgs,height,width))
    else:
        Nimgs = 14
        channels = 1
        imgs = np.empty((Nimgs,height,width,channels))    
        groundTruth = np.empty((Nimgs,height,width))
        # border_masks = np.empty((Nimgs,height,width))

    if train_test == "train":    
        O = 0
        for l in range(0,360,6):        
            for path,subdirs,files in os.walk(imgs_dir): #list all files, directories in the path
                files.sort(key=lambda subdirs: int(subdirs[6:8]))
                for i in range(len(files)):
                    img = Image.open(imgs_dir+files[i])
                    groundTruth_name = files[i][0:9] + "_1stHO.png"
                    g_truth = Image.open(groundTruth_dir + groundTruth_name) 
                    border_masks_name = ""
                    if train_test=="train":         
                        t_img = img.rotate(l)
                        t_g_truth = g_truth.rotate(l)
                        for j in range(2):
                            if j== 0:
                                img = t_img.transpose(Image.FLIP_LEFT_RIGHT)
                                g_truth = t_g_truth.transpose(Image.FLIP_LEFT_RIGHT)
                            else:
                                img = t_img
                                g_truth = t_g_truth
                            img = np.asarray(img)
                            img = img.reshape(img.shape[0],img.shape[1],3)
                            img = np.asarray(img)
                            g_truth = np.asarray(g_truth)
                            img = rgb2gray(img)
                            img = img.reshape(img.shape[0],img.shape[1])
                            img = clahe_equalized(img)
                            img = adjust_gamma(img)
                            img = img.reshape(img.shape[0],img.shape[1],1)
                            imgs[O] = img
                            groundTruth[O] = g_truth
                            O+=1
            print(l)
        print("O："+str(O))  
        # print(imgs.shape)     
        # imgs = my_PreProc(imgs) 
        # groundTruth = my_PreProc(groundTruth) 
        # border_masks = my_PreProc(border_masks) 
    else:
        print("start testing!")
        for path,subdirs,files in os.walk(imgs_dir): #list all files, directories in the path
            files.sort(key=lambda subdirs: int(subdirs[6:8]))
            for i in range(len(files)):
                img = Image.open(imgs_dir+files[i])
                groundTruth_name = files[i][0:9] + "_1stHO.png"
                g_truth = Image.open(groundTruth_dir + groundTruth_name)  
                # border_masks_name = files[i][0:2] + "_test_mask.gif"
                # b_mask = Image.open(borderMasks_dir + border_masks_name)
                img = np.asarray(img)
                img = img.reshape(img.shape[0],img.shape[1],3)
                img = rgb2gray(img)
                # img = fs(img)
                # img = pz(img)
                img = img.reshape(img.shape[0],img.shape[1])
                img = clahe_equalized(img)
                img = adjust_gamma(img)
                img = img.reshape(img.shape[0],img.shape[1],1)
                # img = dataset_normalized(img)
                imgs[i] = np.asarray(img)
                groundTruth[i] = np.asarray(g_truth)
                # border_masks[i] = np.asarray(b_mask)
    imgs = dataset_normalized(imgs)
    # imgs = qb(imgs,border_masks)#定位眼球部分
    # groundTruth = dataset_normalized(groundTruth)
    # border_masks =dataset_normalized(border_masks)
    print ("imgs max: " +str(np.max(imgs)))
    print ("imgs min: " +str(np.min(imgs)))
    print ("imgs shape: " +str(imgs.shape))
    print ("groundTruth max: " +str(np.max(groundTruth)))
    print ("groundTruth min: " +str(np.min(groundTruth)))
    print ("groundTruth shape: " +str(groundTruth.shape))
    # print ("border_masks max: " +str(np.max(border_masks)))
    # print ("border_masks min: " +str(np.min(border_masks)))
    # print ("border_masks shape: " +str(groundTruth.shape))
    # assert(np.max(groundTruth)==255 and np.max(border_masks)==255)
    # assert(np.min(groundTruth)==0 and np.min(border_masks)==0)
    print ("ground truth and border masks are correctly withih pixel value range 0-255 (black-white)")
    
    #reshaping for my standard tensors
    imgs = np.transpose(imgs,(0,3,1,2))
    assert(imgs.shape == (Nimgs,channels,height,width))
    groundTruth = np.reshape(groundTruth,(Nimgs,1,height,width))
    # border_masks = np.reshape(border_masks,(Nimgs,1,height,width))
    assert(imgs.shape == (Nimgs,channels,height,width))
    # assert(border_masks.shape == (Nimgs,1,height,width))  
    # return imgs, groundTruth, border_masks
    return imgs, groundTruth

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
#getting the training datasets
imgs_train, groundTruth_train = get_datasets(original_imgs_train,groundTruth_imgs_train,borderMasks_imgs_train,"train")
print ("saving train datasets")
write_hdf5(imgs_train, dataset_path + "CHASE_DB1_dataset_imgs_train.hdf5")
write_hdf5(groundTruth_train, dataset_path + "CHASE_DB1_dataset_groundTruth_train.hdf5")
# write_hdf5(border_masks_train,dataset_path + "DRIVE_dataset_borderMasks_train.hdf5")

#getting the testing datasets
imgs_test, groundTruth_test = get_datasets(original_imgs_test,groundTruth_imgs_test,borderMasks_imgs_test,"test")
print ("saving test datasets")
write_hdf5(imgs_test,dataset_path + "CHASE_DB1_dataset_imgs_test.hdf5")
write_hdf5(groundTruth_test, dataset_path + "CHASE_DB1_dataset_groundTruth_test.hdf5")
# write_hdf5(border_masks_test,dataset_path + "DRIVE_dataset_borderMasks_test.hdf5")
