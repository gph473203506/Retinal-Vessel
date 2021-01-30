# from __future__ import division

# # 这个python文件的目的是生成DRIVE数据集的hdf5文件


# import os
# import h5py
# import numpy as np
# from PIL import Image
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# def write_hdf5(arr,outfile):#写一个hdf5文件，arr为文件内容，outfil为文件保存位置
#     with h5py.File(outfile,"w") as f:
#         f.create_dataset("image",data=arr,dtype=arr.dtype)

# #------------初始化图片的路径--------------------------------------------------------------
# #训练集
# original_imgs_train="./DRIVE/training/images/"#图片路径
# groundTruth_imgs_train = "./DRIVE/training/1st_manual/"#GT路径
# borderMasks_imgs_train = "./DRIVE/training/mask/"#背景路径
# #测试集
# original_imgs_test = "./DRIVE/test/images/"
# groundTruth_imgs_test = "./DRIVE/test/1st_manual/"
# borderMasks_imgs_test = "./DRIVE/test/mask/"
# #---------------------------------------------------------------------------------------------

# #初始化一些变量
# Nimgs = 20 #图片数量
# channels = 3 #通道数
# height = 584 #图片高度
# width = 565 #图片宽度
# dataset_path = "./DRIVE_datasets_training_testing/" #hdf5文件保留位置

# def get_datasets(imgs_dir,groundTruth_dir,borderMasks_dir,train_test="null"):
#     #imgs_dir:图片地址。groundTruth_dir:GT地址。borderMasks_dir:背景地址。train_test:目标类型
#     imgs = np.empty((Nimgs,height,width,channels))
#     groundTruth = np.empty((Nimgs,height,width))
#     border_masks = np.empty((Nimgs,height,width))

#     for path,subdirs,files in os.walk(imgs_dir):#遍历imgs_dir文件夹里边的内容
#         for i in range(len(files)):
#             #存储原始图片
#             print("original images：" +files[i])
#             img = Image.open(imgs_dir+files[i])
#             imgs[i] = np.asarray(img)
#             #存储GT图片
#             groundTruth_name = files[i][0:2] + "_manual1.gif"
#             print ("ground truth name: " + groundTruth_name)
#             g_truth = Image.open(groundTruth_dir + groundTruth_name)
#             groundTruth[i] = np.asarray(g_truth)
#             #存储背景图片
#             border_masks_name = ""
#             if train_test=="train":
#                 border_masks_name = files[i][0:2] + "_training_mask.gif"
#             elif train_test=="test":
#                 border_masks_name = files[i][0:2] + "_test_mask.gif"
#             else:
#                 print("specify if train or test!!")
#                 exit()
#             print ("border masks name: " + border_masks_name)
#             b_mask = Image.open(borderMasks_dir + border_masks_name)
#             border_masks[i] = np.asarray(b_mask)
#     print ("imgs max: " +str(np.max(imgs)))
#     print ("imgs min: " +str(np.min(imgs)))
#     assert(np.max(groundTruth)==255 and np.max(border_masks)==255)
#     assert(np.min(groundTruth)==0 and np.min(border_masks)==0)
#     print ("ground truth and border masks are correctly withih pixel value range 0-255 (black-white)")
#     #改变我们输入图片的通道顺序
#     imgs = np.transpose(imgs,(0,3,1,2))
#     assert(imgs.shape == (Nimgs,channels,height,width))
#     groundTruth = np.reshape(groundTruth,(Nimgs,1,height,width))
#     border_masks = np.reshape(border_masks,(Nimgs,1,height,width))
#     assert(groundTruth.shape == (Nimgs,1,height,width))
#     assert(border_masks.shape == (Nimgs,1,height,width))
#     #imgs:存储图片的数组。 groundTruth：存储GT图片的数组。 border_masks：存储背景图片的数组。
#     return imgs, groundTruth, border_masks
    

# if not os.path.exists(dataset_path):#若保留hdf5文件的文件夹不存在，则创建该文件夹
#     os.makedirs(dataset_path)

# #获取训练集数据
# imgs_train,groundTruth_train,border_masks_train=get_datasets(original_imgs_train,groundTruth_imgs_train,borderMasks_imgs_train,"train")
# print ("saving train datasets")
# write_hdf5(imgs_train,dataset_path + "DRIVE_dataset_imgs_train.hdf5")
# write_hdf5(groundTruth_train,dataset_path + "DRIVE_dataset_groundTruth_train.hdf5")
# write_hdf5(border_masks_train,dataset_path + "DRIVE_dataset_borderMasks_train.hdf5")

# #获取测试集数据
# imgs_test, groundTruth_test, border_masks_test = get_datasets(original_imgs_test,groundTruth_imgs_test,borderMasks_imgs_test,"test")
# print ("saving test datasets")
# write_hdf5(imgs_test,dataset_path + "DRIVE_dataset_imgs_test.hdf5")
# write_hdf5(groundTruth_test, dataset_path + "DRIVE_dataset_groundTruth_test.hdf5")
# write_hdf5(border_masks_test,dataset_path + "DRIVE_dataset_borderMasks_test.hdf5")










from __future__ import division
#==========================================================
#
#  This prepare the hdf5 datasets of the DRIVE database
#
#============================================================

import os
import h5py
import numpy as np
from PIL import Image



def write_hdf5(arr,outfile):
  with h5py.File(outfile,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)


#------------Path of the images --------------------------------------------------------------
#train
original_imgs_train = "./DRIVE/training/images/"
groundTruth_imgs_train = "./DRIVE/training/1st_manual/"
borderMasks_imgs_train = "./DRIVE/training/mask/"
#test
original_imgs_test = "./DRIVE/test/images/"
groundTruth_imgs_test = "./DRIVE/test/1st_manual/"
borderMasks_imgs_test = "./DRIVE/test/mask/"
#---------------------------------------------------------------------------------------------

Nimgs = 20
channels = 3
height = 584
width = 565
dataset_path = "./DRIVE_datasets_training_testing/"

def get_datasets(imgs_dir,groundTruth_dir,borderMasks_dir,train_test="null"):
    imgs = np.empty((Nimgs,height,width,channels))
    groundTruth = np.empty((Nimgs,height,width))
    border_masks = np.empty((Nimgs,height,width))
    for path, subdirs, files in os.walk(imgs_dir): #list all files, directories in the path
        for i in range(len(files)):
            #original
            print ("original image: " +files[i])
            img = Image.open(imgs_dir+files[i])
            imgs[i] = np.asarray(img)
            #corresponding ground truth
            groundTruth_name = files[i][0:2] + "_manual1.gif"
            print ("ground truth name: " + groundTruth_name)
            g_truth = Image.open(groundTruth_dir + groundTruth_name)
            groundTruth[i] = np.asarray(g_truth)
            #corresponding border masks
            border_masks_name = ""
            if train_test=="train":
                border_masks_name = files[i][0:2] + "_training_mask.gif"
            elif train_test=="test":
                border_masks_name = files[i][0:2] + "_test_mask.gif"
            else:
                print ("specify if train or test!!")
                exit()
            print ("border masks name: " + border_masks_name)
            b_mask = Image.open(borderMasks_dir + border_masks_name)
            border_masks[i] = np.asarray(b_mask)

    print ("imgs max: " +str(np.max(imgs)))
    print ("imgs min: " +str(np.min(imgs)))
    assert(np.max(groundTruth)==255 and np.max(border_masks)==255)
    assert(np.min(groundTruth)==0 and np.min(border_masks)==0)
    print ("ground truth and border masks are correctly withih pixel value range 0-255 (black-white)")
    #reshaping for my standard tensors
    imgs = np.transpose(imgs,(0,3,1,2))
    assert(imgs.shape == (Nimgs,channels,height,width))
    groundTruth = np.reshape(groundTruth,(Nimgs,1,height,width))
    border_masks = np.reshape(border_masks,(Nimgs,1,height,width))
    assert(groundTruth.shape == (Nimgs,1,height,width))
    assert(border_masks.shape == (Nimgs,1,height,width))
    return imgs, groundTruth, border_masks

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
#getting the training datasets
imgs_train, groundTruth_train, border_masks_train = get_datasets(original_imgs_train,groundTruth_imgs_train,borderMasks_imgs_train,"train")
print ("saving train datasets")
write_hdf5(imgs_train, dataset_path + "DRIVE_dataset_imgs_train.hdf5")
write_hdf5(groundTruth_train, dataset_path + "DRIVE_dataset_groundTruth_train.hdf5")
write_hdf5(border_masks_train,dataset_path + "DRIVE_dataset_borderMasks_train.hdf5")

#getting the testing datasets
imgs_test, groundTruth_test, border_masks_test = get_datasets(original_imgs_test,groundTruth_imgs_test,borderMasks_imgs_test,"test")
print ("saving test datasets")
write_hdf5(imgs_test,dataset_path + "DRIVE_dataset_imgs_test.hdf5")
write_hdf5(groundTruth_test, dataset_path + "DRIVE_dataset_groundTruth_test.hdf5")
write_hdf5(border_masks_test,dataset_path + "DRIVE_dataset_borderMasks_test.hdf5")
