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


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
channels = 1
height = 584
width = 565
dataset_path = "./DRIVE_datasets_training_testing/"


# def get_datasets(imgs_dir,groundTruth_dir,borderMasks_dir,train_test="null"):
# # 遍历图片，imgs_dir是图片的地址，groundTruth_dir是GT地址，borderMasks_dir是背景地址
# # 返回的imgs, groundTruth, border_masks分别对应图片、GT和背景
#     imgs = np.empty((Nimgs,height,width,channels))
#     groundTruth = np.empty((Nimgs,height,width))
#     border_masks = np.empty((Nimgs,height,width))
#     for path,subdirs,files in os.walk(imgs_dir): #list all files, directories in the path
#         for i in range(len(files)):
#             #original
#             print ("original image: " +files[i])
#             img = Image.open(imgs_dir+files[i])
#             imgs[i] = np.asarray(img)
#             #corresponding ground truth
#             groundTruth_name = files[i][0:2] + "_manual1.gif"
#             print ("ground truth name: " + groundTruth_name)
#             g_truth = Image.open(groundTruth_dir + groundTruth_name)
#             groundTruth[i] = np.asarray(g_truth)
#             #corresponding border masks
#             border_masks_name = ""
#             if train_test=="train":
#                 border_masks_name = files[i][0:2] + "_training_mask.gif"
#             elif train_test=="test":
#                 border_masks_name = files[i][0:2] + "_test_mask.gif"
#             else:
#                 print ("specify if train or test!!")
#                 exit()
#             print ("border masks name: " + border_masks_name)
#             b_mask = Image.open(borderMasks_dir + border_masks_name)
#             border_masks[i] = np.asarray(b_mask)

#     print ("imgs max: " +str(np.max(imgs)))
#     print ("imgs min: " +str(np.min(imgs)))
#     print ("imgs shape: " +str(imgs.shape))
#     print ("groundTruth max: " +str(np.max(groundTruth)))
#     print ("groundTruth min: " +str(np.min(groundTruth)))
#     print ("groundTruth shape: " +str(groundTruth.shape))
#     print ("border_masks max: " +str(np.max(border_masks)))
#     print ("border_masks min: " +str(np.min(border_masks)))
#     print ("border_masks shape: " +str(groundTruth.shape))
#     assert(np.max(groundTruth)==255 and np.max(border_masks)==255)
#     assert(np.min(groundTruth)==0 and np.min(border_masks)==0)
#     print ("ground truth and border masks are correctly withih pixel value range 0-255 (black-white)")
#     #reshaping for my standard tensors
#     imgs = np.transpose(imgs,(0,3,1,2))
#     assert(imgs.shape == (Nimgs,channels,height,width))
#     groundTruth = np.reshape(groundTruth,(Nimgs,1,height,width))
#     border_masks = np.reshape(border_masks,(Nimgs,1,height,width))
#     assert(groundTruth.shape == (Nimgs,1,height,width))
#     assert(border_masks.shape == (Nimgs,1,height,width))  
#     return imgs, groundTruth, border_masks

P = 1

def get_datasets(imgs_dir,groundTruth_dir,borderMasks_dir,train_test="null"):
# 遍历图片，imgs_dir是图片的地址，groundTruth_dir是GT地址，borderMasks_dir是背景地址
# 返回的imgs, groundTruth, border_masks分别对应图片、GT和背景
    if train_test == "train":
        Nimgs = 1200
        channels = 1
        imgs = np.empty((Nimgs,height,width,1))
        groundTruth = np.empty((Nimgs,height,width))
        border_masks = np.empty((Nimgs,height,width))
    else:
        Nimgs = 20
        channels = 1
        imgs = np.empty((Nimgs,height,width,channels))    
        groundTruth = np.empty((Nimgs,height,width))
        border_masks = np.empty((Nimgs,height,width))

    if train_test == "train":    
        O = 0
        for l in range(0,360,12):        
            for path,subdirs,files in os.walk(imgs_dir): #list all files, directories in the path
                files.sort(key=lambda subdirs: int(subdirs[:2]))
                for i in range(len(files)):
                    #original
                    # print ("original image: " +files[i])
                    img = Image.open(imgs_dir+files[i])
                    #corresponding ground truth
                    groundTruth_name = files[i][0:2] + "_manual1.gif"
                    # print ("ground truth name: " + groundTruth_name)
                    # print('sca：'+str(sca))
                    g_truth = Image.open(groundTruth_dir + groundTruth_name)    
                    
                    #corresponding border masks
                    border_masks_name = ""
                    if train_test=="train":
                        border_masks_name = files[i][0:2] + "_training_mask.gif"
                        b_mask = Image.open(borderMasks_dir + border_masks_name)                       

                        r_img = img.rotate(l)
                        r_g_truth = g_truth.rotate(l)
                        r_b_mask = b_mask.rotate(l)
                        # img = resizeImage(img,sca,P)
                        # g_truth = resizeImage(g_truth,sca,P)
                        # b_mask = resizeImage(b_mask,sca,P)
                        for j in range(2):
                            if j== 0:
                                img = r_img.transpose(Image.FLIP_LEFT_RIGHT)
                                # print("image3")
                                g_truth = r_g_truth.transpose(Image.FLIP_LEFT_RIGHT)
                                b_mask = r_b_mask.transpose(Image.FLIP_LEFT_RIGHT)
                                img = np.asarray(img)
                                
                            else:
                                img = r_img
                                g_truth = r_g_truth
                                b_mask = r_b_mask
                                img = np.asarray(r_img)   
                            
                            # print(img.shape)
                            # img = img.reshape(img.shape[0],img.shape[1],3)
                            # img = np.asarray(img)
                            g_truth = np.asarray(g_truth)
                            b_mask = np.asarray(b_mask)
                            img = rgb2gray(img)
                            img = img.reshape(img.shape[0],img.shape[1])
                            img = clahe_equalized(img)
                            img = adjust_gamma(img)
                            img = img.reshape(img.shape[0],img.shape[1],1)
                            imgs[O] = img 
                            groundTruth[O] = g_truth
                            border_masks[O] = b_mask
                            O+=1                                    
            print(l)
        print("O："+str(O))  
        assert(O==Nimgs)
        # print(imgs.shape)     
        # imgs = my_PreProc(imgs) 
        # groundTruth = my_PreProc(groundTruth) 
        # border_masks = my_PreProc(border_masks) 
    else:
        print("start testing!")
        for path,subdirs,files in os.walk(imgs_dir): #list all files, directories in the path
            files.sort(key=lambda subdirs: int(subdirs[:2]))
            for i in range(len(files)):
                img = Image.open(imgs_dir+files[i])
                groundTruth_name = files[i][0:2] + "_manual1.gif"
                g_truth = Image.open(groundTruth_dir + groundTruth_name)  
                border_masks_name = files[i][0:2] + "_test_mask.gif"
                b_mask = Image.open(borderMasks_dir + border_masks_name)
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
                border_masks[i] = np.asarray(b_mask)
    imgs = dataset_normalized(imgs)
    imgs = qb(imgs,border_masks)#定位眼球部分
    # groundTruth = dataset_normalized(groundTruth)
    # border_masks =dataset_normalized(border_masks)
    print ("imgs max: " +str(np.max(imgs)))
    print ("imgs min: " +str(np.min(imgs)))
    print ("imgs shape: " +str(imgs.shape))
    print ("groundTruth max: " +str(np.max(groundTruth)))
    print ("groundTruth min: " +str(np.min(groundTruth)))
    print ("groundTruth shape: " +str(groundTruth.shape))
    print ("border_masks max: " +str(np.max(border_masks)))
    print ("border_masks min: " +str(np.min(border_masks)))
    print ("border_masks shape: " +str(groundTruth.shape))
    assert(np.max(groundTruth)==255 and np.max(border_masks)==255)
    assert(np.min(groundTruth)==0 and np.min(border_masks)==0)
    print ("ground truth and border masks are correctly withih pixel value range 0-255 (black-white)")
    
    #reshaping for my standard tensors
    imgs = np.transpose(imgs,(0,3,1,2))
    assert(imgs.shape == (Nimgs,channels,height,width))
    groundTruth = np.reshape(groundTruth,(Nimgs,1,height,width))
    border_masks = np.reshape(border_masks,(Nimgs,1,height,width))
    assert(imgs.shape == (Nimgs,channels,height,width))
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
