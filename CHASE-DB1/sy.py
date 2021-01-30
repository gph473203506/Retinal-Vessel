from __future__ import division
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)
import numpy as np 
from matplotlib import pyplot as plt 
from help_functions import *
import cv2



#在不变量或本地变量上运行训练
path_data = './CHASE_DB1_datasets_training_testing/'

#原始测试图像(用于FOV选择)
DRIVE_test_imgs_original = path_data + 'CHASE_DB1_dataset_imgs_train.hdf5'
test_imgs_orig = load_hdf5(DRIVE_test_imgs_original)#测试集原始图片

# DRIVE_test_border_masks = path_data + 'DRIVE_dataset_borderMasks_train.hdf5'``
# test_border_masks = load_hdf5(DRIVE_test_border_masks)#测试集背景图片

# ini_gt = load_hdf5(path_data + 'DRIVE_dataset_groundTruth_train.hdf5')
# ini_gt = np.einsum('klij->kijl', ini_gt)

# print(ini_gt.shape)
name_experiment = 'test'
path_experiment = './' +name_experiment + '/'
print(test_imgs_orig.shape)
test_imgs_orig = np.einsum('klij->kijl', test_imgs_orig)
for idx in range(14):
    arr = test_imgs_orig[idx*2+1,:,:,:]
    cv2.imwrite(path_experiment+'_train_'+str(idx)+'.png',arr)
