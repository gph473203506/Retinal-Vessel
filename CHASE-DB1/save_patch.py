from __future__ import division
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
from extract_patches import *
from help_functions import *
from extract_patches import get_data_training

# 从配置文件加载设置
#数据集patch化
path_data = './CHASE_DB1_datasets_training_testing/'

print('extracting patches')
patches_imgs_train,parches_masks_train = get_data_training(
    DRIVE_train_imgs_original = path_data + 'CHASE_DB1_dataset_imgs_train.hdf5',#images
    DRIVE_train_groudTruth = path_data + 'CHASE_DB1_dataset_groundTruth_train.hdf5',#masks
    patch_height = 64,
    patch_width = 64,
    N_subimgs = 168000,
    inside_FOV = 'True'#只在FOV中选择补丁(默认== True)
)

#将结果保存为npy文件
np.save('patches_imgs_train',patches_imgs_train)
np.save('patches_masks_train',parches_masks_train)