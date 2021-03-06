# -*- coding: utf-8 -*-
from __future__ import division
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

import models as M
import numpy as np
from help_functions import *
from keras.callbacks import ModelCheckpoint, TensorBoard,ReduceLROnPlateau
from keras import callbacks
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
#========= Load settings from Config file
#patch to the datasets
path_data = './DRIVE_datasets_training_testing/'
#Experiment name
name_experiment = 'test'
#training settings

batch_size = 1

####################################  Load Data #####################################3
patches_imgs_train  = np.load('patches_imgs_train.npy')
patches_masks_train = np.load('patches_masks_train.npy')

# patches_imgs_train  = patches_imgs_train[:,:,:,2:563]
# patches_masks_train = patches_masks_train[:,:,:,2:563]

patches_imgs_train = np.einsum('klij->kijl', patches_imgs_train)
patches_masks_train = np.einsum('klij->kijl', patches_masks_train)


print('Patch extracted')

#model = M.unet2_segment(input_size = (64,64,1))

model = M.GCN_BR_SPA_ConvLSTM_RVASPP_Mnet(input_size = (64,64,1))
model.summary()#输出模型各层的参数状况

print('Training')

nb_epoch = 25

mcp_save = ModelCheckpoint('weight_lstm.hdf5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

history = model.fit(patches_imgs_train,patches_masks_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              shuffle=True,
              verbose=1,
              validation_split=0.2, callbacks=[mcp_save, reduce_lr_loss] )
