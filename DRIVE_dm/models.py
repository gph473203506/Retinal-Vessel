from __future__ import division
from keras import losses
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout,SeparableConv2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.utils.vis_utils import plot_model as plot
from keras.optimizers import SGD
from keras.optimizers import *
import numpy as np
from keras.layers import *

# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def GCN_BR_SPA_ConvLSTM_RVASPP_Mnet(input_size = (256,256,1)):
    def deaspp(x,lst):
        x = Conv2D(256, [3,3], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)

        b11=DepthwiseConv2D((3,3),dilation_rate=(lst[0],lst[0]),padding="same",use_bias=False)(x)
        b11=BatchNormalization()(b11)
        b11=Activation("relu")(b11)
        b11=Conv2D(256,(1,1),padding="same",use_bias=False)(b11)
        b11=BatchNormalization()(b11)
        b11=Activation("relu")(b11)

        x12 = Concatenate()([x,b11])
        b12 = DepthwiseConv2D((3,3),dilation_rate=(lst[1],lst[1]),padding="same",use_bias=False)(x12)
        b12=BatchNormalization()(b12)
        b12=Activation("relu")(b12)
        b12=Conv2D(256,(1,1),padding="same",use_bias=False)(b12)
        b12=BatchNormalization()(b12)
        b12=Activation("relu")(b12)

        x13 = Concatenate()([x12,b12])
        b13 = DepthwiseConv2D((3,3),dilation_rate=(lst[2],lst[2]),padding="same",use_bias=False)(x13)
        b13=BatchNormalization()(b13)
        b13=Activation("relu")(b13)
        b13=Conv2D(256,(1,1),padding="same",use_bias=False)(b13)
        b13=BatchNormalization()(b13)
        b13=Activation("relu")(b13)

        xjg = Concatenate()([x13,b13])
        xjg=Conv2D(256,(1,1),padding="same",use_bias=False)(xjg)
        xjg=BatchNormalization()(xjg)
        xjg=Activation("relu")(xjg)
        return xjg
    def RVaspp(x,input_shape,out_stride):
        scale_img_2 = AveragePooling2D(pool_size=(2, 2), name='input_scale2')(x)#256*256*1
        scale_img_3 = AveragePooling2D(pool_size=(2, 2), name='input_scale3')(scale_img_2)#128*128*1

        # x1 = deaspp(x,[4,3,2])
        # x2 = deaspp(scale_img_2,[5,4,3])
        x1 = deaspp(x,[6,7,8])
        x2 = deaspp(scale_img_2,[2,3,4])
        up2 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(x2)
        # x3 = deaspp(scale_img_3,[6,5,4])
        # up3 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(x3)
        # up3 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(up3)

        jg = Concatenate()([x1,up2])
        jg=Conv2D(512,(1,1),padding="same",use_bias=False)(jg)
        jg=BatchNormalization()(jg)
        jg=Activation("relu")(jg)
        jg=Dropout(0.5)(jg)
        return jg
    def Dot_layer(tensor):
        return Lambda(lambda tensor:K.batch_dot(tensor[0],tensor[1]))(tensor)
    N = input_size[0]
    inputs = Input(input_size) #(None, 64, 64, 1) 
    # bjs = Input(input_size) 
    s_img_2 = AveragePooling2D(pool_size=(2, 2), name='in_scale2')(inputs)#256*256*1
    s_img_3 = AveragePooling2D(pool_size=(2, 2), name='in_scale3')(s_img_2)#128*128*1
    
    # jg1 = multiply([inputs,bjs])
    conv11 = Conv2D(64, [3,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)#64*64*64
    conv11 = Conv2D(64, [1,3], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv11)
    conv12 = Conv2D(64, [1,3], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv12 = Conv2D(64, [3,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv12)
    conv1 = add([conv11,conv12])
    conv1 = BatchNormalization(axis=3)(conv1)
    #BR1
    conv1b = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1b = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(conv1b)
    conv1 = add([conv1,conv1b])
    conv1 = BatchNormalization(axis=3)(conv1)
        
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)#32*32*64  
    
    # jg2 = multiply([pool1,scale_img_2])
    s2 = SeparableConv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(s_img_2)
    is2 = Concatenate()([pool1,s2])
    conv21 = Conv2D(128, [3,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(is2)#32*32*128
    conv21 = Conv2D(128, [1,3], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv21)
    conv22 = Conv2D(128, [1,3], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(is2)
    conv22 = Conv2D(128, [3,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv22)
    conv2 = add([conv21,conv22])
    conv2 = BatchNormalization(axis=3)(conv2)
    #BR2
    conv2b = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2b = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(conv2b)
    conv2 = add([conv2,conv2b])
    conv2 = BatchNormalization(axis=3)(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)#16*16*128 
    
    # jg3 = multiply([pool2,scale_img_3])
    s3 = SeparableConv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(s_img_3)
    is3 = Concatenate()([pool2,s3])
    conv31 = Conv2D(256, [3,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(is3)#64*64*64
    conv31 = Conv2D(256, [1,3], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv31)
    conv32 = Conv2D(256, [1,3], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(is3)
    conv32 = Conv2D(256, [3,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv32)
    conv3 = add([conv31,conv32])
    conv3 = BatchNormalization(axis=3)(conv3)
    #BR3
    conv3b = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3b = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(conv3b)
    conv3 = add([conv3,conv3b])
    conv3 = BatchNormalization(axis=3)(conv3)

    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)#没有使用dropout 8*8*256

    # jg4 = multiply([pool3,scale_img_4])
    drop4_3=RVaspp(pool3,N,2)
    

    print("drop4_3："+str(drop4_3.shape))
    up6 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(drop4_3)#(None, 16, 16, 256) 
    print("up6："+str(up6.shape))
    up6 = BatchNormalization(axis=3)(up6)#(None, 16, 16, 256) 
    up6 = Activation('relu')(up6)#(None, 16, 16, 256) 

    print("drop3："+str(drop3.shape)+"up6："+str(up6.shape))
    #DANet1
    #Position Attention1
    PB = Reshape(target_shape=(16*16,256))(up6) #N*C  
    PC = Reshape(target_shape=(256,16*16))(up6) #C*N 
    # print(PB.shape)
    PS = Dot_layer([PB,PC])
    PS = Activation('softmax')(PS)    
    PF = Dot_layer([PS,PB])
    PF = Reshape(target_shape=(16,16,256))(PF)#H*W*C 
    PE = Add()([PF,up6])
    PE = BatchNormalization(axis=3)(PE)

    #Channel Attention1
    CB = Reshape(target_shape=(16*16,256))(drop3)#N*C   
    CC = Reshape(target_shape=(256,16*16))(drop3)#C*N
    CS = Dot_layer([CC,CB])
    CS = Activation('softmax')(CS)
    CF = Dot_layer([CB,CS])
    CF = Reshape(target_shape=(16,16,256))(CF)#H*W*C
    CE = Add()([CF,drop3])
    CE = BatchNormalization(axis=3)(CE)

    x1 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 256))(CE)#(None, 1, 16, 16, 25    
    x2 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 256))(PE)#(None, 1, 16, 16, 25
    merge6  = concatenate([x1,x2], axis = 1)# (None, 2, 16, 16, 25
    merge6 = ConvLSTM2D(filters = 128, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge6)#(None, 16, 16, 128)
            
    conv6 = SeparableConv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)#(None, 16, 16, 256) 
    conv6 = SeparableConv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)#(None, 16, 16, 256) 

    up7 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv6)#(None, 32, 32, 128)
    up7 = BatchNormalization(axis=3)(up7)#(None, 32, 32, 128)
    up7 = Activation('relu')(up7)#(None, 32, 32, 128)

    #DANet2
    #Position Attention1
    PB = Reshape(target_shape=(32*32,128))(up7) #N*C  
    PC = Reshape(target_shape=(128,32*32))(up7) #C*N 
    # print(PB.shape)
    PS = Dot_layer([PB,PC])
    PS = Activation('softmax')(PS)    
    PF = Dot_layer([PS,PB])
    PF = Reshape(target_shape=(32,32,128))(PF)#H*W*C 
    PE = Add()([PF,up7])
    PE = BatchNormalization(axis=3)(PE)

    #Channel Attention1
    CB = Reshape(target_shape=(32*32,128))(conv2)#N*C   
    CC = Reshape(target_shape=(128,32*32))(conv2)#C*N
    CS = Dot_layer([CC,CB])
    CS = Activation('softmax')(CS)
    CF = Dot_layer([CB,CS])
    CF = Reshape(target_shape=(32,32,128))(CF)#H*W*C
    CE = Add()([CF,conv2])
    CE = BatchNormalization(axis=3)(CE)

    x1 = Reshape(target_shape=(1, np.int32(N/2), np.int32(N/2), 128))(CE)#(None, 1, 32, 32, 128    
    x2 = Reshape(target_shape=(1, np.int32(N/2), np.int32(N/2), 128))(PE)#(None, 1, 32, 32, 128
    merge7  = concatenate([x1,x2], axis = 1) #(None, 2, 32, 32, 12
    merge7 = ConvLSTM2D(filters = 64, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge7)#(None, 32, 32, 64) 
        
    conv7 = SeparableConv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)#(None, 32, 32, 128)
    conv7 = SeparableConv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)#(None, 32, 32, 128)
    
    up8 = Conv2DTranspose(64, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv7)#(None, 64, 64, 64) 
    up8 = BatchNormalization(axis=3)(up8)#(None, 64, 64, 64) 
    up8 = Activation('relu')(up8)#64*64*64    

    #DANet2
    #Position Attention1
    PB = Reshape(target_shape=(64*64,64))(up8) #N*C  
    PC = Reshape(target_shape=(64,64*64))(up8) #C*N 
    # print(PB.shape)
    PS = Dot_layer([PB,PC])
    # PS = Dot()([PB,PC])#N*N
    PS = Activation('softmax')(PS)    
    PF = Dot_layer([PS,PB])
    # PF = Dot()([PS,PB])
    PF = Reshape(target_shape=(64,64,64))(PF)#H*W*C 
    PE = Add()([PF,up8])
    PE = BatchNormalization(axis=3)(PE)

    #Channel Attention1
    CB = Reshape(target_shape=(64*64,64))(conv1)#N*C   
    CC = Reshape(target_shape=(64,64*64))(conv1)#C*N
    # CS = Dot()([CC,CB])#C*C
    CS = Dot_layer([CC,CB])
    CS = Activation('softmax')(CS)
    # CF = Dot()([CB,CS])
    CF = Dot_layer([CB,CS])
    CF = Reshape(target_shape=(64,64,64))(CF)#H*W*C
    CE = Add()([CF,conv1])
    CE = BatchNormalization(axis=3)(CE)

    x1 = Reshape(target_shape=(1,  N, N, 64))(CE)#(None, 1,  N, N, 64    
    x2 = Reshape(target_shape=(1, N, N, 64))(PE)#(None, 1,  N, N, 64
    merge8  = concatenate([x1,x2], axis = 1) #(None, 2, 64, 64, 64
    merge8 = ConvLSTM2D(filters = 32, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge8)#(None, 64, 64, 32)   
    
    conv8 = SeparableConv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)#(None, 64, 64, 64) 
    conv8 = SeparableConv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)#(None, 64, 64, 64) 
    conv8 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)#(None, 64, 64, 2) 
    conv9 = Conv2D(1, 1, activation = 'sigmoid')(conv8)#(None, 64, 64, 1)
    model = Model(inputs =inputs,outputs = conv9)    
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

def GCN_BR_SPA_ConvLSTM_Mnet(input_size = (256,256,1)):
    def deaspp(x,lst):
        x = Conv2D(256, [3,3], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)

        b11=DepthwiseConv2D((3,3),dilation_rate=(lst[0],lst[0]),padding="same",use_bias=False)(x)
        b11=BatchNormalization()(b11)
        b11=Activation("relu")(b11)
        b11=Conv2D(256,(1,1),padding="same",use_bias=False)(b11)
        b11=BatchNormalization()(b11)
        b11=Activation("relu")(b11)

        x12 = Concatenate()([x,b11])
        b12 = DepthwiseConv2D((3,3),dilation_rate=(lst[1],lst[1]),padding="same",use_bias=False)(x12)
        b12=BatchNormalization()(b12)
        b12=Activation("relu")(b12)
        b12=Conv2D(256,(1,1),padding="same",use_bias=False)(b12)
        b12=BatchNormalization()(b12)
        b12=Activation("relu")(b12)

        x13 = Concatenate()([x12,b12])
        b13 = DepthwiseConv2D((3,3),dilation_rate=(lst[2],lst[2]),padding="same",use_bias=False)(x13)
        b13=BatchNormalization()(b13)
        b13=Activation("relu")(b13)
        b13=Conv2D(256,(1,1),padding="same",use_bias=False)(b13)
        b13=BatchNormalization()(b13)
        b13=Activation("relu")(b13)

        xjg = Concatenate()([x13,b13])
        xjg=Conv2D(256,(1,1),padding="same",use_bias=False)(xjg)
        xjg=BatchNormalization()(xjg)
        xjg=Activation("relu")(xjg)
        return xjg
    def RVaspp(x,input_shape,out_stride):
        scale_img_2 = AveragePooling2D(pool_size=(2, 2), name='input_scale2')(x)#256*256*1
        scale_img_3 = AveragePooling2D(pool_size=(2, 2), name='input_scale3')(scale_img_2)#128*128*1

        # x1 = deaspp(x,[4,3,2])
        # x2 = deaspp(scale_img_2,[5,4,3])
        x1 = deaspp(x,[6,7,8])
        x2 = deaspp(scale_img_2,[2,3,4])
        up2 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(x2)
        # x3 = deaspp(scale_img_3,[6,5,4])
        # up3 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(x3)
        # up3 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(up3)

        jg = Concatenate()([x1,up2])
        jg=Conv2D(512,(1,1),padding="same",use_bias=False)(jg)
        jg=BatchNormalization()(jg)
        jg=Activation("relu")(jg)
        jg=Dropout(0.5)(jg)
        return jg
    def Dot_layer(tensor):
        return Lambda(lambda tensor:K.batch_dot(tensor[0],tensor[1]))(tensor)
    N = input_size[0]
    inputs = Input(input_size) #(None, 64, 64, 1) 
    # bjs = Input(input_size) 
    s_img_2 = AveragePooling2D(pool_size=(2, 2), name='in_scale2')(inputs)#256*256*1
    s_img_3 = AveragePooling2D(pool_size=(2, 2), name='in_scale3')(s_img_2)#128*128*1
    
    # jg1 = multiply([inputs,bjs])
    conv11 = Conv2D(64, [3,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)#64*64*64
    conv11 = Conv2D(64, [1,3], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv11)
    conv12 = Conv2D(64, [1,3], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv12 = Conv2D(64, [3,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv12)
    conv1 = add([conv11,conv12])
    conv1 = BatchNormalization(axis=3)(conv1)
    #BR1
    conv1b = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1b = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(conv1b)
    conv1 = add([conv1,conv1b])
    conv1 = BatchNormalization(axis=3)(conv1)
        
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)#32*32*64  
    
    # jg2 = multiply([pool1,scale_img_2])
    s2 = SeparableConv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(s_img_2)
    is2 = Concatenate()([pool1,s2])
    conv21 = Conv2D(128, [3,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(is2)#32*32*128
    conv21 = Conv2D(128, [1,3], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv21)
    conv22 = Conv2D(128, [1,3], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(is2)
    conv22 = Conv2D(128, [3,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv22)
    conv2 = add([conv21,conv22])
    conv2 = BatchNormalization(axis=3)(conv2)
    #BR2
    conv2b = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2b = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(conv2b)
    conv2 = add([conv2,conv2b])
    conv2 = BatchNormalization(axis=3)(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)#16*16*128 
    
    # jg3 = multiply([pool2,scale_img_3])
    s3 = SeparableConv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(s_img_3)
    is3 = Concatenate()([pool2,s3])
    conv31 = Conv2D(256, [3,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(is3)#64*64*64
    conv31 = Conv2D(256, [1,3], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv31)
    conv32 = Conv2D(256, [1,3], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(is3)
    conv32 = Conv2D(256, [3,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv32)
    conv3 = add([conv31,conv32])
    conv3 = BatchNormalization(axis=3)(conv3)
    #BR3
    conv3b = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3b = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(conv3b)
    conv3 = add([conv3,conv3b])
    conv3 = BatchNormalization(axis=3)(conv3)

    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)#没有使用dropout 8*8*256

    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)     
    conv4_1 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4_1 = Dropout(0.5)(conv4_1)
    conv4_2 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop4_1)     
    conv4_2 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4_2)
    drop4_3 = Dropout(0.5)(conv4_2)

    

    print("drop4_3："+str(drop4_3.shape))
    up6 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(drop4_3)#(None, 16, 16, 256) 
    print("up6："+str(up6.shape))
    up6 = BatchNormalization(axis=3)(up6)#(None, 16, 16, 256) 
    up6 = Activation('relu')(up6)#(None, 16, 16, 256) 

    print("drop3："+str(drop3.shape)+"up6："+str(up6.shape))
    #DANet1
    #Position Attention1
    PB = Reshape(target_shape=(16*16,256))(up6) #N*C  
    PC = Reshape(target_shape=(256,16*16))(up6) #C*N 
    # print(PB.shape)
    PS = Dot_layer([PB,PC])
    PS = Activation('softmax')(PS)    
    PF = Dot_layer([PS,PB])
    PF = Reshape(target_shape=(16,16,256))(PF)#H*W*C 
    PE = Add()([PF,up6])
    PE = BatchNormalization(axis=3)(PE)

    #Channel Attention1
    CB = Reshape(target_shape=(16*16,256))(drop3)#N*C   
    CC = Reshape(target_shape=(256,16*16))(drop3)#C*N
    CS = Dot_layer([CC,CB])
    CS = Activation('softmax')(CS)
    CF = Dot_layer([CB,CS])
    CF = Reshape(target_shape=(16,16,256))(CF)#H*W*C
    CE = Add()([CF,drop3])
    CE = BatchNormalization(axis=3)(CE)

    x1 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 256))(CE)#(None, 1, 16, 16, 25    
    x2 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 256))(PE)#(None, 1, 16, 16, 25
    merge6  = concatenate([x1,x2], axis = 1)# (None, 2, 16, 16, 25
    merge6 = ConvLSTM2D(filters = 128, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge6)#(None, 16, 16, 128)
            
    conv6 = SeparableConv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)#(None, 16, 16, 256) 
    conv6 = SeparableConv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)#(None, 16, 16, 256) 

    up7 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv6)#(None, 32, 32, 128)
    up7 = BatchNormalization(axis=3)(up7)#(None, 32, 32, 128)
    up7 = Activation('relu')(up7)#(None, 32, 32, 128)

    #DANet2
    #Position Attention1
    PB = Reshape(target_shape=(32*32,128))(up7) #N*C  
    PC = Reshape(target_shape=(128,32*32))(up7) #C*N 
    # print(PB.shape)
    PS = Dot_layer([PB,PC])
    PS = Activation('softmax')(PS)    
    PF = Dot_layer([PS,PB])
    PF = Reshape(target_shape=(32,32,128))(PF)#H*W*C 
    PE = Add()([PF,up7])
    PE = BatchNormalization(axis=3)(PE)

    #Channel Attention1
    CB = Reshape(target_shape=(32*32,128))(conv2)#N*C   
    CC = Reshape(target_shape=(128,32*32))(conv2)#C*N
    CS = Dot_layer([CC,CB])
    CS = Activation('softmax')(CS)
    CF = Dot_layer([CB,CS])
    CF = Reshape(target_shape=(32,32,128))(CF)#H*W*C
    CE = Add()([CF,conv2])
    CE = BatchNormalization(axis=3)(CE)

    x1 = Reshape(target_shape=(1, np.int32(N/2), np.int32(N/2), 128))(CE)#(None, 1, 32, 32, 128    
    x2 = Reshape(target_shape=(1, np.int32(N/2), np.int32(N/2), 128))(PE)#(None, 1, 32, 32, 128
    merge7  = concatenate([x1,x2], axis = 1) #(None, 2, 32, 32, 12
    merge7 = ConvLSTM2D(filters = 64, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge7)#(None, 32, 32, 64) 
        
    conv7 = SeparableConv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)#(None, 32, 32, 128)
    conv7 = SeparableConv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)#(None, 32, 32, 128)
    
    up8 = Conv2DTranspose(64, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv7)#(None, 64, 64, 64) 
    up8 = BatchNormalization(axis=3)(up8)#(None, 64, 64, 64) 
    up8 = Activation('relu')(up8)#64*64*64    

    #DANet2
    #Position Attention1
    PB = Reshape(target_shape=(64*64,64))(up8) #N*C  
    PC = Reshape(target_shape=(64,64*64))(up8) #C*N 
    # print(PB.shape)
    PS = Dot_layer([PB,PC])
    # PS = Dot()([PB,PC])#N*N
    PS = Activation('softmax')(PS)    
    PF = Dot_layer([PS,PB])
    # PF = Dot()([PS,PB])
    PF = Reshape(target_shape=(64,64,64))(PF)#H*W*C 
    PE = Add()([PF,up8])
    PE = BatchNormalization(axis=3)(PE)

    #Channel Attention1
    CB = Reshape(target_shape=(64*64,64))(conv1)#N*C   
    CC = Reshape(target_shape=(64,64*64))(conv1)#C*N
    # CS = Dot()([CC,CB])#C*C
    CS = Dot_layer([CC,CB])
    CS = Activation('softmax')(CS)
    # CF = Dot()([CB,CS])
    CF = Dot_layer([CB,CS])
    CF = Reshape(target_shape=(64,64,64))(CF)#H*W*C
    CE = Add()([CF,conv1])
    CE = BatchNormalization(axis=3)(CE)

    x1 = Reshape(target_shape=(1,  N, N, 64))(CE)#(None, 1,  N, N, 64    
    x2 = Reshape(target_shape=(1, N, N, 64))(PE)#(None, 1,  N, N, 64
    merge8  = concatenate([x1,x2], axis = 1) #(None, 2, 64, 64, 64
    merge8 = ConvLSTM2D(filters = 32, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge8)#(None, 64, 64, 32)   
    
    conv8 = SeparableConv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)#(None, 64, 64, 64) 
    conv8 = SeparableConv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)#(None, 64, 64, 64) 
    conv8 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)#(None, 64, 64, 2) 
    conv9 = Conv2D(1, 1, activation = 'sigmoid')(conv8)#(None, 64, 64, 1)
    model = Model(inputs =inputs,outputs = conv9)    
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

def GCN_BR_ConvLSTM_Mnet(input_size = (256,256,1)):
    def deaspp(x,lst):
        x = Conv2D(256, [3,3], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)

        b11=DepthwiseConv2D((3,3),dilation_rate=(lst[0],lst[0]),padding="same",use_bias=False)(x)
        b11=BatchNormalization()(b11)
        b11=Activation("relu")(b11)
        b11=Conv2D(256,(1,1),padding="same",use_bias=False)(b11)
        b11=BatchNormalization()(b11)
        b11=Activation("relu")(b11)

        x12 = Concatenate()([x,b11])
        b12 = DepthwiseConv2D((3,3),dilation_rate=(lst[1],lst[1]),padding="same",use_bias=False)(x12)
        b12=BatchNormalization()(b12)
        b12=Activation("relu")(b12)
        b12=Conv2D(256,(1,1),padding="same",use_bias=False)(b12)
        b12=BatchNormalization()(b12)
        b12=Activation("relu")(b12)

        x13 = Concatenate()([x12,b12])
        b13 = DepthwiseConv2D((3,3),dilation_rate=(lst[2],lst[2]),padding="same",use_bias=False)(x13)
        b13=BatchNormalization()(b13)
        b13=Activation("relu")(b13)
        b13=Conv2D(256,(1,1),padding="same",use_bias=False)(b13)
        b13=BatchNormalization()(b13)
        b13=Activation("relu")(b13)

        xjg = Concatenate()([x13,b13])
        xjg=Conv2D(256,(1,1),padding="same",use_bias=False)(xjg)
        xjg=BatchNormalization()(xjg)
        xjg=Activation("relu")(xjg)
        return xjg
    def RVaspp(x,input_shape,out_stride):
        scale_img_2 = AveragePooling2D(pool_size=(2, 2), name='input_scale2')(x)#256*256*1
        scale_img_3 = AveragePooling2D(pool_size=(2, 2), name='input_scale3')(scale_img_2)#128*128*1

        # x1 = deaspp(x,[4,3,2])
        # x2 = deaspp(scale_img_2,[5,4,3])
        x1 = deaspp(x,[6,7,8])
        x2 = deaspp(scale_img_2,[2,3,4])
        up2 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(x2)
        # x3 = deaspp(scale_img_3,[6,5,4])
        # up3 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(x3)
        # up3 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(up3)

        jg = Concatenate()([x1,up2])
        jg=Conv2D(512,(1,1),padding="same",use_bias=False)(jg)
        jg=BatchNormalization()(jg)
        jg=Activation("relu")(jg)
        jg=Dropout(0.5)(jg)
        return jg
    def Dot_layer(tensor):
        return Lambda(lambda tensor:K.batch_dot(tensor[0],tensor[1]))(tensor)
    N = input_size[0]
    inputs = Input(input_size) #(None, 64, 64, 1) 
    # bjs = Input(input_size) 
    s_img_2 = AveragePooling2D(pool_size=(2, 2), name='in_scale2')(inputs)#256*256*1
    s_img_3 = AveragePooling2D(pool_size=(2, 2), name='in_scale3')(s_img_2)#128*128*1
    
    # jg1 = multiply([inputs,bjs])
    conv11 = Conv2D(64, [3,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)#64*64*64
    conv11 = Conv2D(64, [1,3], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv11)
    conv12 = Conv2D(64, [1,3], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv12 = Conv2D(64, [3,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv12)
    conv1 = add([conv11,conv12])
    conv1 = BatchNormalization(axis=3)(conv1)
    #BR1
    conv1b = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1b = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(conv1b)
    conv1 = add([conv1,conv1b])
    conv1 = BatchNormalization(axis=3)(conv1)
        
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)#32*32*64  
    
    # jg2 = multiply([pool1,scale_img_2])
    s2 = SeparableConv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(s_img_2)
    is2 = Concatenate()([pool1,s2])
    conv21 = Conv2D(128, [3,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(is2)#32*32*128
    conv21 = Conv2D(128, [1,3], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv21)
    conv22 = Conv2D(128, [1,3], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(is2)
    conv22 = Conv2D(128, [3,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv22)
    conv2 = add([conv21,conv22])
    conv2 = BatchNormalization(axis=3)(conv2)
    #BR2
    conv2b = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2b = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(conv2b)
    conv2 = add([conv2,conv2b])
    conv2 = BatchNormalization(axis=3)(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)#16*16*128 
    
    # jg3 = multiply([pool2,scale_img_3])
    s3 = SeparableConv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(s_img_3)
    is3 = Concatenate()([pool2,s3])
    conv31 = Conv2D(256, [3,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(is3)#64*64*64
    conv31 = Conv2D(256, [1,3], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv31)
    conv32 = Conv2D(256, [1,3], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(is3)
    conv32 = Conv2D(256, [3,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv32)
    conv3 = add([conv31,conv32])
    conv3 = BatchNormalization(axis=3)(conv3)
    #BR3
    conv3b = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3b = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(conv3b)
    conv3 = add([conv3,conv3b])
    conv3 = BatchNormalization(axis=3)(conv3)

    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)#没有使用dropout 8*8*256

    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)     
    conv4_1 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4_1 = Dropout(0.5)(conv4_1)
    conv4_2 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop4_1)     
    conv4_2 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4_2)
    drop4_3 = Dropout(0.5)(conv4_2)

    

    print("drop4_3："+str(drop4_3.shape))
    up6 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(drop4_3)#(None, 16, 16, 256) 
    print("up6："+str(up6.shape))
    up6 = BatchNormalization(axis=3)(up6)#(None, 16, 16, 256) 
    up6 = Activation('relu')(up6)#(None, 16, 16, 256) 

    print("drop3："+str(drop3.shape)+"up6："+str(up6.shape))
    #DANet1
    #Position Attention1
    PB = Reshape(target_shape=(16*16,256))(up6) #N*C  
    PC = Reshape(target_shape=(256,16*16))(up6) #C*N 
    # print(PB.shape)
    PS = Dot_layer([PB,PC])
    PS = Activation('softmax')(PS)    
    PF = Dot_layer([PS,PB])
    PF = Reshape(target_shape=(16,16,256))(PF)#H*W*C 
    PE = Add()([PF,up6])
    PE = BatchNormalization(axis=3)(PE)

    #Channel Attention1
    CB = Reshape(target_shape=(16*16,256))(drop3)#N*C   
    CC = Reshape(target_shape=(256,16*16))(drop3)#C*N
    CS = Dot_layer([CC,CB])
    CS = Activation('softmax')(CS)
    CF = Dot_layer([CB,CS])
    CF = Reshape(target_shape=(16,16,256))(CF)#H*W*C
    CE = Add()([CF,drop3])
    CE = BatchNormalization(axis=3)(CE)

    x1 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 256))(CE)#(None, 1, 16, 16, 25    
    x2 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 256))(PE)#(None, 1, 16, 16, 25
    merge6  = concatenate([x1,x2], axis = 1)# (None, 2, 16, 16, 25
    merge6 = ConvLSTM2D(filters = 128, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge6)#(None, 16, 16, 128)
            
    conv6 = SeparableConv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)#(None, 16, 16, 256) 
    conv6 = SeparableConv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)#(None, 16, 16, 256) 

    up7 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv6)#(None, 32, 32, 128)
    up7 = BatchNormalization(axis=3)(up7)#(None, 32, 32, 128)
    up7 = Activation('relu')(up7)#(None, 32, 32, 128)

    #DANet2
    #Position Attention1
    PB = Reshape(target_shape=(32*32,128))(up7) #N*C  
    PC = Reshape(target_shape=(128,32*32))(up7) #C*N 
    # print(PB.shape)
    PS = Dot_layer([PB,PC])
    PS = Activation('softmax')(PS)    
    PF = Dot_layer([PS,PB])
    PF = Reshape(target_shape=(32,32,128))(PF)#H*W*C 
    PE = Add()([PF,up7])
    PE = BatchNormalization(axis=3)(PE)

    #Channel Attention1
    CB = Reshape(target_shape=(32*32,128))(conv2)#N*C   
    CC = Reshape(target_shape=(128,32*32))(conv2)#C*N
    CS = Dot_layer([CC,CB])
    CS = Activation('softmax')(CS)
    CF = Dot_layer([CB,CS])
    CF = Reshape(target_shape=(32,32,128))(CF)#H*W*C
    CE = Add()([CF,conv2])
    CE = BatchNormalization(axis=3)(CE)

    x1 = Reshape(target_shape=(1, np.int32(N/2), np.int32(N/2), 128))(CE)#(None, 1, 32, 32, 128    
    x2 = Reshape(target_shape=(1, np.int32(N/2), np.int32(N/2), 128))(PE)#(None, 1, 32, 32, 128
    merge7  = concatenate([x1,x2], axis = 1) #(None, 2, 32, 32, 12
    merge7 = ConvLSTM2D(filters = 64, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge7)#(None, 32, 32, 64) 
        
    conv7 = SeparableConv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)#(None, 32, 32, 128)
    conv7 = SeparableConv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)#(None, 32, 32, 128)
    
    up8 = Conv2DTranspose(64, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv7)#(None, 64, 64, 64) 
    up8 = BatchNormalization(axis=3)(up8)#(None, 64, 64, 64) 
    up8 = Activation('relu')(up8)#64*64*64    

    #DANet2
    #Position Attention1
    PB = Reshape(target_shape=(64*64,64))(up8) #N*C  
    PC = Reshape(target_shape=(64,64*64))(up8) #C*N 
    # print(PB.shape)
    PS = Dot_layer([PB,PC])
    # PS = Dot()([PB,PC])#N*N
    PS = Activation('softmax')(PS)    
    PF = Dot_layer([PS,PB])
    # PF = Dot()([PS,PB])
    PF = Reshape(target_shape=(64,64,64))(PF)#H*W*C 
    PE = Add()([PF,up8])
    PE = BatchNormalization(axis=3)(PE)

    #Channel Attention1
    CB = Reshape(target_shape=(64*64,64))(conv1)#N*C   
    CC = Reshape(target_shape=(64,64*64))(conv1)#C*N
    # CS = Dot()([CC,CB])#C*C
    CS = Dot_layer([CC,CB])
    CS = Activation('softmax')(CS)
    # CF = Dot()([CB,CS])
    CF = Dot_layer([CB,CS])
    CF = Reshape(target_shape=(64,64,64))(CF)#H*W*C
    CE = Add()([CF,conv1])
    CE = BatchNormalization(axis=3)(CE)

    x1 = Reshape(target_shape=(1,  N, N, 64))(CE)#(None, 1,  N, N, 64    
    x2 = Reshape(target_shape=(1, N, N, 64))(PE)#(None, 1,  N, N, 64
    merge8  = concatenate([x1,x2], axis = 1) #(None, 2, 64, 64, 64
    merge8 = ConvLSTM2D(filters = 32, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge8)#(None, 64, 64, 32)   
    
    conv8 = SeparableConv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)#(None, 64, 64, 64) 
    conv8 = SeparableConv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)#(None, 64, 64, 64) 
    conv8 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)#(None, 64, 64, 2) 
    conv9 = Conv2D(1, 1, activation = 'sigmoid')(conv8)#(None, 64, 64, 1)
    model = Model(inputs =inputs,outputs = conv9)    
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model    

def GCN_BR_SPA_ConvLSTM_net(input_size = (256,256,1)):
    def deaspp(x,lst):
        x = Conv2D(256, [3,3], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)

        b11=DepthwiseConv2D((3,3),dilation_rate=(lst[0],lst[0]),padding="same",use_bias=False)(x)
        b11=BatchNormalization()(b11)
        b11=Activation("relu")(b11)
        b11=Conv2D(256,(1,1),padding="same",use_bias=False)(b11)
        b11=BatchNormalization()(b11)
        b11=Activation("relu")(b11)

        x12 = Concatenate()([x,b11])
        b12 = DepthwiseConv2D((3,3),dilation_rate=(lst[1],lst[1]),padding="same",use_bias=False)(x12)
        b12=BatchNormalization()(b12)
        b12=Activation("relu")(b12)
        b12=Conv2D(256,(1,1),padding="same",use_bias=False)(b12)
        b12=BatchNormalization()(b12)
        b12=Activation("relu")(b12)

        x13 = Concatenate()([x12,b12])
        b13 = DepthwiseConv2D((3,3),dilation_rate=(lst[2],lst[2]),padding="same",use_bias=False)(x13)
        b13=BatchNormalization()(b13)
        b13=Activation("relu")(b13)
        b13=Conv2D(256,(1,1),padding="same",use_bias=False)(b13)
        b13=BatchNormalization()(b13)
        b13=Activation("relu")(b13)

        xjg = Concatenate()([x13,b13])
        xjg=Conv2D(256,(1,1),padding="same",use_bias=False)(xjg)
        xjg=BatchNormalization()(xjg)
        xjg=Activation("relu")(xjg)
        return xjg
    def RVaspp(x,input_shape,out_stride):
        scale_img_2 = AveragePooling2D(pool_size=(2, 2), name='input_scale2')(x)#256*256*1
        scale_img_3 = AveragePooling2D(pool_size=(2, 2), name='input_scale3')(scale_img_2)#128*128*1

        # x1 = deaspp(x,[4,3,2])
        # x2 = deaspp(scale_img_2,[5,4,3])
        x1 = deaspp(x,[6,7,8])
        x2 = deaspp(scale_img_2,[2,3,4])
        up2 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(x2)
        # x3 = deaspp(scale_img_3,[6,5,4])
        # up3 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(x3)
        # up3 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(up3)

        jg = Concatenate()([x1,up2])
        jg=Conv2D(512,(1,1),padding="same",use_bias=False)(jg)
        jg=BatchNormalization()(jg)
        jg=Activation("relu")(jg)
        jg=Dropout(0.5)(jg)
        return jg
    def Dot_layer(tensor):
        return Lambda(lambda tensor:K.batch_dot(tensor[0],tensor[1]))(tensor)
    N = input_size[0]
    inputs = Input(input_size) #(None, 64, 64, 1) 
    
    # jg1 = multiply([inputs,bjs])
    conv11 = Conv2D(64, [3,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)#64*64*64
    conv11 = Conv2D(64, [1,3], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv11)
    conv12 = Conv2D(64, [1,3], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv12 = Conv2D(64, [3,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv12)
    conv1 = add([conv11,conv12])
    conv1 = BatchNormalization(axis=3)(conv1)
    #BR1
    conv1b = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1b = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(conv1b)
    conv1 = add([conv1,conv1b])
    conv1 = BatchNormalization(axis=3)(conv1)
        
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)#32*32*64  
    conv21 = Conv2D(128, [3,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)#32*32*128
    conv21 = Conv2D(128, [1,3], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv21)
    conv22 = Conv2D(128, [1,3], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv22 = Conv2D(128, [3,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv22)
    conv2 = add([conv21,conv22])
    conv2 = BatchNormalization(axis=3)(conv2)
    #BR2
    conv2b = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2b = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(conv2b)
    conv2 = add([conv2,conv2b])
    conv2 = BatchNormalization(axis=3)(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)#16*16*128 
    
    conv31 = Conv2D(256, [3,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)#64*64*64
    conv31 = Conv2D(256, [1,3], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv31)
    conv32 = Conv2D(256, [1,3], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv32 = Conv2D(256, [3,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv32)
    conv3 = add([conv31,conv32])
    conv3 = BatchNormalization(axis=3)(conv3)
    #BR3
    conv3b = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3b = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(conv3b)
    conv3 = add([conv3,conv3b])
    conv3 = BatchNormalization(axis=3)(conv3)

    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)#没有使用dropout 8*8*256

    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)     
    conv4_1 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4_1 = Dropout(0.5)(conv4_1)
    conv4_2 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop4_1)     
    conv4_2 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4_2)
    drop4_3 = Dropout(0.5)(conv4_2)    

    print("drop4_3："+str(drop4_3.shape))
    up6 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(drop4_3)#(None, 16, 16, 256) 
    print("up6："+str(up6.shape))
    up6 = BatchNormalization(axis=3)(up6)#(None, 16, 16, 256) 
    up6 = Activation('relu')(up6)#(None, 16, 16, 256) 

    print("drop3："+str(drop3.shape)+"up6："+str(up6.shape))
    #DANet1
    #Position Attention1
    PB = Reshape(target_shape=(16*16,256))(up6) #N*C  
    PC = Reshape(target_shape=(256,16*16))(up6) #C*N 
    # print(PB.shape)
    PS = Dot_layer([PB,PC])
    PS = Activation('softmax')(PS)    
    PF = Dot_layer([PS,PB])
    PF = Reshape(target_shape=(16,16,256))(PF)#H*W*C 
    PE = Add()([PF,up6])
    PE = BatchNormalization(axis=3)(PE)

    #Channel Attention1
    CB = Reshape(target_shape=(16*16,256))(drop3)#N*C   
    CC = Reshape(target_shape=(256,16*16))(drop3)#C*N
    CS = Dot_layer([CC,CB])
    CS = Activation('softmax')(CS)
    CF = Dot_layer([CB,CS])
    CF = Reshape(target_shape=(16,16,256))(CF)#H*W*C
    CE = Add()([CF,drop3])
    CE = BatchNormalization(axis=3)(CE)

    x1 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 256))(CE)#(None, 1, 16, 16, 25    
    x2 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 256))(PE)#(None, 1, 16, 16, 25
    merge6  = concatenate([x1,x2], axis = 1)# (None, 2, 16, 16, 25
    merge6 = ConvLSTM2D(filters = 128, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge6)#(None, 16, 16, 128)
            
    conv6 = SeparableConv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)#(None, 16, 16, 256) 
    conv6 = SeparableConv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)#(None, 16, 16, 256) 

    up7 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv6)#(None, 32, 32, 128)
    up7 = BatchNormalization(axis=3)(up7)#(None, 32, 32, 128)
    up7 = Activation('relu')(up7)#(None, 32, 32, 128)

    #DANet2
    #Position Attention1
    PB = Reshape(target_shape=(32*32,128))(up7) #N*C  
    PC = Reshape(target_shape=(128,32*32))(up7) #C*N 
    # print(PB.shape)
    PS = Dot_layer([PB,PC])
    PS = Activation('softmax')(PS)    
    PF = Dot_layer([PS,PB])
    PF = Reshape(target_shape=(32,32,128))(PF)#H*W*C 
    PE = Add()([PF,up7])
    PE = BatchNormalization(axis=3)(PE)

    #Channel Attention1
    CB = Reshape(target_shape=(32*32,128))(conv2)#N*C   
    CC = Reshape(target_shape=(128,32*32))(conv2)#C*N
    CS = Dot_layer([CC,CB])
    CS = Activation('softmax')(CS)
    CF = Dot_layer([CB,CS])
    CF = Reshape(target_shape=(32,32,128))(CF)#H*W*C
    CE = Add()([CF,conv2])
    CE = BatchNormalization(axis=3)(CE)

    x1 = Reshape(target_shape=(1, np.int32(N/2), np.int32(N/2), 128))(CE)#(None, 1, 32, 32, 128    
    x2 = Reshape(target_shape=(1, np.int32(N/2), np.int32(N/2), 128))(PE)#(None, 1, 32, 32, 128
    merge7  = concatenate([x1,x2], axis = 1) #(None, 2, 32, 32, 12
    merge7 = ConvLSTM2D(filters = 64, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge7)#(None, 32, 32, 64) 
        
    conv7 = SeparableConv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)#(None, 32, 32, 128)
    conv7 = SeparableConv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)#(None, 32, 32, 128)
    
    up8 = Conv2DTranspose(64, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv7)#(None, 64, 64, 64) 
    up8 = BatchNormalization(axis=3)(up8)#(None, 64, 64, 64) 
    up8 = Activation('relu')(up8)#64*64*64    

    #DANet2
    #Position Attention1
    PB = Reshape(target_shape=(64*64,64))(up8) #N*C  
    PC = Reshape(target_shape=(64,64*64))(up8) #C*N 
    # print(PB.shape)
    PS = Dot_layer([PB,PC])
    # PS = Dot()([PB,PC])#N*N
    PS = Activation('softmax')(PS)    
    PF = Dot_layer([PS,PB])
    # PF = Dot()([PS,PB])
    PF = Reshape(target_shape=(64,64,64))(PF)#H*W*C 
    PE = Add()([PF,up8])
    PE = BatchNormalization(axis=3)(PE)

    #Channel Attention1
    CB = Reshape(target_shape=(64*64,64))(conv1)#N*C   
    CC = Reshape(target_shape=(64,64*64))(conv1)#C*N
    # CS = Dot()([CC,CB])#C*C
    CS = Dot_layer([CC,CB])
    CS = Activation('softmax')(CS)
    # CF = Dot()([CB,CS])
    CF = Dot_layer([CB,CS])
    CF = Reshape(target_shape=(64,64,64))(CF)#H*W*C
    CE = Add()([CF,conv1])
    CE = BatchNormalization(axis=3)(CE)

    x1 = Reshape(target_shape=(1,  N, N, 64))(CE)#(None, 1,  N, N, 64    
    x2 = Reshape(target_shape=(1, N, N, 64))(PE)#(None, 1,  N, N, 64
    merge8  = concatenate([x1,x2], axis = 1) #(None, 2, 64, 64, 64
    merge8 = ConvLSTM2D(filters = 32, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge8)#(None, 64, 64, 32)   
    
    conv8 = SeparableConv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)#(None, 64, 64, 64) 
    conv8 = SeparableConv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)#(None, 64, 64, 64) 
    conv8 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)#(None, 64, 64, 2) 
    conv9 = Conv2D(1, 1, activation = 'sigmoid')(conv8)#(None, 64, 64, 1)
    model = Model(inputs =inputs,outputs = conv9)    
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

def ConvLSTM_Mnet(input_size = (256,256,1)):
    N = input_size[0]
    inputs = Input(input_size) #(None, 64, 64, 1) 
    # bjs = Input(input_size) 
    s_img_2 = AveragePooling2D(pool_size=(2, 2), name='in_scale2')(inputs)#256*256*1
    s_img_3 = AveragePooling2D(pool_size=(2, 2), name='in_scale3')(s_img_2)#128*128*1
     
    # jg1 = multiply([inputs,bjs])
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)  
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) 
     
    # jg2 = multiply([pool1,scale_img_2])
    s2 = SeparableConv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(s_img_2)
    is2 = Concatenate()([pool1,s2])

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(is2)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) 
     
     # jg3 = multiply([pool2,scale_img_3])
    s3 = SeparableConv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(s_img_3)
    is3 = Concatenate()([pool2,s3])

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(is3)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    drop3 = Dropout(0.5)(conv3)
    # drop3 = GaussianDropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)#没有使用dropout

    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)     
    conv4_1 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4_1 = Dropout(0.5)(conv4_1)
    conv4_2 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop4_1)     
    conv4_2 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4_2)
    drop4_3 = Dropout(0.5)(conv4_2)


    up6 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(drop4_3)#(None, 16, 16, 256) 
    up6 = BatchNormalization(axis=3)(up6)#(None, 16, 16, 256) 
    up6 = Activation('relu')(up6)#(None, 16, 16, 256) 

    # print("drop3："+str(drop3.shape)+"up6："+str(up6.shape))
    x1 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 256))(drop3)#(None, 1, 16, 16, 25
    x2 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 256))(up6)#(None, 1, 16, 16, 25
    merge6  = concatenate([x1,x2], axis = 1)# (None, 2, 16, 16, 25
    merge6 = ConvLSTM2D(filters = 128, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge6)#(None, 16, 16, 128)
               
    conv6 = SeparableConv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)#(None, 16, 16, 256) 
    conv6 = SeparableConv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)#(None, 16, 16, 256) 

    up7 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv6)#(None, 32, 32, 128)
    up7 = BatchNormalization(axis=3)(up7)#(None, 32, 32, 128)
    up7 = Activation('relu')(up7)#(None, 32, 32, 128)

    x1 = Reshape(target_shape=(1, np.int32(N/2), np.int32(N/2), 128))(conv2)#(None, 1, 32, 32, 12
    x2 = Reshape(target_shape=(1, np.int32(N/2), np.int32(N/2), 128))(up7)#(None, 1, 32, 32, 12
    merge7  = concatenate([x1,x2], axis = 1) #(None, 2, 32, 32, 12
    merge7 = ConvLSTM2D(filters = 64, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge7)#(None, 32, 32, 64) 
          
    conv7 = SeparableConv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)#(None, 32, 32, 128)
    conv7 = SeparableConv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)#(None, 32, 32, 128)
     
    up8 = Conv2DTranspose(64, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv7)#(None, 64, 64, 64) 
    up8 = BatchNormalization(axis=3)(up8)#(None, 64, 64, 64) 
    up8 = Activation('relu')(up8)#64*64*64    

    x1 = Reshape(target_shape=(1, N, N, 64))(conv1)# (None, 1, 64, 64, 64
    x2 = Reshape(target_shape=(1, N, N, 64))(up8)# (None, 1, 64, 64, 64
    merge8  = concatenate([x1,x2], axis = 1) #(None, 2, 64, 64, 64
    merge8 = ConvLSTM2D(filters = 32, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge8)#(None, 64, 64, 32)   
     
    conv8 = SeparableConv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)#(None, 64, 64, 64) 
    conv8 = SeparableConv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)#(None, 64, 64, 64) 

    conv8 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)#(None, 64, 64, 2) 
    conv9 = Conv2D(1, 1, activation = 'sigmoid')(conv8)#(None, 64, 64, 1)

    model = Model(inputs =inputs,outputs = conv9)   
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model