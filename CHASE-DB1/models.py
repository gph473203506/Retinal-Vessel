from __future__ import division
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.utils.vis_utils import plot_model as plot
from keras.optimizers import SGD
from keras.optimizers import *
from keras.layers import *     
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"



def BCDU_net_D3(input_size = (256,256,1)):
    N = input_size[0]
    inputs = Input(input_size) 
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)  
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # pool1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',strides = 2)(conv1) 

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # pool2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',strides = 2)(conv2) 

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    drop3 = Dropout(0.5)(conv3)
    # drop3 = GaussianDropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)#没有使用dropout
    # pool3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',strides = 2)(drop3)
     
    # D1
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)     
    conv4_1 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    # drop4_1 = Dropout(0.5)(conv4_1)
    drop4_1 = GaussianDropout(0.5)(conv4_1)
    # D2
    conv4_2 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop4_1)     
    conv4_2 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4_2)
    # conv4_2 = Dropout(0.5)(conv4_2)
    conv4_2 = GaussianDropout(0.5)(conv4_2)
    # D3
    merge_dense = concatenate([conv4_2,drop4_1], axis = 3)
    conv4_3 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge_dense)     
    conv4_3 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4_3)
    # drop4_3 = Dropout(0.5)(conv4_3)
    drop4_3 = GaussianDropout(0.5)(conv4_3)
    
    
    up6 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(drop4_3)
    up6 = BatchNormalization(axis=3)(up6)
    up6 = Activation('relu')(up6)

    x1 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 256))(drop3)
    x2 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 256))(up6)
    merge6  = concatenate([x1,x2], axis = 1) 
    merge6 = ConvLSTM2D(filters = 128, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge6)
            
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv6)
    up7 = BatchNormalization(axis=3)(up7)
    up7 = Activation('relu')(up7)

    x1 = Reshape(target_shape=(1, np.int32(N/2), np.int32(N/2), 128))(conv2)
    x2 = Reshape(target_shape=(1, np.int32(N/2), np.int32(N/2), 128))(up7)
    merge7  = concatenate([x1,x2], axis = 1) 
    merge7 = ConvLSTM2D(filters = 64, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge7)
        
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2DTranspose(64, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv7)
    up8 = BatchNormalization(axis=3)(up8)
    up8 = Activation('relu')(up8)    

    x1 = Reshape(target_shape=(1, N, N, 64))(conv1)
    x2 = Reshape(target_shape=(1, N, N, 64))(up8)
    merge8  = concatenate([x1,x2], axis = 1) 
    merge8 = ConvLSTM2D(filters = 32, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge8)    
    
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    # conv8 = concatenate([up8,conv8])
    conv8 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv9 = Conv2D(1, 1, activation = 'sigmoid')(conv8)

    model = Model(inputs =inputs,outputs = conv9)
    # model = Model(input = inputs, output = conv9)
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model
        

def BCDU_net_D1(input_size = (256,256,1)):
    N = input_size[0]
    inputs = Input(input_size) 
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
  
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # D1
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)     
    conv4_1 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4_1 = Dropout(0.5)(conv4_1)   
    
    up6 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(drop4_1)
    up6 = BatchNormalization(axis=3)(up6)
    up6 = Activation('relu')(up6)

    x1 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 256))(drop3)
    x2 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 256))(up6)
    merge6  = concatenate([x1,x2], axis = 1) 
    merge6 = ConvLSTM2D(filters = 128, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge6)
            
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv6)
    up7 = BatchNormalization(axis=3)(up7)
    up7 = Activation('relu')(up7)

    x1 = Reshape(target_shape=(1, np.int32(N/2), np.int32(N/2), 128))(conv2)
    x2 = Reshape(target_shape=(1, np.int32(N/2), np.int32(N/2), 128))(up7)
    merge7  = concatenate([x1,x2], axis = 1) 
    merge7 = ConvLSTM2D(filters = 64, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge7)
        
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2DTranspose(64, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv7)
    up8 = BatchNormalization(axis=3)(up8)
    up8 = Activation('relu')(up8)    

    x1 = Reshape(target_shape=(1, N, N, 64))(conv1)
    x2 = Reshape(target_shape=(1, N, N, 64))(up8)
    merge8  = concatenate([x1,x2], axis = 1) 
    merge8 = ConvLSTM2D(filters = 32, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge8)    
    
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = Multiply([up8,conv8])
    conv8 = Add([up8,conv8])
    conv8 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv9 = Conv2D(1, 1, activation = 'sigmoid')(conv8)

    model = Model(input = inputs, output = conv9)
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model   
    
    
def BCDU_net_D2(input_size = (256,256,1)):
    N = input_size[0]
    inputs = Input(input_size) 

    #Spatial Path
    conv111 = Conv2D(64, [3,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)#64*64*64
    conv111 = Conv2D(64, [1,3], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv111)
    conv112 = Conv2D(64, [1,3], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv112 = Conv2D(64, [3,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv112)
    conv11 = add([conv111,conv112])
    conv11 = BatchNormalization(axis=3)(conv11)
    pool11 = MaxPooling2D(pool_size=(2, 2))(conv11)#32*32*64  

    conv121 = Conv2D(128, [3,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool11)#32*32*128
    conv121 = Conv2D(128, [1,3], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv121)
    conv122 = Conv2D(128, [1,3], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool11)
    conv122 = Conv2D(128, [3,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv122)
    conv12 = add([conv121,conv122])
    conv12 = BatchNormalization(axis=3)(conv12)
    pool12 = MaxPooling2D(pool_size=(2, 2))(conv12)#16*16*128

    conv131 = Conv2D(256, [3,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool12)#64*64*64
    conv131 = Conv2D(256, [1,3], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv131)
    conv132 = Conv2D(256, [1,3], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool12)
    conv132 = Conv2D(256, [3,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv132)
    conv13 = add([conv131,conv132])
    conv13 = BatchNormalization(axis=3)(conv13)
    drop13 = Dropout(0.5)(conv13)
    pool13 = MaxPooling2D(pool_size=(2, 2))(drop13)#没有使用dropout 8*8*256
    print("drop13"+str(drop13.shape)+"pool13"+str(pool13.shape))

    # conv11 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv11)  
    # pool11 = MaxPooling2D(pool_size=(2, 2))(conv11)#32*32*64  
    # conv12 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool11)#32*32*128
    # conv12 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv12)
    # pool12 = MaxPooling2D(pool_size=(2, 2))(conv12)#16*16*128
    # conv13 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool12)#16*16*256
    # conv13 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv13)
    # drop13 = Dropout(0.5)(conv13)
    # pool13 = MaxPooling2D(pool_size=(2, 2))(drop13)#没有使用dropout 8*8*256

    #Context Path
    conv211 =Conv2D(64, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv212 =Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv211)
    conv213 =Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv212)#64*64*64
    merge21 = concatenate([conv211,conv212,conv213], axis = 3)#64*64*192
    pool21 = MaxPooling2D(pool_size=(2, 2))(merge21)#32*32*192
    conv221 =Conv2D(128, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool21)#32*32*128
    conv222 =Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv221)
    conv223 =Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv222)
    merge22 = concatenate([conv221,conv222,conv223], axis = 3)#32*32*384
    pool22 = MaxPooling2D(pool_size=(2, 2))(merge22)#16*16*384
    conv231 =Conv2D(256, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool22)#16*16*256
    conv232 =Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv231)#16*16*256
    conv233 =Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv232)
    merge23 = concatenate([conv231,conv232,conv233], axis = 3)#16*16*768
    pool23 = MaxPooling2D(pool_size=(2, 2))(merge23)#8*8*768
    print("merge23: "+str(merge23.shape)+"pool23："+str(pool23.shape))
    
    pool23 =Conv2D(256, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool23)#8*8*256
    pool3 = concatenate([pool13,pool23],axis=3)#8*8*512

    dd3 = concatenate([drop13,merge23],axis=3)
    dd3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dd3)
    dd2 = concatenate([conv12,merge22],axis=3)
    dd2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dd2)
    dd1 = concatenate([conv11,merge21],axis=3)
    dd1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dd1)

    # D1
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)     
    conv4_1 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4_1 = Dropout(0.5)(conv4_1)
    # D2
    conv4_2 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop4_1)     
    conv4_2 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4_2)
    conv4_2 = Dropout(0.5)(conv4_2)
    # D3
    merge_dense = concatenate([conv4_2,drop4_1], axis = 3)
    conv4_3 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge_dense)     
    conv4_3 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4_3)
    drop4_3 = Dropout(0.5)(conv4_3)
    
    up6 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(drop4_3)
    up6 = BatchNormalization(axis=3)(up6)
    up6 = Activation('relu')(up6)

    x1 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 256))(dd3)
    x2 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 256))(up6)
    merge6  = concatenate([x1,x2], axis = 1) 
    merge6 = ConvLSTM2D(filters = 128, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge6)
            
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv6)
    up7 = BatchNormalization(axis=3)(up7)
    up7 = Activation('relu')(up7)

    x1 = Reshape(target_shape=(1, np.int32(N/2), np.int32(N/2), 128))(dd2)
    x2 = Reshape(target_shape=(1, np.int32(N/2), np.int32(N/2), 128))(up7)
    merge7  = concatenate([x1,x2], axis = 1) 
    merge7 = ConvLSTM2D(filters = 64, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge7)
        
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2DTranspose(64, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv7)
    up8 = BatchNormalization(axis=3)(up8)
    up8 = Activation('relu')(up8)    

    x1 = Reshape(target_shape=(1, N, N, 64))(dd1)
    x2 = Reshape(target_shape=(1, N, N, 64))(up8)
    merge8  = concatenate([x1,x2], axis = 1) 
    merge8 = ConvLSTM2D(filters = 32, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge8)    
    
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    # conv8 = concatenate([up8,conv8])
    conv8 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv9 = Conv2D(1, 1, activation = 'sigmoid')(conv8)

    model = Model(inputs =inputs,outputs = conv9)
    # model = Model(input = inputs, output = conv9)
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model    


def GCN_BCDU_net_D3(input_size = (256,256,1)):
    N = input_size[0]
    inputs = Input(input_size) #(None, 64, 64, 1) 
    # conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)#(None, 64, 64, 64)
    # conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)#(None, 64, 64, 64)  
    # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)#(None, 32, 32, 64)  
    conv11 = Conv2D(64, [3,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)#64*64*64
    conv11 = Conv2D(64, [1,3], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv11)
    conv12 = Conv2D(64, [1,3], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv12 = Conv2D(64, [3,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv12)
    conv1 = add([conv11,conv12])
    conv1 = BatchNormalization(axis=3)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)#32*32*64  
    
    
    # conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)#(None, 32, 32, 128)
    # conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)#(None, 32, 32, 128)
    # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)#(None, 16, 16, 128)
    conv21 = Conv2D(128, [3,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)#32*32*128
    conv21 = Conv2D(128, [1,3], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv21)
    conv22 = Conv2D(128, [1,3], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv22 = Conv2D(128, [3,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv22)
    conv2 = add([conv21,conv22])
    conv2 = BatchNormalization(axis=3)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)#16*16*128 
    
    # conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)#(None, 16, 16, 256)
    # conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)#(None, 16, 16, 256)
    # drop3 = Dropout(0.5)(conv3)#16*16*256
    # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)#(None, 8, 8, 256)
    conv31 = Conv2D(256, [3,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)#64*64*64
    conv31 = Conv2D(256, [1,3], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv31)
    conv32 = Conv2D(256, [1,3], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv32 = Conv2D(256, [3,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv32)
    conv3 = add([conv31,conv32])
    conv3 = BatchNormalization(axis=3)(conv3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)#没有使用dropout 8*8*256


    # D1
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)#(None, 8, 8, 512)     
    conv41 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)#(None, 8, 8, 512)
    # conv411 = Conv2D(512, [7,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    # conv411 = Conv2D(512, [1,7], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv411)
    # conv412 = Conv2D(512, [1,7], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)   
    # conv412 = Conv2D(512, [7,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv412)
    # conv41 = add([conv411,conv412])
    # conv41 = BatchNormalization(axis=3)(conv41) 
    drop41 = Dropout(0.5)(conv41)
    # D2
    conv42 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop41)#(None, 8, 8, 512)      
    conv42 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv42)#(None, 8, 8, 512) 
    # conv421 = Conv2D(512, [7,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv41)
    # conv421 = Conv2D(512, [1,7], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv421)
    # conv422 = Conv2D(512, [1,7], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv41)   
    # conv422 = Conv2D(512, [7,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv422)
    # conv42 = add([conv421,conv422])
    # conv42 = BatchNormalization(axis=3)(conv42) 
    conv42 = Dropout(0.5)(conv42)#(None, 8, 8, 512) 
    # D3
    merge_dense = concatenate([conv42,drop41], axis = 3)#(None, 8, 8, 1024)
    conv43 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge_dense)#(None, 8, 8, 512)     
    conv43 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv43)#(None, 8, 8, 512)
    # conv431 = Conv2D(512, [7,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv42)
    # conv431 = Conv2D(512, [1,7], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv431)
    # conv432 = Conv2D(512, [1,7], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv42)   
    # conv432 = Conv2D(512, [7,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv432)
    # conv43 = add([conv431,conv432])
    # conv43 = BatchNormalization(axis=3)(conv43) 
    drop4_3 = Dropout(0.5)(conv43)#(None, 8, 8, 512)


    up6 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(drop4_3)#(None, 16, 16, 256) 
    up6 = BatchNormalization(axis=3)(up6)#(None, 16, 16, 256) 
    up6 = Activation('relu')(up6)#(None, 16, 16, 256) 

    print("drop3："+str(drop3.shape)+"up6："+str(up6.shape))
    x1 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 256))(drop3)#(None, 1, 16, 16, 25
    x2 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 256))(up6)#(None, 1, 16, 16, 25
    merge6  = concatenate([x1,x2], axis = 1)# (None, 2, 16, 16, 25
    merge6 = ConvLSTM2D(filters = 128, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge6)#(None, 16, 16, 128)
            
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)#(None, 16, 16, 256) 
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)#(None, 16, 16, 256) 
    # conv61 = Conv2D(256, [7,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    # conv61 = Conv2D(256, [1,7], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv61)
    # conv62 = Conv2D(256, [1,7], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)   
    # conv62 = Conv2D(256, [7,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv62)
    # conv6 = add([conv61,conv62])
    # conv6 = BatchNormalization(axis=3)(conv6) 

    up7 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv6)#(None, 32, 32, 128)
    up7 = BatchNormalization(axis=3)(up7)#(None, 32, 32, 128)
    up7 = Activation('relu')(up7)#(None, 32, 32, 128)

    x1 = Reshape(target_shape=(1, np.int32(N/2), np.int32(N/2), 128))(conv2)#(None, 1, 32, 32, 12
    x2 = Reshape(target_shape=(1, np.int32(N/2), np.int32(N/2), 128))(up7)#(None, 1, 32, 32, 12
    merge7  = concatenate([x1,x2], axis = 1) #(None, 2, 32, 32, 12
    merge7 = ConvLSTM2D(filters = 64, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge7)#(None, 32, 32, 64) 
        
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)#(None, 32, 32, 128)
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)#(None, 32, 32, 128)
    # conv71 = Conv2D(128, [7,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    # conv71 = Conv2D(128, [1,7], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv71)
    # conv72 = Conv2D(128, [1,7], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)   
    # conv72 = Conv2D(128, [7,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv72)
    # conv7 = add([conv71,conv72])
    # conv7 = BatchNormalization(axis=3)(conv7) 
    
    up8 = Conv2DTranspose(64, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv7)#(None, 64, 64, 64) 
    up8 = BatchNormalization(axis=3)(up8)#(None, 64, 64, 64) 
    up8 = Activation('relu')(up8)#64*64*64    

    x1 = Reshape(target_shape=(1, N, N, 64))(conv1)# (None, 1, 64, 64, 64
    x2 = Reshape(target_shape=(1, N, N, 64))(up8)# (None, 1, 64, 64, 64
    merge8  = concatenate([x1,x2], axis = 1) #(None, 2, 64, 64, 64
    merge8 = ConvLSTM2D(filters = 32, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge8)#(None, 64, 64, 32)   
    
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)#(None, 64, 64, 64) 
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)#(None, 64, 64, 64) 
    # conv8 = concatenate([up8,conv8])
    conv8 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)#(None, 64, 64, 2) 
    conv9 = Conv2D(1, 1, activation = 'sigmoid')(conv8)#(None, 64, 64, 1)

    # model = Model(input = inputs, output = conv9)
    model = Model(inputs =inputs,outputs = conv9)   
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model


def GCN_BR_BCDU_net_D3(input_size = (256,256,1)):
    N = input_size[0]
    inputs = Input(input_size) #(None, 64, 64, 1) 
    # conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)#(None, 64, 64, 64)
    # conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)#(None, 64, 64, 64)  
    # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)#(None, 32, 32, 64)  
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
    
    
    # conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)#(None, 32, 32, 128)
    # conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)#(None, 32, 32, 128)
    # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)#(None, 16, 16, 128)
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
    
    # conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)#(None, 16, 16, 256)
    # conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)#(None, 16, 16, 256)
    # drop3 = Dropout(0.5)(conv3)#16*16*256
    # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)#(None, 8, 8, 256)
    conv31 = Conv2D(256, [3,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)#64*64*64
    conv31 = Conv2D(256, [1,3], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv31)
    conv32 = Conv2D(256, [1,3], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv32 = Conv2D(256, [3,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv32)
    conv3 = add([conv31,conv32])
    conv3 = BatchNormalization(axis=3)(conv3)
    #BR2
    conv3b = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3b = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(conv3b)
    conv3 = add([conv3,conv3b])
    conv3 = BatchNormalization(axis=3)(conv3)

    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)#没有使用dropout 8*8*256


    # D1
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)#(None, 8, 8, 512)     
    conv41 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)#(None, 8, 8, 512)
    # conv411 = Conv2D(512, [7,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    # conv411 = Conv2D(512, [1,7], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv411)
    # conv412 = Conv2D(512, [1,7], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)   
    # conv412 = Conv2D(512, [7,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv412)
    # conv41 = add([conv411,conv412])
    # conv41 = BatchNormalization(axis=3)(conv41) 
    drop41 = Dropout(0.5)(conv41)
    # D2
    conv42 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop41)#(None, 8, 8, 512)      
    conv42 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv42)#(None, 8, 8, 512) 
    # conv421 = Conv2D(512, [7,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv41)
    # conv421 = Conv2D(512, [1,7], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv421)
    # conv422 = Conv2D(512, [1,7], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv41)   
    # conv422 = Conv2D(512, [7,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv422)
    # conv42 = add([conv421,conv422])
    # conv42 = BatchNormalization(axis=3)(conv42) 
    conv42 = Dropout(0.5)(conv42)#(None, 8, 8, 512) 
    # D3
    merge_dense = concatenate([conv42,drop41], axis = 3)#(None, 8, 8, 1024)
    conv43 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge_dense)#(None, 8, 8, 512)     
    conv43 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv43)#(None, 8, 8, 512)
    # conv431 = Conv2D(512, [7,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv42)
    # conv431 = Conv2D(512, [1,7], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv431)
    # conv432 = Conv2D(512, [1,7], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv42)   
    # conv432 = Conv2D(512, [7,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv432)
    # conv43 = add([conv431,conv432])
    # conv43 = BatchNormalization(axis=3)(conv43) 
    drop4_3 = Dropout(0.5)(conv43)#(None, 8, 8, 512)


    up6 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(drop4_3)#(None, 16, 16, 256) 
    up6 = BatchNormalization(axis=3)(up6)#(None, 16, 16, 256) 
    up6 = Activation('relu')(up6)#(None, 16, 16, 256) 

    print("drop3："+str(drop3.shape)+"up6："+str(up6.shape))
    x1 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 256))(drop3)#(None, 1, 16, 16, 25
    x2 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 256))(up6)#(None, 1, 16, 16, 25
    merge6  = concatenate([x1,x2], axis = 1)# (None, 2, 16, 16, 25
    merge6 = ConvLSTM2D(filters = 128, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge6)#(None, 16, 16, 128)
            
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)#(None, 16, 16, 256) 
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)#(None, 16, 16, 256) 
    # conv61 = Conv2D(256, [7,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    # conv61 = Conv2D(256, [1,7], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv61)
    # conv62 = Conv2D(256, [1,7], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)   
    # conv62 = Conv2D(256, [7,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv62)
    # conv6 = add([conv61,conv62])
    # conv6 = BatchNormalization(axis=3)(conv6) 

    up7 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv6)#(None, 32, 32, 128)
    up7 = BatchNormalization(axis=3)(up7)#(None, 32, 32, 128)
    up7 = Activation('relu')(up7)#(None, 32, 32, 128)

    x1 = Reshape(target_shape=(1, np.int32(N/2), np.int32(N/2), 128))(conv2)#(None, 1, 32, 32, 12
    x2 = Reshape(target_shape=(1, np.int32(N/2), np.int32(N/2), 128))(up7)#(None, 1, 32, 32, 12
    merge7  = concatenate([x1,x2], axis = 1) #(None, 2, 32, 32, 12
    merge7 = ConvLSTM2D(filters = 64, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge7)#(None, 32, 32, 64) 
        
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)#(None, 32, 32, 128)
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)#(None, 32, 32, 128)
    # conv71 = Conv2D(128, [7,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    # conv71 = Conv2D(128, [1,7], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv71)
    # conv72 = Conv2D(128, [1,7], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)   
    # conv72 = Conv2D(128, [7,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv72)
    # conv7 = add([conv71,conv72])
    # conv7 = BatchNormalization(axis=3)(conv7) 
    
    up8 = Conv2DTranspose(64, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv7)#(None, 64, 64, 64) 
    up8 = BatchNormalization(axis=3)(up8)#(None, 64, 64, 64) 
    up8 = Activation('relu')(up8)#64*64*64    

    x1 = Reshape(target_shape=(1, N, N, 64))(conv1)# (None, 1, 64, 64, 64
    x2 = Reshape(target_shape=(1, N, N, 64))(up8)# (None, 1, 64, 64, 64
    merge8  = concatenate([x1,x2], axis = 1) #(None, 2, 64, 64, 64
    merge8 = ConvLSTM2D(filters = 32, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge8)#(None, 64, 64, 32)   
    
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)#(None, 64, 64, 64) 
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)#(None, 64, 64, 64) 
    # conv8 = concatenate([up8,conv8])
    conv8 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)#(None, 64, 64, 2) 
    conv9 = Conv2D(1, 1, activation = 'sigmoid')(conv8)#(None, 64, 64, 1)

    # model = Model(input = inputs, output = conv9)
    model = Model(inputs =inputs,outputs = conv9)   
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

def BR_BCDU_net_D3(input_size = (256,256,1)):
    N = input_size[0]
    inputs = Input(input_size) #(None, 64, 64, 1) 
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)#(None, 64, 64, 64)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)#(None, 64, 64, 64)  
    # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)#(None, 32, 32, 64)  
    # conv11 = Conv2D(64, [3,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)#64*64*64
    # conv11 = Conv2D(64, [1,3], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv11)
    # conv12 = Conv2D(64, [1,3], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    # conv12 = Conv2D(64, [3,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv12)
    # conv1 = add([conv11,conv12])
    # conv1 = BatchNormalization(axis=3)(conv1)
    #BR1
    conv1b = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1b = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(conv1b)
    conv1 = add([conv1,conv1b])
    conv1 = BatchNormalization(axis=3)(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)#32*32*64  
    
    
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)#(None, 32, 32, 128)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)#(None, 32, 32, 128)
    # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)#(None, 16, 16, 128)
    # conv21 = Conv2D(128, [3,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)#32*32*128
    # conv21 = Conv2D(128, [1,3], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv21)
    # conv22 = Conv2D(128, [1,3], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    # conv22 = Conv2D(128, [3,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv22)
    # conv2 = add([conv21,conv22])
    # conv2 = BatchNormalization(axis=3)(conv2)
    #BR2
    conv2b = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2b = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(conv2b)
    conv2 = add([conv2,conv2b])
    conv2 = BatchNormalization(axis=3)(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)#16*16*128 
    
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)#(None, 16, 16, 256)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)#(None, 16, 16, 256)
    # drop3 = Dropout(0.5)(conv3)#16*16*256
    # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)#(None, 8, 8, 256)
    # conv31 = Conv2D(256, [3,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)#64*64*64
    # conv31 = Conv2D(256, [1,3], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv31)
    # conv32 = Conv2D(256, [1,3], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    # conv32 = Conv2D(256, [3,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv32)
    # conv3 = add([conv31,conv32])
    conv3 = BatchNormalization(axis=3)(conv3)
    #BR3
    conv3b = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3b = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(conv3b)
    conv3 = add([conv3,conv3b])
    conv3 = BatchNormalization(axis=3)(conv3)

    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)#没有使用dropout 8*8*256


    # D1
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)#(None, 8, 8, 512)     
    conv41 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)#(None, 8, 8, 512)
    # conv411 = Conv2D(512, [7,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    # conv411 = Conv2D(512, [1,7], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv411)
    # conv412 = Conv2D(512, [1,7], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)   
    # conv412 = Conv2D(512, [7,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv412)
    # conv41 = add([conv411,conv412])
    # conv41 = BatchNormalization(axis=3)(conv41) 
    drop41 = Dropout(0.5)(conv41)
    # D2
    conv42 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop41)#(None, 8, 8, 512)      
    conv42 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv42)#(None, 8, 8, 512) 
    # conv421 = Conv2D(512, [7,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv41)
    # conv421 = Conv2D(512, [1,7], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv421)
    # conv422 = Conv2D(512, [1,7], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv41)   
    # conv422 = Conv2D(512, [7,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv422)
    # conv42 = add([conv421,conv422])
    # conv42 = BatchNormalization(axis=3)(conv42) 
    conv42 = Dropout(0.5)(conv42)#(None, 8, 8, 512) 
    # D3
    merge_dense = concatenate([conv42,drop41], axis = 3)#(None, 8, 8, 1024)
    conv43 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge_dense)#(None, 8, 8, 512)     
    conv43 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv43)#(None, 8, 8, 512)
    # conv431 = Conv2D(512, [7,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv42)
    # conv431 = Conv2D(512, [1,7], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv431)
    # conv432 = Conv2D(512, [1,7], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv42)   
    # conv432 = Conv2D(512, [7,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv432)
    # conv43 = add([conv431,conv432])
    # conv43 = BatchNormalization(axis=3)(conv43) 
    drop4_3 = Dropout(0.5)(conv43)#(None, 8, 8, 512)


    up6 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(drop4_3)#(None, 16, 16, 256) 
    up6 = BatchNormalization(axis=3)(up6)#(None, 16, 16, 256) 
    up6 = Activation('relu')(up6)#(None, 16, 16, 256) 

    print("drop3："+str(drop3.shape)+"up6："+str(up6.shape))
    x1 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 256))(drop3)#(None, 1, 16, 16, 25
    x2 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 256))(up6)#(None, 1, 16, 16, 25
    merge6  = concatenate([x1,x2], axis = 1)# (None, 2, 16, 16, 25
    merge6 = ConvLSTM2D(filters = 128, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge6)#(None, 16, 16, 128)
            
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)#(None, 16, 16, 256) 
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)#(None, 16, 16, 256) 
    # conv61 = Conv2D(256, [7,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    # conv61 = Conv2D(256, [1,7], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv61)
    # conv62 = Conv2D(256, [1,7], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)   
    # conv62 = Conv2D(256, [7,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv62)
    # conv6 = add([conv61,conv62])
    # conv6 = BatchNormalization(axis=3)(conv6) 

    up7 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv6)#(None, 32, 32, 128)
    up7 = BatchNormalization(axis=3)(up7)#(None, 32, 32, 128)
    up7 = Activation('relu')(up7)#(None, 32, 32, 128)

    x1 = Reshape(target_shape=(1, np.int32(N/2), np.int32(N/2), 128))(conv2)#(None, 1, 32, 32, 12
    x2 = Reshape(target_shape=(1, np.int32(N/2), np.int32(N/2), 128))(up7)#(None, 1, 32, 32, 12
    merge7  = concatenate([x1,x2], axis = 1) #(None, 2, 32, 32, 12
    merge7 = ConvLSTM2D(filters = 64, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge7)#(None, 32, 32, 64) 
        
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)#(None, 32, 32, 128)
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)#(None, 32, 32, 128)
    # conv71 = Conv2D(128, [7,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    # conv71 = Conv2D(128, [1,7], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv71)
    # conv72 = Conv2D(128, [1,7], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)   
    # conv72 = Conv2D(128, [7,1], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv72)
    # conv7 = add([conv71,conv72])
    # conv7 = BatchNormalization(axis=3)(conv7) 
    
    up8 = Conv2DTranspose(64, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv7)#(None, 64, 64, 64) 
    up8 = BatchNormalization(axis=3)(up8)#(None, 64, 64, 64) 
    up8 = Activation('relu')(up8)#64*64*64    

    x1 = Reshape(target_shape=(1, N, N, 64))(conv1)# (None, 1, 64, 64, 64
    x2 = Reshape(target_shape=(1, N, N, 64))(up8)# (None, 1, 64, 64, 64
    merge8  = concatenate([x1,x2], axis = 1) #(None, 2, 64, 64, 64
    merge8 = ConvLSTM2D(filters = 32, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge8)#(None, 64, 64, 32)   
    
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)#(None, 64, 64, 64) 
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)#(None, 64, 64, 64) 
    # conv8 = concatenate([up8,conv8])
    conv8 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)#(None, 64, 64, 2) 
    conv9 = Conv2D(1, 1, activation = 'sigmoid')(conv8)#(None, 64, 64, 1)

    # model = Model(input = inputs, output = conv9)
    model = Model(inputs =inputs,outputs = conv9)   
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model