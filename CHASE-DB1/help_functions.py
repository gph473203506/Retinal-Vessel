from __future__ import division
import h5py
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# def load_hdf5(infile):
#     with h5py.File(infile,"r") as f:
#         return f["image"][()]
def load_hdf5(infile):#返回infile.hdf5文件的内容
  with h5py.File(infile,"r") as f:  #"with" close the file after its nested commands
    return f["image"][()]
def write_hdf5(arr,outfile):#将arr数组里边的数据存到outfile.hdf5文件中
  with h5py.File(outfile,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)       

# def write_hdf5(arr,outfile):
#     with h5py.File(outfile,"w") as f:
#         f.create_dataset("image", data=arr, dtype=arr.dtype)

def rgb2gray(rgb):#将rgb图片转换为二值图片
    assert(len(rgb.shape)==4)
    assert(rgb.shape[1]==3)
    bn_imgs = rgb[:,0,:,:]*0.299 + rgb[:,1,:,:]*0.587 + rgb[:,2,:,:]*0.114
    bn_imgs = np.reshape(bn_imgs,(rgb.shape[0],1,rgb.shape[2],rgb.shape[3]))
    return bn_imgs

#按每列对一组图像的行进行分组
def group_images(data,per_row):
    assert data.shape[0]%per_row==0
    assert (data.shape[1]==1 or data.shape[1]==3)#第二维是通道数，即是二值图像或者RGB图像  （图片数量，通道数，图片高度，图片宽度）
    data = np.transpose(data,(0,2,3,1))#调整图片的格式  
    all_stripe = []
    for i in range(int(data.shape[0]/per_row)):
        stripe = data[i*per_row]
        for k in range(i*per_row+1, i*per_row+per_row):
            stripe = np.concatenate((stripe,data[k]),axis=1)
        all_stripe.append(stripe)
    totimg = all_stripe[0]
    for i in range(1,len(all_stripe)):
        totimg = np.concatenate((totimg,all_stripe[i]),axis=0)
    return totimg


#初始化图片（as PIL image, NOT as matplotlib!）
def visualize(data,filename):
    assert (len(data.shape)==3)#height*width*channels
    img = None
    if data.shape[2]==1:#黑白图片
        data = np.reshape(data,(data.shape[0],data.shape[1]))
    if np.max(data)>1: 
        img = Image.fromarray(data.astype(np.uint8))#图像已经0-255
    else:
        img = Image.fromarray((data*255).astype(np.uint8))
    img.save(filename + '.png')
    return img


#给unet正确的准备mask
def masks_Unet(masks):
    assert(len(masks.shape)==4)
    assert(masks.shape[1]==1)
    im_h = masks.shape[2]
    im_w = masks.shape[3]
    masks = np.reshape(masks,(masks.shape[0],im_h*im_w))
    new_masks = np.empty(masks.shape[0],im_h*im_w,2)
    for i in range(masks.shape[0]):
        for j in range(im_h*im_w):
            if masks[i,j] == 0:
                new_masks[i,j,0]=1
                new_masks[i,j,1]=0
            else:
                new_masks[i,j,1]=0
                new_masks[i,j,0]=1
    return new_masks


def pred_to_imgs(pred,patch_height,patch_width,mode="original"):
    assert(len(pred.shape)==3) #3D数组:(Npatches,height*width,2)
    assert(pred.shape[2]==2) #检查分类是两种
    pred_images = np.empty((pred.shape[0],pred.shape[1]))#(Npatches,height*width)
    if mode == "original":
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                pred_images[i,pix]=pred[i,pix,1]
    elif mode == "threshold":
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                if pred[i,pix,1]>=0.5:
                    pred_images[i,pix]=1
                else:
                    pred_images[i,pix]=0
    else:
        print("mpde "+str(mode) + "not recognized, it can be 'original' or 'threshold'")
        exit()
    pred_images = np.reshape(pred_images,(pred_images.shape[0],1,patch_height,patch_width))
    return pred_images