from __future__ import division
import numpy as np
import random
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


from help_functions import load_hdf5
from help_functions import visualize
from help_functions import group_images

from pre_processing import my_PreProc

#加载原始图片，然后返回提取的补丁
def get_data_training(DRIVE_train_imgs_original,
                      DRIVE_train_groudTruth,
                      patch_height,
                      patch_width,
                      N_subimgs,
                      inside_FOV):
    train_imgs_original = load_hdf5(DRIVE_train_imgs_original)#加载原始图片
    train_masks = load_hdf5(DRIVE_train_groudTruth)#加载GT图片
    
    # train_imgs = my_PreProc(train_imgs_original)#对图片进行预处理
    train_imgs = train_imgs_original
    # train_masks = train_masks/255#0-1

    train_imgs = train_imgs[:,:,:,19:979] #从下往上剪，现在是565*565
    train_masks = train_masks[:,:,:,19:979]
    print ("train masks range (min-max): " +str(np.min(train_masks)) +' - '+str(np.max(train_masks)))
    # print(train_imgs.shape)
    # print(train_masks.shape)
    data_consistency_check(train_imgs,train_masks)
   
    # #检查masks在0-1范围内
    # print ("train masks range (min-max): " +str(np.min(train_masks)) +' - '+str(np.max(train_masks)))
    assert(np.min(train_masks)==0 and np.max(train_masks)==1)

    print ("\ntrain images/masks shape:")
    print (train_imgs.shape)
    print ("train images range (min-max): " +str(np.min(train_imgs)) +' - '+str(np.max(train_imgs)))
    print ("train masks are within 0-1\n")

    # 从完整的图像中提取训练补丁
    patches_imgs_train,patches_masks_train = extract_random(train_imgs,train_masks,patch_height,patch_width,N_subimgs,inside_FOV)
    data_consistency_check(patches_imgs_train,patches_masks_train)

    print("\ntrain PATCHES images/masks shape:")
    print (patches_imgs_train.shape)
    print ("train PATCHES images range (min-max): " +str(np.min(patches_imgs_train)) +' - '+str(np.max(patches_imgs_train)))

    return patches_imgs_train,patches_masks_train#返回训练用的patches_imgs和patches_masks


#加载原始数据并返回提取的补丁进行测试
def get_data_testing(DRIVE_test_imgs_original,DRIVE_test_groudTruth,Imgs_to_test,patch_height,patch_width):
    ###测试
    test_imgs_original = load_hdf5(DRIVE_test_imgs_original)
    test_masks = load_hdf5(DRIVE_test_groudTruth)

    test_imgs = my_PreProc(test_imgs_original)
    # test_masks = test_masks/255.0

    # test_imgs = test_imgs[:,:,9:574,:] #从下往上剪，现在是565*565
    # test_masks = test_masks[:,:,9:574,:]

    # 扩展图像和masks，这样它们就可以精确地按patch大小划分
    test_imgs = test_imgs[0:Imgs_to_test,:,:,:]
    test_masks = test_masks[0:Imgs_to_test,:,:,:]
    test_imgs =paint_border(test_imgs,patch_height,patch_width)
    test_masks = paint_border(test_masks,patch_height,patch_width)

    data_consistency_check(test_imgs,test_masks)

    assert(np.max(test_masks)==1 and np.min(test_masks)==0)

    print ("\ntest images/masks shape:")
    print(test_imgs.shape)
    print("test images range (min-max): " +str(np.min(test_imgs))+ '-'+str(np.max(test_imgs)))
    print("test masks are within 0-1\n")

    #从图片中提取测试用的patches
    patches_imgs_test = extract_ordered(test_imgs,patch_height,patch_width)
    patches_masks_test = extract_ordered(test_masks,patch_height,patch_width)
    data_consistency_check(patches_imgs_test,patches_masks_test)

    print("\ntest PATCHES images/masks shape：")
    print(patches_imgs_test.shape)
    print("test PATCHES images range (min-max)："+str(np.min(patches_imgs_test))+'-'+str(np.max(patches_imgs_test)))
    
    return patches_imgs_test,patches_masks_test#返回测试用的patches图片集和patches标记集（GT）


#加载原始数据并返回提取的补丁进行测试
#将ground truth返回到它的原始类型
def get_data_testing_overlap(DRIVE_test_imgs_original,DRIVE_test_groudTruth,Imgs_to_test,patch_height,patch_width,stride_height,stride_width):
    ##test
    test_imgs_original = load_hdf5(DRIVE_test_imgs_original)
    test_masks = load_hdf5(DRIVE_test_groudTruth)

    # test_imgs = my_PreProc(test_imgs_original)
    test_imgs = test_imgs_original
    # test_masks = test_masks/255.0
    # test_imgs = test_imgs[:,:,9:574,:] #从下往上剪，现在是565*565
    # test_masks = test_masks[:,:,9:574,:]

    # 扩展图像和masks，这样它们就可以精确地按masks大小划分
    test_imgs = test_imgs[0:Imgs_to_test,:,:,:]
    test_masks = test_masks[0:Imgs_to_test,:,:,:]
    test_imgs = paint_border_overlap(test_imgs,patch_height,patch_width,stride_height,stride_width)

    #检查masks的值在0-1之间
    assert(np.max(test_masks)==1 and np.min(test_masks)==0)

    print("\ntest images shape:")
    print(test_imgs.shape)
    print ("\ntest mask shape:")
    print(test_masks.shape)
    print ("test images range (min-max): " +str(np.min(test_imgs)) +' - '+str(np.max(test_imgs)))
    print("test masks are within in 0-1\n")

    #从整个图片中提取测试的patches
    patches_imgs_test = extract_ordered_overlap(test_imgs,patch_height,patch_width,stride_height,stride_width)

    print("\ntest PATCHES images shape:")
    print(patches_imgs_test.shape)
    print("test PATCHES images range (min-max)："+str(np.min(patches_imgs_test))+'-'+str(np.max(patches_imgs_test)))

    return patches_imgs_test,test_imgs.shape[2],test_imgs.shape[3],test_masks



#检查数据是否合理
def data_consistency_check(imgs,masks):
    assert(len(imgs.shape)==len(masks.shape))
    assert(imgs.shape[0]==masks.shape[0])
    assert(imgs.shape[2]==masks.shape[2])
    assert(imgs.shape[3]==masks.shape[3])
    assert(masks.shape[1]==1)
    assert(imgs.shape[1]==1 or imgs.shape[1]==3)


#在完整的训练图像中随机提取patch
#内部或全图
def extract_random(full_imgs,full_masks,patch_h,patch_w,N_patches,inside = True):
    if(N_patches%full_imgs.shape[0] != 0):
        print("N_patches: plase enter a multiple of 20")
        exit()
    assert (len(full_imgs.shape)==4 and len(full_masks.shape)==4)  #4D arrays
    assert (full_imgs.shape[1]==1 or full_imgs.shape[1]==3)  #check the channel is 1 or 3
    assert (full_masks.shape[1]==1)   #masks only black and white
    assert (full_imgs.shape[2] == full_masks.shape[2] and full_imgs.shape[3] == full_masks.shape[3])
    patches = np.empty((N_patches,full_imgs.shape[1],patch_h,patch_w))
    patches_masks = np.empty((N_patches,full_masks.shape[1],patch_h,patch_w))
    img_h = full_imgs.shape[2]#图片的高度
    img_w =full_imgs.shape[3]#图片的宽度
    #(0,0)在图像的中心
    patch_per_img = int(N_patches/full_imgs.shape[0])#每个图片的patch数，所以必须是整数
    print("patches per full image："+str(patch_per_img))
    iter_tot = 0#存下标，最后的结果就是patch的总数
    for i in range(full_imgs.shape[0]):#遍历所有图像
        k=0
        while k < patch_per_img:
            x_center = random.randint(0+int(patch_w/2),img_w-int(patch_w/2))#patch中心点，随机取，范围必须在patch_w/2---img_w-patch_w/2
            y_center = random.randint(0+int(patch_h/2),img_h-int(patch_h/2))
            if inside==True:
                if is_patch_inside_FOV(x_center,y_center,img_w,img_h,patch_h)==False:
                    continue
            patch = full_imgs[i,:,y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2)]
            patch_mask = full_masks[i,:,y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2)]
            patches[iter_tot]=patch
            patches_masks[iter_tot]=patch_mask
            iter_tot += 1
            k += 1
    print('k='+str(k))
    return patches,patches_masks



# 检查patch是否完全包含在FOV中  FOV( field of view 视野)
def is_patch_inside_FOV(x,y,img_w,img_h,patch_h):
    x_ = x-int(img_w/2)#原点(0,0)移动到图像中心
    y_ = y-int(img_h/2)#原点(0,0)移动到图像中心
    R_inside = 270 - int(patch_h * np.sqrt(2.0) / 2.0)#半径是270(来自驱动db文档)，减去补丁对角线(假设它是正方形的#这是FOV中包含完整补丁的限制
    radius = np.sqrt((x_*x_)+(y_*y_))
    if radius < R_inside:
        return True
    else:
        return False

#填充边缘
def paint_border_overlap(full_imgs,patch_h,patch_w,stride_h,stride_w):
    assert(len(full_imgs.shape)==4)
    assert(full_imgs.shape[1]==1 or full_imgs.shape[1]==3)
    img_h = full_imgs.shape[2]
    img_w = full_imgs.shape[3]
    leftover_h = (img_h-patch_h)%stride_h#剩下的h尺寸
    leftover_w = (img_w-patch_w)%stride_w#剩下的w尺寸
    if(leftover_h != 0):#改变img_h的尺寸
        print ("\nthe side H is not compatible with the selected stride of " +str(stride_h))
        print ("img_h " +str(img_h) + ", patch_h " +str(patch_h) + ", stride_h " +str(stride_h))
        print ("(img_h - patch_h) MOD stride_h: " +str(leftover_h))
        print ("So the H dim will be padded with additional " +str(stride_h - leftover_h) + " pixels")
        tmp_full_imgs = np.zeros((full_imgs.shape[0],full_imgs.shape[1],img_h+(stride_h-leftover_h),img_w))
        tmp_full_imgs[0:full_imgs.shape[0],0:full_imgs.shape[1],0:img_h,0:img_w] = full_imgs
        full_imgs = tmp_full_imgs
    if(leftover_w != 0):#改变img_w的尺寸
        print ("the side W is not compatible with the selected stride of " +str(stride_w))
        print ("img_w " +str(img_w) + ", patch_w " +str(patch_w) + ", stride_w " +str(stride_w))
        print ("(img_w - patch_w) MOD stride_w: " +str(leftover_w))
        print ("So the W dim will be padded with additional " +str(stride_w - leftover_w) + " pixels")
        tmp_full_imgs = np.zeros((full_imgs.shape[0],full_imgs.shape[1],full_imgs.shape[2],img_w+(stride_w-leftover_w)))
        tmp_full_imgs[0:full_imgs.shape[0],0:full_imgs.shape[1],0:full_imgs.shape[2],0:img_w] = full_imgs
        full_imgs = tmp_full_imgs
    print ("new full images shape: \n" +str(full_imgs.shape))
    return full_imgs


#扩大整个图片
def paint_border(data,patch_h,patch_w):
    assert(len(data.shape)==4)
    assert(data.shape[1]==1 or data.shape[1]==3)
    img_h=data.shape[2]
    img_w=data.shape[3]
    new_img_h = 0
    new_img_w = 0
    if (img_h%patch_h)==0:
        new_img_h = img_h
    else:
        new_img_h = ((int(img_h)/int(patch_h))+1)*patch_h
    if(img_w%patch_w)==0:
        new_img_w = img_w
    else:
        new_img_w = ((int(img_w)/int(patch_w))+1)*patch_w
    new_data = np.zeros((data.shape[0],data.shape[1],int(new_img_h),int(new_img_w)))
    new_data[:,:,0:img_h,0:img_w] = data[:,:,:,:]
    return new_data


#从完整图片中提取patches
def extract_ordered_overlap(full_imgs,patch_h,patch_w,stride_h,stride_w):
    assert(len(full_imgs.shape)==4)
    assert(full_imgs.shape[1]==1 or full_imgs.shape[1]==3)
    img_h = full_imgs.shape[2]
    img_w = full_imgs.shape[3]
    assert((img_h-patch_h)%stride_h==0 and (img_w-patch_w)%stride_w==0)
    N_patches_img = ((img_h-patch_h)//stride_h+1)*((img_w-patch_w)//stride_w+1)#每个图片提取的patches数
    N_patches_tot = N_patches_img*full_imgs.shape[0]#所有图片提取的patches数
    print ("Number of patches on h : "+str(((img_h-patch_h)//stride_h+1)))
    print ("Number of patches on w : " +str(((img_w-patch_w)//stride_w+1)))
    print ("number of patches per image: " +str(N_patches_img) +", totally for this dataset: " +str(N_patches_tot))
    patches = np.empty((N_patches_tot,full_imgs.shape[1],patch_h,patch_w))
    iter_tot = 0#下标，最后的结果是patches的总数
    for i in range(full_imgs.shape[0]):#循环整个图像
        for h in range((img_h-patch_h)//stride_h+1):
            for w in range((img_w-patch_w)//stride_w+1):
                patch = full_imgs[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]
                patches[iter_tot]=patch
                iter_tot+=1
    assert(iter_tot==N_patches_tot)
    return patches#返回patches的array数组


#从图片中提取patches
def extract_ordered(full_imgs,patch_h,patch_w):
    assert(len(full_imgs.shape)==4)
    assert(full_imgs.shape[1]==1 or full_imgs.shape[1]==3)
    img_h = full_imgs.shape[2]
    img_w = full_imgs.shape[3]
    N_patches_h = int(img_h/patch_h)
    if(img_h%patch_h != 0):
        print("warning："+str(N_patches_h)+"patches in height,with about" + str(img_h%patch_h) + "pixels left over")
    N_patches_w = int(img_w/patch_w)
    if(img_w%patch_w != 0):
        print("warning："+str(N_patches_w)+" patches in width, with about " +str(img_w%patch_w) +" pixels left over")
    print("number of patches per image："+str(N_patches_h*N_patches_w))
    N_patches_tot = (N_patches_h*N_patches_w)*full_imgs.shape[0]
    patches = np.empty((N_patches_tot,full_imgs.shape[1],patch_h,patch_w))

    iter_tot = 0
    for i in range(full_imgs.shape[0]):
        for h in range(N_patches_h):
            for w in range(N_patches_w):
                patch = full_imgs[i,:,h*patch_h:(h*patch_h)+patch_h,w*patch_w:(w*patch_w)+patch_w]
                patches[iter_tot]=patch
                iter_tot+=1
    assert(iter_tot==N_patches_tot)
    return patches


#通过patches构造完整的图片
def recompone_overlap(preds,img_h,img_w,stride_h,stride_w):
    assert(len(preds.shape)==4)
    assert(preds.shape[1]==1 or preds.shape[1]==3)
    patch_h = preds.shape[2]
    patch_w = preds.shape[3]
    N_patches_h = (img_h-patch_h)//stride_h+1
    N_patches_w = (img_w-patch_w)//stride_w+1
    N_patches_img = N_patches_h*N_patches_w
    print ("N_patches_h: " +str(N_patches_h))
    print ("N_patches_w: " +str(N_patches_w))
    print ("N_patches_img: " +str(N_patches_img))
    assert(preds.shape[0]%N_patches_img==0)
    N_full_imgs = preds.shape[0]//N_patches_img
    print ("According to the dimension inserted, there are " +str(N_full_imgs) +" full images (of " +str(img_h)+"x" +str(img_w) +" each)")
    full_prob = np.zeros((N_full_imgs,preds.shape[1],img_h,img_w))
    full_sum = np.zeros((N_full_imgs,preds.shape[1],img_h,img_w))

    k=0#迭代器遍历所有patches
    for i in range(N_full_imgs):
        for h in range((img_h-patch_h)//stride_h+1):
            for w in range((img_w-patch_w)//stride_w+1):
                full_prob[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]+=preds[k]
                full_sum[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]+=1
                k+=1
    assert(k==preds.shape[0])
    assert(np.min(full_sum)>=1.0)#至少有一个
    final_avg = full_prob/full_sum
    print(final_avg.shape)
    assert(np.max(final_avg)<=1.0)#像素的最大值是1.0
    assert(np.min(final_avg)>=0.0)#像素的最小值是0.0
    return final_avg


#通过patches构造完整的图片
def recompone(data,N_h,N_w):#N_h:每张图片在h维度所分成的patch数
    print('data.shape：'+str(data.shape))
    assert(data.shape[1]==1 or data.shape[1]==3)
    assert(len(data.shape)==4)
    N_patch_per_img = N_w*N_h
    print(str(data.shape[0])+'----'+str(N_patch_per_img))
    assert(data.shape[0]%N_patch_per_img == 0)
    N_full_imgs = data.shape[0]/N_patch_per_img
    patch_h = data.shape[2]
    patch_w = data.shape[3]
    N_patch_per_img = N_w*N_h
    #定义并且开始重建
    full_recomp = np.empty((int(N_full_imgs),data.shape[1],N_h*patch_h,N_w*patch_w))
    k = 0#完整图片的下标
    s = 0#单个patch的下标
    while(s<data.shape[0]):
        #重建一个
        single_recon = np.empty((data.shape[1],N_h*patch_h,N_w*patch_w))
        for h in range(N_h):
            for w in range(N_w):
                single_recon[:,h*patch_h:(h*patch_h)+patch_h,w*patch_w:(w*patch_w)+patch_w]
                s+=1
        full_recomp[k]=single_recon#一张一张重建的图片放进去
        k+=1
    assert(k==N_full_imgs)
    return full_recomp


#函数的目的是将整张图片在FOV区域外的部分设置为黑色（0）
def kill_border(data,original_imgs_border_masks):
    assert(len(data.shape)==4)
    assert(data.shape[1]==1 or data.shape[1]==3)
    height = data.shape[2]
    width = data.shape[3]
    for i in range(data.shape[0]):
        for x in range(width):
            for y in range(height):
                if inside_FOV_DRIVE(i,x,y,original_imgs_border_masks)==False:
                    data[i,:,y,x]=0.0
                    

#返回images和masks中包含在FOV中的像素
def pred_only_FOV(data_imgs,data_masks):
    assert(len(data_imgs.shape)==4 and len(data_masks.shape)==4)
    assert(data_imgs.shape[0] == data_masks.shape[0])
    assert(data_imgs.shape[2] == data_masks.shape[2])
    assert(data_imgs.shape[3] == data_masks.shape[3])
    assert(data_imgs.shape[1]==1 and data_masks.shape[1]==1)
    height = data_imgs.shape[2]
    width = data_imgs.shape[3]
    new_pred_imgs = []
    new_pred_masks = []
    for i in range(data_imgs.shape[0]):#迭代遍历所有图片
        for x in range(width):
            for y in range(height):
                new_pred_imgs.append(data_imgs[i,:,y,x])
                new_pred_masks.append(data_masks[i,:,y,x])
    new_pred_imgs = np.asarray(new_pred_imgs)
    new_pred_masks = np.asarray(new_pred_masks)
    return new_pred_imgs,new_pred_masks



def inside_FOV_DRIVE(i,x,y,DRIVE_masks):
    assert(len(DRIVE_masks.shape)==4)
    assert(DRIVE_masks.shape[1]==1)
    # DRIVE_masks =DRIVE_masks/255.0

    if(x >= DRIVE_masks.shape[3] or y >= DRIVE_masks.shape[2]):#图片比原始的更大
        return False

    if(DRIVE_masks[i,0,y,x]>0):
        return True#表示它可以正常工作
    else:
        return False    
