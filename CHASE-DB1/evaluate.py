from __future__ import division
# 对GPU进行按需分配
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import models as M 
import numpy as np 
from matplotlib import pyplot as plt 

#Keras
from keras.models import load_model
from keras.models import model_from_json
from keras.models import Model

#scikit learn
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score

#help_functions.py
from help_functions import *
#extract_patches.py
from extract_patches import paint_border
from extract_patches import get_data_testing
from extract_patches import get_data_testing_overlap
from extract_patches import recompone
from extract_patches import recompone_overlap
from extract_patches import kill_border
from extract_patches import pred_only_FOV

from pre_processing import my_PreProc



#========= 要从中读取的配置文件 =======
#===========================================
#在不变量或本地变量上运行训练
path_data = './CHASE_DB1_datasets_training_testing/'

#原始测试图像(用于FOV选择)
DRIVE_test_imgs_original = path_data + 'CHASE_DB1_dataset_imgs_test.hdf5'
test_imgs_orig = load_hdf5(DRIVE_test_imgs_original)#测试集原始图片
full_img_height = test_imgs_orig.shape[2]
full_img_width = test_imgs_orig.shape[3]
#DRIVE数据集提供的边界掩码
# DRIVE_test_border_masks = path_data + 'CHASE_DB1_dataset_borderMasks_test.hdf5'
# test_border_masks = load_hdf5(DRIVE_test_border_masks)#测试集背景图片
#patches的尺寸
patch_height = 64
patch_width = 64
#在平均输出情况下的步幅
stride_height = 5
stride_width = 5
assert (stride_height < patch_height and stride_width < patch_width)
#模型名称
name_experiment = 'test'
path_experiment = './' +name_experiment + '/'
#N个完整的图像要预测
imgs_to_test = 20
#预测图像的分组
N_visual = 1
#====== 平均模式 ===========
average_mode = True

#加载数据，并划分成patches
ini_gt = load_hdf5(path_data + 'CHASE_DB1_dataset_groundTruth_test.hdf5')
patches_imgs_test = None
new_height = None
new_width = None
masks_test = None
patches_masks_test = None
if average_mode == True:
    patches_imgs_test,new_height,new_width,masks_test = get_data_testing_overlap(
        DRIVE_test_imgs_original =DRIVE_test_imgs_original,
        DRIVE_test_groudTruth = path_data + 'CHASE_DB1_dataset_groundTruth_test.hdf5',#masks
        Imgs_to_test = 20,
        patch_height = patch_height,
        patch_width = patch_width,
        stride_height = stride_height,
        stride_width =stride_width
    )
else:
    patches_imgs_test,patches_masks_test = get_data_testing(
        DRIVE_test_imgs_original = DRIVE_test_imgs_original,
        DRIVE_test_groudTruth = path_data + 'DRIVE_dataset_groundTruth_test.hdf5',#masks
        Imgs_to_test = 20,
        patch_height = patch_height,
        patch_width = patch_width
    )





#================ patches的预测结果 ==================================
best_last = 'best'
patches_imgs_test = np.einsum('klij->kijl',patches_imgs_test)

model = M.GCN_BR_RVASPP_Ghost_Mnet(input_size=(64,64,1))
model.summary()
model.load_weights('weight_lstm.hdf5')
predictions = model.predict(patches_imgs_test,batch_size=16,verbose=1)

predictions = np.einsum('kijl->klij',predictions)
patches_imgs_test = np.einsum('kijl->klij',patches_imgs_test)
print(patches_imgs_test.shape)

pred_patches = predictions

print("predicted images size：")
print(predictions.shape)

#==============在相应的图像中转换预测数组=================

#========== 详细说明和可视化预测的图像 ====================
pred_imgs = None
orig_imgs = None
gtruth_masks = None
if average_mode == True:
    pred_imgs = recompone_overlap(pred_patches,new_height,new_width,stride_height,stride_width)#prediction
    # orig_imgs = my_PreProc(test_imgs_orig[0:pred_imgs.shape[0]])#originals
    orig_imgs = test_imgs_orig[0:pred_imgs.shape[0],:,:,:]
    gtruth_masks = masks_test#ground truth masks
else:
    pred_imgs = recompone(pred_patches,10,9)#predictions#完整的预测图片
    orig_imgs = recompone(patches_imgs_test,10,9)#originals#完整的原始图片
    gtruth_masks = recompone(patches_masks_test,10,9)#masks完整的GT图片

#在预测中，应用DRIVE的标记#超过FOV区域的值设为0
# print('killing border')
# kill_border(pred_imgs,test_border_masks)

##恢复到原始的尺寸
orig_imgs = orig_imgs[:,:,0:full_img_height,0:full_img_width]
pred_imgs = pred_imgs[:,:,0:full_img_height,0:full_img_width]
np.save('pred_imgs',pred_imgs)
print("Ori imgs shape："+str(orig_imgs.shape))
print("pred imgs shape："+str(pred_imgs.shape))
print("Gtruth imgs shape："+str(gtruth_masks.shape))

np.save('results',pred_imgs)
np.save('origin',orig_imgs)
np.save('gtruth',gtruth_masks)

assert(orig_imgs.shape[0]==pred_imgs.shape[0] and orig_imgs.shape[0]==gtruth_masks.shape[0])
N_predicted = orig_imgs.shape[0]
group = N_visual
assert(N_predicted%group==0)



#====== 评估结果==============
print ("\n\n========  Evaluate the results =======================")
# #预测结果只包括在FOV区域里边的
y_scores,y_true = pred_only_FOV(pred_imgs,gtruth_masks)#预测的图片结果，GT图片，背景图片
# #y_scores预测图片，y_true:GT图片
print(y_scores.shape)

print("Calculating results only inside the FOV:")
print("y scores pixels："+str(y_scores.shape[0])+"(radius 270:270*270*3.14==228906, including background around retina："+str(pred_imgs.shape[0]*pred_imgs.shape[2]*pred_imgs.shape[3])+"(584*565==329960)")
print ("y true pixels: " +str(y_true.shape[0]) +" (radius 270: 270*270*3.14==228906), including background around retina: " +str(gtruth_masks.shape[2]*gtruth_masks.shape[3]*gtruth_masks.shape[0])+" (584*565==329960)")

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
AUC_ROC = roc_auc_score(y_true, y_scores)
# test_integral = np.trapz(tpr,fpr) #trapz is numpy integration
print ("\nArea under the ROC curve: " +str(AUC_ROC))
roc_curve =plt.figure()
plt.plot(fpr,tpr,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
plt.title('ROC curve')
plt.xlabel("FPR (False Positive Rate)")
plt.ylabel("TPR (True Positive Rate)")
plt.legend(loc="lower right")
plt.savefig(path_experiment+"ROC.png")


#Precision-recaall curve
precision,recall,thresholds = precision_recall_curve(y_true,y_scores)
precision = np.fliplr([precision])[0]
recall = np.fliplr([recall])[0]
AUC_prec_rec = np.trapz(precision,recall)
print ("\nArea under Precision-Recall curve: " +str(AUC_prec_rec))
prec_rec_curve = plt.figure()
plt.plot(recall,precision,'-',label='Area Under the Curve (PRC = %0.4f)'%AUC_prec_rec)
plt.title('Precision - Recall curve')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower right")
plt.savefig(path_experiment+"Precision_recall.png")

#混淆矩阵
threshold_confusion = 0.5
print("\nConfusion matrix:  Custom threshold (for positive) of " +str(threshold_confusion))
y_pred = np.empty((y_scores.shape[0]))
for i in range(y_scores.shape[0]):
    if y_scores[i]>=threshold_confusion:
        y_pred[i]=1
    else:
        y_pred[i]=0
confusion = confusion_matrix(y_true,y_pred)
print(confusion)
accuracy = 0
if float(np.sum(confusion))!=0:
    accuracy = float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion))
print("Global Accuracy："+str(accuracy))
specificity = 0
if float(confusion[0,0]+confusion[0,1])!=0:
    specificity = float(confusion[0,0])/float(confusion[0,0]+confusion[0,1])
print("Specificity："+str(specificity))
sensitivity = 0
if float(confusion[1,1]+confusion[1,0])!=0:
    sensitivity = float(confusion[1,1])/float(confusion[1,1]+confusion[1,0])
print("Sensitivity："+str(sensitivity))
precision = 0
if float(confusion[1,1]+confusion[0,1])!=0:
    precision = float(confusion[1,1])/float(confusion[1,1]+confusion[0,1])
print("Precision："+str(precision))

#Jaccard similarity index
# jaccard_index = jaccard_score(y_true,y_pred,normalize = True)
# print("\nJaccard similarity score: "+str(jaccard_index))

#F1 score
F1_score = f1_score(y_true,y_pred,labels=None,average='binary',sample_weight=None)
print ("\nF1 score (F-measure): " +str(F1_score))

#Save the results
file_perf = open(path_experiment+'performances.txt','w')
file_perf.write("Area under the ROC curve: "+str(AUC_ROC)
                + "\nArea under Precision-Recall curve: " +str(AUC_prec_rec)
                # + "\nJaccard similarity score: " +str(jaccard_index)
                + "\nF1 score (F-measure): " +str(F1_score)
                +"\n\nConfusion matrix:"
                +str(confusion)
                +"\nACCURACY: " +str(accuracy)
                +"\nSENSITIVITY: " +str(sensitivity)
                +"\nSPECIFICITY: " +str(specificity)
                +"\nPRECISION: " +str(precision)
                )
file_perf.close()

#可视化
# Visualize
fig,ax = plt.subplots(14,3,figsize=[15,15],dpi=600)

for idx in range(14):
    ax[idx, 0].imshow(np.squeeze(orig_imgs[idx]))
    ax[idx, 1].imshow(np.squeeze(gtruth_masks[idx]), cmap='gray')
    ax[idx, 2].imshow(np.squeeze(pred_imgs[idx]), cmap='gray')

plt.savefig(path_experiment+'sample_results.png')

pred_imgs = np.einsum('klij->kijl', pred_imgs)
for idx in range(14):
    # plt.imsave(path_experiment+str(idx)+'.png',np.squeeze(pred_imgs[idx])
    # fig,ax = plt.subplots(1,1,figsize=[584,565])
    plt.imshow(np.squeeze(pred_imgs[idx]), cmap='gray')
# plt.axis('off')
# # cv2.imwrite(path_experiment+str(idx)+'.png',arr)
    plt.imsave(path_experiment+str(idx+1)+'.png',np.squeeze(pred_imgs[idx]),cmap='gray')

# orig_imgs = np.einsum('klij->kijl', orig_imgs)
# for idx in range(20):
    
#     plt.imsave(path_experiment+str(idx+1)+'.png',np.squeeze(pred_imgs[idx]),cmap='gray')
