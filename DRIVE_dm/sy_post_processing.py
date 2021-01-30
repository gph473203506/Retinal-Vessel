import os
import cv2
from PIL import Image
import numpy as np
from help_functions import *
import matplotlib.pyplot as plt
from skimage import morphology 
#scikit learn
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
# from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score

path_experiment='./test/post/'
path_experiment1='./test/'

#删除小面积区域 输入图像，黑底白字，输出黑底白字
def Remove_Small_Objects(blur_img,min_size=5):
    blur_img = 1 - blur_img
    arr = blur_img > 0
    temp_img = morphology.remove_small_holes(arr, area_threshold=min_size, connectivity=1)
    temp_img = 1 * (temp_img + 0)
    return 1 - temp_img
def sy_post_processing(img):
    img=np.array(img)
    # print(np.min(img))
    # print(img.shape)
    # plt.figure("pre")
    # arr=img.flatten()
    # n, bins, patches = plt.hist(arr, bins=256, facecolor='green', alpha=0.75)  
    # plt.savefig(path_experiment+'pre/hist.png')
    new_img = np.empty((img.shape[0],img.shape[1],img.shape[2]))
    for i in range(img.shape[0]):
        for h in range(img.shape[1]):
            for w in range(img.shape[2]):
                if img[i,h,w]>=0.42:
                    new_img[i,h,w]=1
                else:
                    new_img[i,h,w]=0
    # num = np.sum(new_img!=0)
    # # print(sum)
    # ls = new_img[new_img>0]
    # sum1 = np.sum(ls)
    # mean1 = sum1/num

    # mean = np.mean(new_img)
    # # print("mean："+str(mean)+"mean1："+str(mean1))
    # new_img[new_img<=mean1] = 0
    # new_img[new_img>mean1] = 1
    # img = new_img
    # print(np.max(img))
    
    # print(np.max(img))
    # new_img = np.empty((img.shape[1],img.shape[2],img.shape[3]))
    # for i in range(img.shape[0]):
    #     new_img = img[i]
    #     mean = np.mean(new_img)
    #     mean = mean*1.4
    #     new_img[new_img>mean] = 1
    #     new_img[new_img<=mean] = 0
    #     img[i] = new_img
    # plt.figure("post")
    # arr=img.flatten()
    # n, bins, patches = plt.hist(arr, bins=256, facecolor='green', alpha=0.75)  
    # plt.savefig(path_experiment+'post/hist.png')  
    # new_img = Remove_Small_Objects(new_img)  
    return new_img

def post_processing(img):
    img=np.array(img)
    new_img = np.empty((img.shape[1],img.shape[2],img.shape[3]))
    for i in range(img.shape[0]):
        new_img = img[i]
        #求眼球内部的均值
        # num = np.sum(new_img!=0)
        # ls = new_img[new_img>0]
        # sum1 = np.sum(ls)
        # mean1 = sum1/num

        # new_img[new_img<=mean1] = 0
        # new_img = new_img/255.0
        # new_img[new_img<0.5000] = 0
        new_img[new_img<0.5]=0
        new_img[new_img>=0.5]=1
        # # new_img = Remove_Small_Objects(new_img,10000)
       
        img[i] = new_img
    return img
# def zh(img):#二值图像转换为特定颜色
    

# def scdbt(pred_img,orig_img):
#     pred_img=np.array(pred_img)
#     orig_img=np.array(orig_img)
#     new_img = np.empty((img.shape[0],img.shape[1],img.shape[2],img.shape[3]))
#     for i in range(pred_img.shape[0]):
#         for j in range(pred_img.shape[2]):
#             for k in range(pred_img.shape[3]):
#                 zh(pred_img[i,:,j,k])
#         temp = np.empty((img.shape[1],img.shape[2],img.shape[3]))
        
#         img[i] = new_img
#     return img



# pred_img = Image.open("./test/gm0.6 zf10.0/1.gif")
# # orig_img = Image.open("./DRIVE/test/1st_manual/01_manual1.gif")

# pred_img = img[0]
# pred_img = pred_img.reshape(pred_img.shape[1]*pred_img.shape[2])


x_img = np.load('pred_imgs.npy')
o_img = np.load('resutls.npy')
x_img=np.array(x_img)
o_img=np.array(o_img)
print(x_img.shape)
xx_img = []
xxx_img = []
oo_img = []
new_img = np.empty((x_img.shape[0],x_img.shape[1],x_img.shape[2],x_img.shape[3]))
for i in range(x_img.shape[0]):
    img = x_img[i]
    pre_imgs_orig = np.einsum('lij->ijl', img)
    plt.imsave(path_experiment1+'pre/'+str(i+1)+'.png',np.squeeze(pre_imgs_orig),cmap='gray')
    new_img = sy_post_processing(pre_imgs_orig)
    post_imgs_orig = new_img
    plt.imsave(path_experiment1+'post/'+str(i+1)+'.png',np.squeeze(post_imgs_orig),cmap='gray')
# for i in range(x_img.shape[0]):
#         for j in range(x_img.shape[1]):
#             for h in range(x_img.shape[2]):
#                 for w in range(x_img.shape[3]):
#                     xxx_img.append(x_img[i,j,h,w])
#                     if x_img[i,j,h,w]>=0.41:
#                         new_img[i,j,h,w]=1
#                     else:
#                         new_img[i,j,h,w]=0
#                     xx_img.append(new_img[i,j,h,w])
#                     oo_img.append(o_img[i,j,h,w])
# y_true = np.array(oo_img)
# y_pred = np.array(xx_img)
# # y_pred = Remove_Small_Objects(y_pred)
# y_scores = np.array(xxx_img)

# fpr, tpr, thresholds = roc_curve(y_true, y_pred)
# AUC_ROC = roc_auc_score(y_true, y_pred)
# # test_integral = np.trapz(tpr,fpr) #trapz is numpy integration
# print ("\nArea under the ROC curve: " +str(AUC_ROC))
# roc_curve =plt.figure()
# plt.plot(fpr,tpr,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
# plt.title('ROC curve')
# plt.xlabel("FPR (False Positive Rate)")
# plt.ylabel("TPR (True Positive Rate)")
# plt.legend(loc="lower right")
# plt.savefig(path_experiment+"ROC.png")

# #Precision-recall curve
# precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
# precision = np.fliplr([precision])[0]  #so the array is increasing (you won't get negative AUC)
# recall = np.fliplr([recall])[0]  #so the array is increasing (you won't get negative AUC)
# AUC_prec_rec = np.trapz(precision,recall)
# print ("\nArea under Precision-Recall curve: " +str(AUC_prec_rec))
# prec_rec_curve = plt.figure()
# plt.plot(recall,precision,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
# plt.title('Precision - Recall curve')
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.legend(loc="lower right")
# plt.savefig(path_experiment+"Precision_recall.png")

# confusion = confusion_matrix(y_true, y_pred)
# print (confusion)
# accuracy = 0
# if float(np.sum(confusion))!=0:
#     accuracy = float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion))
# print ("Global Accuracy: " +str(accuracy))
# specificity = 0
# if float(confusion[0,0]+confusion[0,1])!=0:
#     specificity = float(confusion[0,0])/float(confusion[0,0]+confusion[0,1])
# print ("Specificity: " +str(specificity))
# sensitivity = 0
# if float(confusion[1,1]+confusion[1,0])!=0:
#     sensitivity = float(confusion[1,1])/float(confusion[1,1]+confusion[1,0])
# print ("Sensitivity: " +str(sensitivity))
# precision = 0
# if float(confusion[1,1]+confusion[0,1])!=0:
#     precision = float(confusion[1,1])/float(confusion[1,1]+confusion[0,1])
# print ("Precision: " +str(precision))

# #Jaccard similarity index Jaccard相似性指数
# # jaccard_index = jaccard_similarity_score(y_true, y_pred, normalize=True)
# # print ("\nJaccard similarity score: " +str(jaccard_index))

# #F1 score
# F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)
# print ("\nF1 score (F-measure): " +str(F1_score))

# #Save the results
# file_perf = open(path_experiment+'performances.txt', 'w')
# file_perf.write("Area under the ROC curve: "+str(AUC_ROC)
#                 + "\nArea under Precision-Recall curve: " +str(AUC_prec_rec)
#                 # + "\nJaccard similarity score: " +str(jaccard_index)
#                 +"\nF1 score (F-measure): " +str(F1_score)
#                 +"\n\nConfusion matrix:"
#                 +str(confusion)
#                 +"\nACCURACY: " +str(accuracy)
#                 +"\nSENSITIVITY: " +str(sensitivity)
#                 +"\nSPECIFICITY: " +str(specificity)
#                 +"\nPRECISION: " +str(precision)
#                 )
# file_perf.close()

# Visualize
# fig,ax = plt.subplots(20,3,figsize=[15,15],dpi=600)

# for idx in range(20):
#     ax[idx, 0].imshow(np.squeeze(orig_imgs[idx]))
#     ax[idx, 1].imshow(np.squeeze(gtruth_masks[idx]), cmap='gray')
#     ax[idx, 2].imshow(np.squeeze(pred_imgs[idx]), cmap='gray')

# plt.savefig(path_experiment+'sample_results.png')

# pred_imgs = np.einsum('klij->kijl', pred_imgs)
# for idx in range(20):
#     # plt.imsave(path_experiment+str(idx)+'.png',np.squeeze(pred_imgs[idx])
#     # fig,ax = plt.subplots(1,1,figsize=[584,565])
#     plt.imshow(np.squeeze(pred_imgs[idx]), cmap='gray')
# # plt.axis('off')
# # # cv2.imwrite(path_experiment+str(idx)+'.png',arr)
#     plt.imsave(path_experiment+str(idx+1)+'.png',np.squeeze(pred_imgs[idx]),cmap='gray')