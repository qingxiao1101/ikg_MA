# -*- coding: utf-8 -*-

import os 
import cv2
import pickle 
import utils
import numpy as np

root = '/Volumes/Qing Xiao/ikg/4_detection/deep_learning/val/normal/' 

img_h, img_w = 600, 800
img_list = []
means, stdevs = [], []
imgs_path_list = os.listdir(root)
 
len_ = len(imgs_path_list)
i = 0
for item in imgs_path_list:
    img = cv2.imread(os.path.join(root,item))
    img = cv2.resize(img,(img_w,img_h))
    img = img[:, :, :, np.newaxis]
    img_list.append(img)
    i += 1
    print(i,'/',len_) 

imgs = np.concatenate(img_list, axis=3)
imgs = imgs.astype(np.float32) / 255.
 
for i in range(3):
    pixels = imgs[:, :, i, :].ravel()  # 拉成一行
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))
 
# BGR --> RGB ， CV读取的需要转换，PIL读取的不用转换
means.reverse()
stdevs.reverse()
 
print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))


'''
composite:
val:
    normMean = [0.11987238, 0.020331234, 0.10855926]
    normStd = [0.17639472, 0.04002757, 0.16733186]
    
train:
    normMean = [0.101935625, 0.016865494, 0.07960085]
    normStd = [0.17989501, 0.040014643, 0.1433427]
'''

'''
normal vectors:
val:
normMean = [0.50575083, 0.47802016, 0.4992119]
normStd = [0.1178565, 0.15889896, 0.053908944]
train:
normMean = [0.50734967, 0.5004253, 0.5000852]
normStd = [0.11684509, 0.12964262, 0.04865679]
'''
