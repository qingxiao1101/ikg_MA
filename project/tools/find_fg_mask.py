# -*- coding: utf-8 -*-
'''
generating the foreground mask for occlusion processing.
'''

import os 
import cv2
import pickle 
import utils
import numpy as np
from skimage import io

root = '/Volumes/Qing Xiao/ikg/3_dataset_generation/test/geometry_image/'
save_root =  '/Volumes/Qing Xiao/ikg/3_dataset_generation/test/fg_mask/'
if not os.path.exists(save_root):
    os.makedirs(save_root)  
 
def read_file_paths(filePath):
    img_names = [x for x in os.listdir(filePath) if x.split('.')[-1] in 'dat|png']
    img_names = sorted(img_names, key=lambda x: int(x.split('.')[-2]), reverse=False)
    img_paths = [os.path.join(filePath, x) for x in img_names]
    return img_paths

img_paths = read_file_paths(root)

for path in img_paths:
    name = path.split('/')[-1]
    print(name)
    img = cv2.imread(path, 0)
    img = cv2.GaussianBlur(img,(7,7),0)
    binary = np.where(img>0, 255, 0)
    cv2.imwrite(os.path.join(save_root, name), binary)

