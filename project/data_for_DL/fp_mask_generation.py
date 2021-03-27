# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np

MODE = 'train'
PATH = "/Volumes/Qing Xiao/ikg"
image_root = PATH + '/3_image_generation/{}/geometry_image/'.format(MODE)
save_root = PATH + '/3_image_generation/{}/fp_mask/'.format(MODE)
if not os.path.exists(save_root):
   os.makedirs(save_root)  

def read_file_paths(filePath):
    img_names = [x for x in os.listdir(filePath) if x.split('.')[-1] in 'txt|png']
    img_names = sorted(img_names, key=lambda x: int(x.split('.')[-2]), reverse=False)
    img_paths = [os.path.join(filePath, x) for x in img_names]
    return img_paths


for path in read_file_paths(image_root):
    nid = path.split('/')[-1].split('.')[0]
    print(nid)
    img = cv2.imread(path, 0)
    img =cv2.medianBlur(img,5)
    img = np.where(img>0, 255, 0)
    cv2.imwrite(os.path.join(save_root, '{}.png'.format(nid)), img)

    