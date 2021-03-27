# -*- coding: utf-8 -*-

import os 
import cv2
import pickle 
import utils
import numpy as np

image_root1 = '/Volumes/Qing Xiao/ikg/3_dataset_generation/test/geometry_image/' 
image_root2 = '/Volumes/Qing Xiao/ikg/3_dataset_generation/test/occl_comp/geometry_image/' 

path_occlusion = '/Volumes/Qing Xiao/ikg/3_dataset_generation/test/occlusion_mask/' 

if not os.path.exists(path_occlusion):
    os.makedirs(path_occlusion)  


def read_file_paths(filePath):
    img_names = [x for x in os.listdir(filePath) if x.split('.')[-1] in 'txt|png']
    img_names = sorted(img_names, key=lambda x: int(x.split('.')[-2]), reverse=False)
    img_paths = [os.path.join(filePath, x) for x in img_names]
    return img_paths

if __name__ == '__main__':
    
    path_imgs_original = read_file_paths(image_root1)
    path_imgs_occl = read_file_paths(image_root2)

    for p1, p2 in zip(path_imgs_original, path_imgs_occl):
        name = p1.split('/')[-1]
        print(name)
        image1 = cv2.imread(p1, 0)
        image2 = cv2.imread(p2, 0)
        tmp = image2 - image1
        cv2.imwrite(os.path.join(path_occlusion, name), tmp)