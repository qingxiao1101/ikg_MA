# -*- coding: utf-8 -*-
import os 
import cv2
import pickle 
import utils
import numpy as np

import shutil


image_root = '/Volumes/Qing Xiao/ikg/4_detection/all_dataset/train/split_normal/images/' 
label_root = '/Volumes/Qing Xiao/ikg/4_detection/all_dataset/train/split_normal/labels/' 

bad_labeled_image_root = '/Volumes/Qing Xiao/ikg/4_detection/all_dataset/train/split_normal/bad_labeled_images/'
good_image_root = '/Volumes/Qing Xiao/ikg/4_detection/all_dataset/train/split_normal/good_images/' 
good_label_root = '/Volumes/Qing Xiao/ikg/4_detection/all_dataset/train/split_normal/good_labels/' 
bad_image_root = '/Volumes/Qing Xiao/ikg/4_detection/all_dataset/train/split_normal/bad_images/' 
bad_label_root = '/Volumes/Qing Xiao/ikg/4_detection/all_dataset/train/split_normal/bad_labels/' 

bad_m_image_root = '/Volumes/Qing Xiao/ikg/4_detection/all_dataset/train/split_normal/bad_manully_images/' 

#txt_root = '/Volumes/Qing Xiao/ikg/4_detection/all_dataset/train/split_composite/labels/'
#save_root = '/Volumes/Qing Xiao/ikg/4_detection/all_dataset/train/split_composite/' 
#save_labled_image = os.path.join(save_root, 'labled_images')

if not os.path.exists(good_image_root):
    os.makedirs(good_image_root)  
if not os.path.exists(good_label_root):
    os.makedirs(good_label_root)  
if not os.path.exists(bad_image_root):
    os.makedirs(bad_image_root)  
if not os.path.exists(bad_label_root):
    os.makedirs(bad_label_root)  
if not os.path.exists(bad_m_image_root):
    os.makedirs(bad_m_image_root)  


def split_good_bad():
    bad_image_path = '/Volumes/Qing Xiao/ikg/4_detection/all_dataset/train/split_composite/bad_images/' 
    good_image_path = '/Volumes/Qing Xiao/ikg/4_detection/all_dataset/train/split_composite/composite_good/images/' 
    bad_names = [x.split('.')[0] for x in os.listdir(bad_image_path) if x.split('.')[-1] in 'txt|png']
    good_names = [x.split('.')[0] for x in os.listdir(good_image_path) if x.split('.')[-1] in 'txt|png']
    for i, x in enumerate(bad_names):
        print('\r {}/{}...'.format(i, len(bad_names)), end='')
        target_img_path = os.path.join(bad_image_root,'{}.png'.format(x))
        source_img_path = os.path.join(image_root, '{}.png'.format(x))    
        shutil.copyfile(source_img_path,target_img_path)
        target_label_path = os.path.join(bad_label_root,'{}.txt'.format(x))
        source_label_path = os.path.join(label_root, '{}.txt'.format(x))    
        shutil.copyfile(source_label_path,target_label_path)
    print('')
    for i, x in enumerate(good_names):
        print('\r {}/{}...'.format(i, len(good_names)), end='')
        target_img_path = os.path.join(good_image_root,'{}.png'.format(x))
        source_img_path = os.path.join(image_root, '{}.png'.format(x))    
        shutil.copyfile(source_img_path,target_img_path)
        target_label_path = os.path.join(good_label_root,'{}.txt'.format(x))
        source_label_path = os.path.join(label_root, '{}.txt'.format(x))    
        shutil.copyfile(source_label_path,target_label_path)
    print('')
        
#split_good_bad()
#raise SystemError
'''
removed_name = list(n_names-c_names)
removed_name = [x.split('.')[0] for x in removed_name]
for i, nid in enumerate(removed_name):
    print(i, ' / ', len(removed_name))
    source_img_path = os.path.join(n_root, '{}.png'.format(nid)) 
    source_label_path = os.path.join(n_label, '{}.txt'.format(nid)) 
    os.remove(source_img_path)
    os.remove(source_label_path)
'''


'''
bad_img_names = [x.split('.')[0] for x in os.listdir(bad_labeled_image_root) if x.split('.')[-1] in 'txt|png']


print(bad_img_names[0])  #853
print(len(bad_img_names))

all_img_names = [x.split('.')[0] for x in os.listdir(image_root) if x.split('.')[-1] in 'txt|png']
for i, x in enumerate(all_img_names):
    print('\r {}/{}...'.format(i, len(all_img_names)), end='')
    if x not in bad_img_names:
        
        
        target_img_path = os.path.join(good_image_root,'{}.png'.format(x))
        source_img_path = os.path.join(image_root, '{}.png'.format(x))
        target_label_path = os.path.join(good_label_root,'{}.txt'.format(x))
        source_label_path = os.path.join(label_root, '{}.txt'.format(x))
        shutil.copyfile(source_img_path,target_img_path)
        shutil.copyfile(source_label_path,target_label_path)
        
    else:
        target_img_path = os.path.join(bad_image_root,'{}.png'.format(x))
        source_img_path = os.path.join(image_root, '{}.png'.format(x))   
        shutil.copyfile(source_img_path,target_img_path)
'''

bad_manully_label_root = '/Volumes/Qing Xiao/ikg/4_detection/all_dataset/train/split_composite/bad_manully_labels/' 
bad_m_names = [x.split('.')[0] for x in os.listdir(bad_manully_label_root) if x.split('.')[-1] in 'txt|png']
print(len(bad_m_names))  #169
for i, x in enumerate(bad_m_names):
    print('\r {}/{}...'.format(i, len(bad_m_names)), end='')
    target_img_path = os.path.join(bad_m_image_root,'{}.png'.format(x))
    source_img_path = os.path.join(bad_image_root, '{}.png'.format(x))    
    shutil.copyfile(source_img_path,target_img_path)

