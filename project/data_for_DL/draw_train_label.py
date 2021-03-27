# -*- coding: utf-8 -*-
import utils
import numpy as np 
import config
import cv2
import os

image_root = '/Users/xiaoqing/Desktop/data_share/composite_mix/train/images/'
txt_root = '/Users/xiaoqing/Desktop/data_share/composite_mix/train/labels/'
save_root = '/Users/xiaoqing/Desktop/data_share/composite_mix/train/labeled_images/'

if not os.path.exists(save_root):
    os.makedirs(save_root)  

def read_file_paths(filePath):
    img_names = [x for x in os.listdir(filePath) if x.split('.')[-1] in 'txt|png']
    img_names = sorted(img_names, key=lambda x: int(x.split('.')[-2]), reverse=False)
    img_paths = [os.path.join(filePath, x) for x in img_names]
    return img_paths

  
def load_label(path_label):
    res = list()
    with open(path_label, 'r') as file:
        tmp = file.readlines()
        if len(tmp)==0:
            return None
        for line in tmp:
            #if int(line.split(' ')[0])==3:
            #    return None
            a = list(map(float, line.split(' ')[1:]))
            res.append(np.array([a]))
    res = np.concatenate(res, axis=0)
    return res


def draw(path_img, save_path, preds, select, nid):
    image = cv2.imread(path_img, 1)
    if preds is None:
        cv2.imwrite(os.path.join(save_path, '{}.png'.format(nid)), image) 
        return 
    for e in preds:
        cv2.rectangle(image, tuple(e[:2]), tuple(e[2:]), (255, 255, 255)) 
    if select is None:
        cv2.imwrite(os.path.join(save_path, '{}.png'.format(nid)), image) 
        return 
    else:
        for e in select:
            cv2.rectangle(image, tuple(e[:2]), tuple(e[2:]), (0, 255, 0))         
    cv2.imwrite(os.path.join(save_path, '{}.png'.format(nid)), image)


def draw_labels(save_path, path_img, label_path):
    image = cv2.imread(path_img, 1)
    img_name = path_img.split('/')[-1]
    h,w,c = image.shape
    labels = load_label(label_path)
    if labels is None:
        return 
    labels = utils.label_convert(labels, w, h)
    for e in labels:
        cv2.rectangle(image, tuple(e[:2]), tuple(e[2:]), (255, 255, 255)) 
    cv2.imwrite(os.path.join(save_path, img_name), image) 
        
'''
image_paths = read_file_paths(image_root)
txt_paths = read_file_paths(txt_root)
for path,t_path in  zip(image_paths, txt_paths):
    nid = path.split('/')[-1].split('.')[0]
    print(nid)
    labels = load_label(t_path)
    draw_labels(save_root,path,t_path)
'''