# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os


def load_label(root, nid):
    path_label = os.path.join(root, 'labels', '{}.txt'.format(nid))
    res = list()
    with open(path_label, 'r') as file:
        for line in file.readlines():
            a = list(map(float, line.split(' ')[1:]))
            res.append(np.array([a]))
    res = np.concatenate(res, axis=0)
    return res


def draw_label(geo_path, save_root, labels, nid):
    image = cv2.imread(geo_path, 1)
    height = image.shape[0] 
    width = image.shape[1]  
    tmp = np.zeros(labels.shape, dtype=np.int16)
    convert = np.zeros(labels.shape, dtype=np.int16)
    tmp[:, 0] = np.round(labels[:, 0]*width).astype(np.int16)
    tmp[:, 1] = np.round(labels[:, 1]*height).astype(np.int16)
    tmp[:, 2] = np.round(labels[:, 2]*width/2).astype(np.int16)
    tmp[:, 3] = np.round(labels[:, 3]*height/2).astype(np.int16)
    convert[:, 0] = tmp[:, 0] - tmp[:, 2]
    convert[:, 1] = tmp[:, 1] - tmp[:, 3]
    convert[:, 2] = tmp[:, 0] + tmp[:, 2]
    convert[:, 3] = tmp[:, 1] + tmp[:, 3]
    for e in convert:
        cv2.rectangle(image, tuple(e[:2]), tuple(e[2:]), (0, 0, 255)) 
    cv2.imwrite(os.path.join(save_root, '{}_labeled.png'.format(nid)), image)     
    #cv2.imshow("Image", image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

def process_all(root):
    geo_path_root = os.path.join(root, 'geometry_image')
    save_root = os.path.join(root, 'labeled_image')
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    dat_files = os.listdir(geo_path_root)
    dat_files = sorted(dat_files, key=lambda x:int(x.split('.')[0]))
    tmp_sum = 0
    for idx, file in enumerate(dat_files):
        nid = int(file.split('.')[0])
        print(idx, '... ', nid)

        #if nid !=28:
        #    continue
        geo_path = os.path.join(geo_path_root, '{}.png'.format(nid))
        labels = load_label(root, nid)
        if len(labels)>1:
            tmp_sum += len(labels)
        #depth_path = os.path.join(depth_path_root, '{}.png'.format(nid))
        #draw_label(geo_path, save_root, labels, nid)
    print(tmp_sum)          
root = '/Volumes/Qing Xiao/ikg/4_detection/part_dataset_v2/' 

process_all(root)