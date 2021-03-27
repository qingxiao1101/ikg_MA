'''
split the long facades with 100 pixels overlap at edge 
'''
# -*- coding: utf-8 -*-
import os 
import cv2
import pickle 
import utils
import numpy as np

MODE = 'train'
image_root = '/Volumes/Qing Xiao/ikg/3_image_generation/{}/composite_image/'.format(MODE) 
txt_root = '/Volumes/Qing Xiao/ikg/3_image_generation/{}/labels/'.format(MODE)
save_root = '/Volumes/Qing Xiao/ikg/3_image_generation/{}/split_composite_edge/'.format(MODE) 
save_labled_image = os.path.join(save_root, 'labled_images')

if not os.path.exists(save_root):
    os.makedirs(save_root)  

if not os.path.exists(save_labled_image):
    os.makedirs(save_labled_image)  
       

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


def write_label(save_path, train_label_coco, nid, idx):

    txt_file = os.path.join(save_path, '{}_{}.txt'.format(nid, idx))
    if os.path.exists(txt_file):
        os.remove(txt_file)
    with open(txt_file,"a+") as f:
        if train_label_coco is None:
            pass
        else:
            for e in train_label_coco:
                f.write("{} {} {} {} {}\n".format(1, e[0],e[1],e[2],e[3]))  

    
def split(save_path, image_path, labels_path):
    save_path_labels = os.path.join(save_path, 'labels')
    if not os.path.exists(save_path_labels):
        os.makedirs(save_path_labels)
    save_path_image = os.path.join(save_path, 'images')
    if not os.path.exists(save_path_image):
        os.makedirs(save_path_image)
    if not os.path.exists(save_path_labels):
        os.makedirs(save_path_labels)
    image = cv2.imread(image_path, 1)
    labels = load_label(labels_path)
    if labels is None:
        return 
    nid = int(image_path.split('/')[-1].split('.')[0])
    
    stack_images = list()
    stack_labels = list()
    h, w, c = image.shape
    
    div = round(w/500.0)
    thres = 100
    if div <= 1:
        cv2.imwrite(os.path.join(save_path_image, '{}_0.png'.format(nid)), image) 
        write_label(save_path_labels, labels, nid, 0)
        return 
    w_ = int(w//max(1, div))
    for i in range(div):
        sub_image = image[:, max(0, w_*i-thres):min(w, w_*(i+1)+thres), :]
        pointform = utils.label_convert(labels, w, h)
        left = pointform[:, 0] >= max(0, w_*i-thres)
        right = pointform[:, 2] <= min(w, w_*(i+1)+thres)
        sign = np.logical_and(left, right)
        if sign.any():
            idx = np.where(sign==True)
            cv2.imwrite(os.path.join(save_path_image, '{}_{}.png'.format(nid, i)), sub_image)
            sub_labels = pointform[idx]
            sub_labels[:, 0] -= max(0, w_*i-thres)
            sub_labels[:, 2] -= max(0, w_*i-thres)
            sub_labels = utils.label_convert_coco(sub_labels, sub_image.shape[1], sub_image.shape[0])
            write_label(save_path_labels, sub_labels, nid, i)
        else:
            continue
    #raise SystemExit('out')

def draw_labels(save_path, path_img, label_path):
    image = cv2.imread(path_img, 1)
    img_name = path_img.split('/')[-1]
    h,w,c = image.shape
    labels = load_label(label_path)
    if labels is None:
        return 
    labels = utils.label_convert(labels, w, h)
    for e in labels:
        e = list(map(int, e))
        cv2.rectangle(image, tuple(e[:2]), tuple(e[2:]), (255, 255, 255)) 
    cv2.imwrite(os.path.join(save_path, img_name), image) 
        
if __name__ == '__main__':
    '''
    path_syn_imgs = read_file_paths(image_root)
    path_txt_labels = read_file_paths(txt_root)
    print(len(path_syn_imgs), len(path_txt_labels))
    for i in range(len(path_syn_imgs)):
        print('\rimage: {}/{}'.format(i, len(path_syn_imgs)), end='')        
        split(save_root, path_syn_imgs[i], path_txt_labels[i])
        
    print("    done!")
    '''
    path_syn_imgs = read_file_paths(os.path.join(save_root, 'images'))
    path_txt_labels = read_file_paths(os.path.join(save_root, 'labels'))
   
    for i in range(len(path_syn_imgs)):
        print('\rimage: {}/{}'.format(i, len(path_syn_imgs)), end='')        
        draw_labels(save_labled_image, path_syn_imgs[i], path_txt_labels[i])
        
    print("    done!")
    




        
