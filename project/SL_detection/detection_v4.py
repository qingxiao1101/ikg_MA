# -*- coding: utf-8 -*-

from skimage import io
import pickle
import numpy as np

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.cluster import DBSCAN
import pcl
import random
import cv2
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure,draw 
import os  
from scipy import stats
import config
import math
from tools.median_filter import median_filter
import scipy.signal as signal
import utils
from draw_misclassification import draw as draw_mis
import pandas as pd

import scipy as sp
import scipy.ndimage


root = '/Volumes/Qing Xiao/ikg/4_detection/rule_based_test92/' 
save_path_root = os.path.join(root, 'detected_dynamic_final')
if not os.path.exists(save_path_root):
   os.makedirs(save_path_root)  

bias = 0.05


def _calculate_average(img, min_val, modality='mean'):
    assert modality in ('mean', 'median', 'mode')
    global bias
    idx = np.where(img.reshape(-1)>min_val)[0]
    if len(idx) == 0:
        return min_val
    tmp = img.reshape(-1)[idx]

    if modality == 'mean':
        return np.mean(tmp) 
    elif modality == 'median':

        #return np.median(tmp)-bias
        '''
        median = np.median(tmp)
        i = np.where(tmp<=median)[0]
        arr = tmp[i]
        std = np.std(arr,ddof=1)
        #print(std)
        '''
        return np.median(tmp)  
        
    else:
        tmp = np.round(tmp, decimals=2)
        mode = stats.mode(tmp)[0][0] 
        return mode
        '''
        if np.sum(np.where(np.abs(mode-tmp)<bias, 1, 0))/(len(tmp)) < 0.3:
            return np.median(tmp) 
        else:
            return mode 
        '''
        
def order_points(pts):
    # pts为轮廓坐标
    # 列表中存储元素分别为左上角，右上角，右下角和左下角
    rect = np.zeros((4, 2), dtype = "float32")
    # 左上角的点具有最小的和，而右下角的点具有最大的和
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # 计算点之间的差值
    # 右上角的点具有最小的差值,
    # 左下角的点具有最大的差值
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # 排序坐标(依次为左上右上右下左下)
    xs = [i[0] for i in rect]
    ys = [i[1] for i in rect]
    xs.sort()
    ys.sort()
    return (xs[1], ys[1]), (xs[2], ys[2])  #leftup, rightdown


def reference_averages(img, density_image, horizontal, vertical, global_avg, bg, modality='median'):
    """
    @horizontal: list, 
    @vertical: list, at least two elements, [0, columns]
    return the image of local reference value and width of scan line
    """
    global bias
    averages = np.zeros(img.shape)
    rows, columns = img.shape
    width_scan_line = np.zeros(img.shape)
    #bg = img.min()
    #divide_part = columns//len_each_part + 1
    for i in range(len(horizontal)-1):
        for j in range(len(vertical)-1):
            avg = _calculate_average(img[horizontal[i]:horizontal[i+1], vertical[j]:vertical[j+1]], bg, modality)
            avg = avg - bias if avg != bg  else global_avg
            averages[horizontal[i]:horizontal[i+1], vertical[j]:vertical[j+1]] = avg
            tmp = density_image[horizontal[i]:horizontal[i+1], vertical[j]:vertical[j+1]]
            a = tmp.mean()
            idx = np.where(tmp.reshape(-1)>0)[0]
            if len(idx)==0:
                b = 5
            else:
                a = np.mean(tmp)
                if a < 30:
                    b = 5
                else:
                    b = 2
            width_scan_line[horizontal[i]:horizontal[i+1], vertical[j]:vertical[j+1]] = b
    return averages, width_scan_line



def refine_divide(ref_image, horizontal_0, vertical_0, nid, save_path):
    rows, columns, _ = ref_image.shape
    region_image = cv2.cvtColor(ref_image, cv2.COLOR_RGB2GRAY)
    region_image=cv2.medianBlur(region_image,5)
    #region_image = cv2.GaussianBlur(region_image,(5,5),0)
    edge = cv2.Canny(region_image, 100, 200)
    edge = np.array(edge)    
    step = 1
    counts = []

    for idx, line in enumerate(edge-step):
        a = np.count_nonzero(edge[idx:idx+step,:])
        counts.append((idx, a))
    counts = sorted(counts, key=lambda x:x[1], reverse=True)
    #horizontal = list()
    for e in counts[:30]:
        h = e[0]

        if h > rows-50:
            continue
        #if occup > 0.2:
        if (np.array([abs(h-x) for x in horizontal_0])<30).any():
            pass
        else:
            horizontal_0.append(h)
    
    counts = []
    edge_t = edge.T
    
    for idx, line in enumerate(edge.T):
        a = np.count_nonzero(line)
        counts.append((idx, a))
    counts = sorted(counts, key=lambda x:x[1], reverse=True)
    #vertical = list()
    
    for e in counts[:30]:
        h = e[0]
        if h > columns-50:
            continue
        #if occup > 0.3:
        if (np.array([abs(h-x) for x in vertical_0])<50).any():
            pass
        else:
            vertical_0.append(h)    
    #horizontal.extend(horizontal_0)
    horizontal_0.sort()
    #vertical.extend(vertical_0)
    vertical_0.sort()
    
    for c in vertical_0:
        cv2.line(edge, (c, 0), (c, rows), (255, 255, 255), 2, 4)
    for h in horizontal_0:
        cv2.line(edge, (0, h), (columns, h), (255, 255, 255), 2, 4)
    path_divide_img = os.path.join(save_path, 'region_divide_refine')
    if not os.path.exists(path_divide_img):
        os.makedirs(path_divide_img)
    cv2.imwrite(os.path.join(path_divide_img, '{}.png'.format(nid)), edge)
    
    return horizontal_0, vertical_0

def region_divide(nid, img, depth_path, ref_plane, save_path):

    averages = np.ones(img.shape, dtype=np.float32)*ref_plane
    rows, columns = img.shape
    divide_bias = 0.075
    tmp_averages = np.expand_dims(averages, axis=2).repeat(3, axis=2)
    tmp_img = np.expand_dims(img, axis=2).repeat(3, axis=2)
    blue_img = np.zeros((rows, columns, 3), dtype=np.uint8)
    blue_img[:,:,0] = 255
    red_img = np.zeros((rows, columns, 3), dtype=np.uint8)
    red_img[:,:,2] = 255
    reference_image = cv2.imread(depth_path, 1)
    reference_image = np.where((tmp_img-tmp_averages)>divide_bias, red_img, reference_image)
    reference_image = np.where(np.abs(tmp_img-tmp_averages)<divide_bias, blue_img, reference_image)
    path_ri = os.path.join(save_path, 'region_image')
    if not os.path.exists(path_ri):
        os.makedirs(path_ri)
    cv2.imwrite(os.path.join(path_ri, '{}.png'.format(nid)), reference_image)
    region_image = cv2.cvtColor(reference_image, cv2.COLOR_RGB2GRAY)
    region_image=cv2.medianBlur(region_image,5)
    #region_image = cv2.GaussianBlur(region_image,(5,5),0)
    '''
    depth_image = cv2.imread(depth_path, 0)
    rows, columns = depth_image.shape
    '''

    sobely = cv2.Canny(region_image, 100, 200)
    
    edge = np.array(sobely)
    
    step = 1
    counts = []

    for idx, line in enumerate(edge-step):
        a = np.count_nonzero(edge[idx:idx+step,:])
        counts.append((idx, a))
    counts = sorted(counts, key=lambda x:x[1], reverse=True)
    horizontal = list()
    for e in counts[:20]:
        h = e[0]

        if h > rows-50:
            continue
        occup = e[1]/columns
        #if occup > 0.2:
        horizontal.append(h)
      
        #cv2.line(sobely, (0, h), (columns, h), (255, 255, 255), 1, 4)

    horizontal.sort()
    horizontal.insert(0, 0)
    horizontal.insert(len(horizontal), rows)
    horizontal.sort(reverse=True)

    tmp1 = [horizontal[0]]
    pre = columns
    for i in range(1, len(horizontal)-1):
        if pre - horizontal[i] > 50:
            tmp1.append(horizontal[i])
            pre = horizontal[i]
    if tmp1[-1] > 0:
        tmp1[-1] = 0
    horizontal = tmp1
    horizontal.sort()
    #print("horizontal2: ", horizontal)
    counts = []
    edge_t = edge.T
    
    for idx, line in enumerate(edge.T):
        a = np.count_nonzero(line)
        counts.append((idx, a))
    counts = sorted(counts, key=lambda x:x[1], reverse=True)
    vertical = list()
    
    for e in counts[:20]:
        h = e[0]
        if h > columns-50:
            continue
        occup = e[1]/rows
        #if occup > 0.3:
        vertical.append(h)
    vertical.sort()
    vertical.insert(0, 0)
    vertical.insert(-1, columns)
    tmp = [vertical[0]]
    for i in range(1, len(vertical)-1):
        if vertical[i] - vertical[i-1] > 80:
            tmp.append(vertical[i])
    if tmp[-1] < columns:
        tmp[-1] = columns
    vertical = tmp
    if len(vertical)==2:
        vertical = [0, int(0.25*columns), int(0.5*columns), int(0.75*columns), columns]

    for c in vertical:
        cv2.line(sobely, (c, 0), (c, rows), (255, 255, 255), 2, 4)
    for h in horizontal:
        cv2.line(sobely, (0, h), (columns, h), (255, 255, 255), 2, 4)
    path_divide_img = os.path.join(save_path, 'region_divide')
    if not os.path.exists(path_divide_img):
        os.makedirs(path_divide_img)
    cv2.imwrite(os.path.join(path_divide_img, '{}.png'.format(nid)), sobely)
    '''
    cv2.imshow("Image", sobely)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    raise InterruptedError()
    '''
    return horizontal, vertical

   
def load_label(root, nid):
    path_label = os.path.join(root, 'labels', '{}.txt'.format(nid))
    res = list()
    with open(path_label, 'r') as file:
        for line in file.readlines():
            a = list(map(float, line.split(' ')[1:]))
            res.append(np.array([a]))
    res = np.concatenate(res, axis=0)
    return res

def generate_line(pos, idx, rows, cols, scann_width, h_direction=True, save_edge=False):
    start = list()
    end = list() 
    bg_start = list()
    bg_end = list()
    if len(pos) > 0:
        count = 0
        for i, e in enumerate(pos):
            if count==0:
                start.append(e)
                count+=1
            else:
                if (e - start[-1]) < count+6: #consider the outliers 
                    count += 1
                else:
                    end.append(pos[i-1])
                    count=0
    
        end.append(pos[-1])
    else:
        return [], []

    if start[0] == 0 and not save_edge:
        bg_start.append(start.pop(0))
        bg_end.append(end.pop(0))
    if len(start)>0:
        if not save_edge:        
            if h_direction and end[-1] > cols-scann_width*2:
                bg_start.append(start.pop(-1))
                bg_end.append(end.pop(-1))
            elif not h_direction and end[-1] > rows-scann_width*2:
                bg_start.append(start.pop(-1))
                bg_end.append(end.pop(-1))
    else:
        return [], []
    h = [idx]*len(start)
    bg_h = [idx] * len(bg_start)       
    return list(zip(start, end, h)), list(zip(bg_start, bg_end, bg_h))       
        

def find_lines(img, averages, avg_densitys, scann_width = 5, printf=True, save_edge=False):
    rows, cols = img.shape
    bg = img.min()
    step = 1
    
    scann_lines_h = list()
    scann_lines_bg_h = list()
    for i in range(0, rows, 1):
        if printf:
            print('\rscann line h: {}/{}'.format(i, rows), end='')
        pos = list()
        for j in range(0, cols-scann_width, step):
            scann_height = int(avg_densitys[i, j])
            if i > rows-6:
                line = img[i, j:j+scann_width]
            else:
                line = img[i:i+scann_height, j:j+scann_width]
            if (line<=averages[i, j]).all():# and avg_densitys[i, j]>0:
                pos.append(j)
        #print(i, ": ", pos)
        sub_lines_h, bg_h = generate_line(pos, i, rows, cols, scann_width, h_direction=True, save_edge=save_edge)
        scann_lines_h.extend(sub_lines_h)
        scann_lines_bg_h.extend(bg_h)
    if printf:
        print('')
    
    scann_lines_v = list()
    scann_lines_bg_v = list()
    for j in range(0, cols, 1):
        if printf:
            print('\rscann line v: {}/{}'.format(j, cols), end='')
        pos = list()
        for i in range(0, rows-scann_width, step):
            scann_height = int(avg_densitys[i, j])
            if j > cols-6:
                line = img[i:i+scann_width, j]
            else:
                line = img[i:i+scann_width, j:j+scann_height]
            if (line<=averages[i, j]).all():# and avg_densitys[i, j]>0:
                pos.append(i)
        #print(i, ": ", pos)
        sub_lines_v, bg_v = generate_line(pos, j, rows, cols, scann_width, h_direction=False, save_edge=save_edge)
        scann_lines_v.extend(sub_lines_v)
        scann_lines_bg_v.extend(bg_v)
        
    if printf:
        print('')
        
    return scann_lines_h, scann_lines_v , scann_lines_bg_h, scann_lines_bg_v               

def candidate_process(img, densitys, rects, fg_mask):
    global bias
    scann_width = 5
    rows, cols = img.shape
    min_val=  img.min()
    new_rects = list()
    for idx, rect in enumerate(rects):
        print('\rcandidate: {}/{}'.format(idx, len(rects)), end='')
        x, y, w, h = rect
        if w>h:
            x0 = max(0, x-w//2)
            y0 = max(0, y-w//2)
            x1 = min(cols, x0+w*2)
            y1 = min(rows, y0+w*2)
        else:
            x0 = max(0, x-h//2)
            y0 = max(0, y-h//2)
            x1 = min(cols, x0+h*2)
            y1 = min(rows, y0+h*2)
        '''    
        x0 = max(0, x-w)
        y0 = max(0, y-h)
        x1 = min(cols, x0+w*3)
        y1 = min(rows, y0+h*3)
        '''
        sub_img = img[y0:y1, x0:x1]
        sub_fg_mask = fg_mask[y0:y1, x0:x1]
        res = np.zeros(sub_img.shape, dtype=np.uint8)
        average = _calculate_average(sub_img, min_val, modality='mode')
        average -= bias
        averages = np.ones(sub_img.shape)*average
        density = int(np.median(densitys[y0:y1, x0:x1]))

        avg_densitys = np.ones(sub_img.shape)*density
        sub_lines_h, sub_lines_v, _, _ = find_lines(sub_img, averages, avg_densitys, scann_width = scann_width, printf=False, save_edge=True)

        scann_line_h = np.zeros(sub_img.shape, dtype=np.uint8)
        scann_line_v = scann_line_h.copy()

        for s,e,i in sub_lines_h:
            #if (e-s)*0.02 < 0.3:  # filter out scann line that less than 0.3m
            #    continue
            if (e-s) > 0.8*(x1-x0):
                continue
            scann_line_h[i, s:e+scann_width] = 255
    
        for s,e,i in sub_lines_v:
            #if (e-s)*0.02 < 0.1:  # filter out scann line that less than 0.3m
            #    continue
            if (e-s) > 0.8*(y1-y0):
                continue
            scann_line_v[s:e+scann_width, i] = 255

        scann_line_h = cv2.cvtColor(scann_line_h, cv2.COLOR_GRAY2BGR)
        gray = cv2.cvtColor(scann_line_h, cv2.COLOR_BGR2GRAY)
        #cv2.imwrite("56_scann_lines.png", bi_img)    
        ret,binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
        
        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        vertical = cv2.erode(binary, verticalStructure)
        vertical = cv2.dilate(vertical, verticalStructure)
        
        # close holes to make it solid rectangle
        kernel = np.ones((5,5),np.uint8)
        close_h = cv2.morphologyEx(vertical, cv2.MORPH_CLOSE, kernel)
     
        scann_line_v = cv2.cvtColor(scann_line_v, cv2.COLOR_GRAY2BGR)
        gray_v = cv2.cvtColor(scann_line_v, cv2.COLOR_BGR2GRAY)
        ret,binary = cv2.threshold(gray_v, 0, 255, cv2.THRESH_BINARY)
        
        hStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        tmp = cv2.erode(binary, hStructure)
        tmp = cv2.dilate(tmp, hStructure)
        
        # close holes to make it solid rectangle
        kernel = np.ones((5,5),np.uint8)
        close_v = cv2.morphologyEx(tmp, cv2.MORPH_CLOSE, kernel)
        
        close = cv2.bitwise_and(close_v, close_h)
        close = cv2.bitwise_and(close, sub_fg_mask)
        
        _,contours, hierarchy = cv2.findContours(close, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        for c in contours:
            #x, y, w, h = cv2.boundingRect(c)
            xn, yn, wn, hn = cv2.boundingRect(c)        
            xn += x0
            yn += y0 + density - 1
            new_rects.append(np.array([[xn, yn, xn+wn, yn+hn]]))
    return np.concatenate(new_rects, axis=0) if len(new_rects) > 0 else None


def processing(path, geo_path, depth_path, density_path, info_path, save_path, nid):
    with open(path, 'rb') as f:
        img = pickle.load(f)
    #img = cv2.imread(depth_path, 0)
    #img = np.array(img)
    rows, columns = img.shape
    #depth_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #depth_image = cv2.cvtColor(depth_image,cv2.COLOR_BGR2GRAY)
    with open(info_path, 'rb') as f:
        info = pickle.load(f)
    ref_plane = info['ref_plane']
    buttom_edge = info['buttom_edge']
    left_edge = info['left_edge']
    horizontal, vertical = region_divide(nid, img, depth_path, ref_plane, save_path)
    geo_img_h = Image.open(geo_path) 
    geo_img_v = geo_img_h.copy()

    density_image = cv2.imread(density_path, 0)    
    bg = img.min()

    #img = signal.medfilt2d(img, (5,5))
    global_avg = _calculate_average(img, bg, modality=config.modality_reference_average)
    averages, _ = reference_averages(img, density_image, horizontal, vertical, global_avg, bg, modality=config.modality_reference_average)
    

    blue_img = np.zeros((rows, columns, 3), dtype=np.uint8)
    blue_img[:,:,0] = 255
    red_img = np.zeros((rows, columns, 3), dtype=np.uint8)
    red_img[:,:,2] = 255
    reference_image = cv2.imread(depth_path, 1)
    reference_image_2 = reference_image.copy()
    tmp_averages = np.expand_dims(averages+bias, axis=2).repeat(3, axis=2)
    tmp_img = np.expand_dims(img, axis=2).repeat(3, axis=2)
    reference_image = np.where((tmp_img-tmp_averages)>bias, red_img, reference_image)
    reference_image = np.where(np.abs(tmp_img-tmp_averages)<bias, blue_img, reference_image)
    
    for c in vertical:
        cv2.line(reference_image, (c, 0), (c, rows), (255, 255, 255), 2, 4)
    for h in horizontal:
        cv2.line(reference_image, (0, h), (columns, h), (255, 255, 255), 2, 4)

    horizontal, vertical = refine_divide(reference_image, horizontal, vertical, nid, save_path)
    averages, avg_density = reference_averages(img, density_image, horizontal, vertical, global_avg, bg, modality=config.modality_reference_average)
    tmp_averages = np.expand_dims(averages+bias, axis=2).repeat(3, axis=2)
    tmp_img = np.expand_dims(img, axis=2).repeat(3, axis=2)
    reference_image_2 = np.where((tmp_img-tmp_averages)>bias, red_img, reference_image_2)
    reference_image_2 = np.where(np.abs(tmp_img-tmp_averages)<bias, blue_img, reference_image_2)
    
           
    for c in vertical:
        cv2.line(reference_image_2, (c, 0), (c, rows), (255, 255, 255), 2, 4)
    for h in horizontal:
        cv2.line(reference_image_2, (0, h), (columns, h), (255, 255, 255), 2, 4)
        
    scann_width = 5
    sub_lines_h, sub_lines_v, bg_h, bg_v = find_lines(img, averages, avg_density, scann_width)

    im_cv_contour = cv2.imread(geo_path, 1)
    im_inner_rect_cand = im_cv_contour.copy()
    im_inner_rect_pred = im_cv_contour.copy()
    
    scann_line_img_h = np.zeros((rows, columns), dtype=np.uint8)
    scann_line_img_v = scann_line_img_h.copy()
    scann_line_bg = scann_line_img_h.copy()

    draw = ImageDraw.Draw(geo_img_h) 
    draw_vertical = ImageDraw.Draw(geo_img_v) 
        
    for s,e,h in sub_lines_h:
        #if (e-s)*0.02 < 0.3:  # filter out scann line that less than 0.3m
        #    continue
        if (e-s)*0.02 > 6.0:
            bg_h.append([s,e,h])
            continue
        scann_line_img_h[h, s:e+scann_width] = 255
        draw.line((s,h)+(e+scann_width,h),fill=255)

    for s,e,w in sub_lines_v:
        #if (e-s)*0.02 < 0.1:  # filter out scann line that less than 0.3m
        #    continue
        if (e-s)*0.02 > 6.0:
            bg_v.append([s,e,w])
            continue
        scann_line_img_v[s:e+scann_width, w] = 255
        draw_vertical.line((w,s)+(w, e+scann_width),fill=255)


    for s,e,h in bg_h:
        scann_line_bg[h, s:e+scann_width] = 255
    for s,e,w in bg_v:
        scann_line_bg[s:e+scann_width, w] = 255

    scann_line_img_h = cv2.cvtColor(scann_line_img_h, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(scann_line_img_h, cv2.COLOR_BGR2GRAY)
    ret,binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    vertical = cv2.erode(binary, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)
    
    
    # close holes to make it solid rectangle
    kernel = np.ones((5,5),np.uint8)
    close_h = cv2.morphologyEx(vertical, cv2.MORPH_CLOSE, kernel)
 
    scann_line_img_v = cv2.cvtColor(scann_line_img_v, cv2.COLOR_GRAY2BGR)
    gray_v = cv2.cvtColor(scann_line_img_v, cv2.COLOR_BGR2GRAY)
    ret,binary = cv2.threshold(gray_v, 0, 255, cv2.THRESH_BINARY)
    
    hStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    tmp = cv2.erode(binary, hStructure)
    tmp = cv2.dilate(tmp, hStructure)
        
    # close holes to make it solid rectangle
    kernel = np.ones((5,5),np.uint8)
    close_v = cv2.morphologyEx(tmp, cv2.MORPH_CLOSE, kernel)


    scann_line_bg = cv2.cvtColor(scann_line_bg, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(scann_line_bg, cv2.COLOR_BGR2GRAY)
    ret,binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    scann_line_bg = cv2.erode(binary, verticalStructure)
    scann_line_bg = cv2.dilate(scann_line_bg, verticalStructure)

    foreground_mask = 255 -  scann_line_bg 
    
    close = cv2.bitwise_and(close_v, close_h)
    close = cv2.bitwise_and(close, foreground_mask)
    
    _,contours, hierarchy = cv2.findContours(close, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    mask = np.ones((rows, columns, 3), dtype="uint8")
    idx = 0
    candidates = list()
    rects = list()
    for c in contours:
	# compute the center of the contour
        
        M = cv2.moments(c)
        #x, y, w, h = cv2.boundingRect(c)
        x, y, w, h = cv2.boundingRect(c)
        # compensation of scann_height
        h = h + int(avg_density[y+h//2, x+w//2]) - 1
        

        leftup = (x,y)
        rightdown = (x+w, y+h)
        #leftup, rightdown = order_points(c.reshape(c.shape[0], 2))
        #w = rightdown[0] - leftup[0]
        #h = rightdown[1] - leftup[1]

        if (w<12 or h <= 12) and (leftup[0]>rows-100):   # w<20, h<12
            continue
        if (w<20 or h <= 12) and (leftup[0]<=rows-100):   # w<20, h<12
            continue
        #if w*h*0.02*0.02<0.1:
        #    continue
        if w*0.02 >6.0 or h*0.02>4.0:
            continue
        if w/h >= 5 or w/h < 0.2:
            continue
        #print(w, h)
        rects.append([x,y,w,h])
        cX = int(M["m10"] / max(M["m00"], 0.0001))
        cY = int(M["m01"] / max(M["m00"], 0.0001))
        # draw the contour and center of the shape on the image
        cv2.drawContours(im_cv_contour, [c], -1, (255, 255, 255), 2)
        
        cv2.circle(im_cv_contour, (cX, cY), 3, (255, 255, 255), -1)
        
        candidates.append(np.array([[leftup[0], leftup[1], rightdown[0], rightdown[1]]]))
        cv2.rectangle(im_inner_rect_cand, leftup, rightdown, (0, 0, 255)) 

        cv2.drawContours(mask, [c], -1, (255, 255, 255), 2)
        cv2.rectangle(mask, leftup, rightdown, (0, 0, 255)) 
        cv2.circle(mask, (cX, cY), 3, (255, 255, 255), -1)
        cv2.putText(mask, "{}".format(idx), (cX - 10, cY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        idx += 1
        #cv2.imshow("Image", im_cv)
        #cv2.waitKey(0)
    if len(candidates)>0:
        candidates = np.concatenate(candidates, axis=0)  
    else:
        candidates = None
    labels = load_label(root, nid)
    if labels is not None:
        labels = utils.label_convert(labels, columns, rows)
        
    final_rects = candidate_process(img, avg_density, rects, foreground_mask)
    final_rects = utils.prediction_filter(final_rects, candidates)
    
    predictions = list()
    if final_rects is not None:
        for e in final_rects:
            x0, y0, x1, y1 = e
            
            leftup = (x0,y0)
            rightdown = (x1, y1)
            
            density = np.median(avg_density[y0:y1, x0:x1])
            w = x1-x0
            h = y1-y0
            #leftup, rightdown = order_points(c.reshape(c.shape[0], 2))
            #w = rightdown[0] - leftup[0]
            #h = rightdown[1] - leftup[1]
    
            if (w<12 or h <= 12) and (leftup[0]>rows-100):   # w<20, h<12
                continue
            if (w<20 or h <= 12) and (leftup[0]<=rows-100):   # w<20, h<12
                continue
            #if w*h*0.02*0.02<0.1:
            #    continue
            if w*0.02 >5.0 or h*0.02>4.0:
                continue
            if w/h >= 5 or w/h < 0.2:
                continue
            
            predictions.append(np.array([[leftup[0], leftup[1], rightdown[0], rightdown[1]]]))
    
            cv2.rectangle(im_inner_rect_pred, leftup, rightdown, (0, 255, 0)) 
    
    if len(predictions)>0:
        predictions = np.concatenate(predictions, axis=0)  
        path_pred_txt = os.path.join(save_path, 'pred_txt')
        if not os.path.exists(path_pred_txt):
            os.makedirs(path_pred_txt)   
        with open(os.path.join(path_pred_txt, '{}.txt'.format(nid)), 'w') as f:
          for i, e in enumerate(predictions):
            #f.write(str(cls_box_scores[i])+' '+ ' '.join(map(str, e))+'\n')
            f.write(' '.join(map(str, e))+'\n')
    else:
        predictions = None                

    #print(candidates[:, 0].max(), candidates[:, 1].max(),candidates[:, 2].max(), candidates[:, 3].max())
    #print(labels[:, 0].max(), labels[:, 1].max(),labels[:, 2].max(), labels[:, 3].max())

    tp, fp, fn = utils.confusion_matrix(predictions, labels)
    pred_height, actual_height = utils.height_error(predictions, labels)
    
    gdf = utils.window_line(predictions, img, averages, info_path, nid)
    path_window_geojson = os.path.join(save_path, 'window_geojson')
    if not os.path.exists(path_window_geojson):
        os.makedirs(path_window_geojson)   

    path_window_ply = os.path.join(save_path, 'ply_detected_windows')
    if not os.path.exists(path_window_ply):
        os.makedirs(path_window_ply)      
    #ply_path = '/Volumes/Qing Xiao/ikg/4_detection/part_dataset_v2/ply_in_depth/28_in_depth.ply'
    utils.window_line_cp(predictions, labels, averages+bias, info_path, path_window_ply, nid)
    
    
    if not gdf.empty:
        gdf.to_file(os.path.join(path_window_geojson, '{}.geojson'.format(nid)), driver='GeoJSON')

    #print("precision:", tp/(tp+fp))
    #print('recall:', tp/(tp+fn))
    path_image_tpfpfn = os.path.join(save_path, 'image_tpfpfn')
    if not os.path.exists(path_image_tpfpfn):
        os.makedirs(path_image_tpfpfn)   
    draw_mis(geo_path, path_image_tpfpfn, predictions, labels, nid)
    
    path_scann_line_in_image_h = os.path.join(save_path, 'scann_line_in_image_h')
    if not os.path.exists(path_scann_line_in_image_h):
        os.makedirs(path_scann_line_in_image_h)
    path_refer_mask_refine = os.path.join(save_path, 'refer_mask_refine')
    if not os.path.exists(path_refer_mask_refine):
        os.makedirs(path_refer_mask_refine)
    path_scann_line_h = os.path.join(save_path, 'scann_line_h')
    if not os.path.exists(path_scann_line_h):
        os.makedirs(path_scann_line_h)
    path_fg_mask = os.path.join(save_path, 'fg_mask')
    if not os.path.exists(path_fg_mask):
        os.makedirs(path_fg_mask)
       
    path_scann_line_v = os.path.join(save_path, 'scann_line_v')
    if not os.path.exists(path_scann_line_v):
        os.makedirs(path_scann_line_v)
        
    path_scann_line_in_image_v = os.path.join(save_path, 'scann_line_in_image_v')
    if not os.path.exists(path_scann_line_in_image_v):
        os.makedirs(path_scann_line_in_image_v)
        
    path_morph = os.path.join(save_path, 'scann_line_morph')
    if not os.path.exists(path_morph):
        os.makedirs(path_morph)

    path_morph_h = os.path.join(save_path, 'scann_line_morph_h')
    if not os.path.exists(path_morph_h):
        os.makedirs(path_morph_h)
    path_morph_v = os.path.join(save_path, 'scann_line_morph_v')
    if not os.path.exists(path_morph_v):
        os.makedirs(path_morph_v)
        
        
    path_contours = os.path.join(save_path, 'contours')
    if not os.path.exists(path_contours):
        os.makedirs(path_contours)
    path_mask = os.path.join(save_path, 'mask')
    if not os.path.exists(path_mask):
        os.makedirs(path_mask)
    path_inner_rect_cand = os.path.join(save_path, 'inner_rect_cand')
    if not os.path.exists(path_inner_rect_cand):
        os.makedirs(path_inner_rect_cand)  
    path_inner_rect_pred = os.path.join(save_path, 'inner_rect_pred')
    if not os.path.exists(path_inner_rect_pred):
        os.makedirs(path_inner_rect_pred)
    path_refer_mask = os.path.join(save_path, 'refer_mask')
    if not os.path.exists(path_refer_mask):
        os.makedirs(path_refer_mask)      
        
    path_scann_line_dbscan = os.path.join(save_path, 'scann_line_dbscan')
    if not os.path.exists(path_scann_line_dbscan):
        os.makedirs(path_scann_line_dbscan)
    path_dbscan_rect = os.path.join(save_path, 'dbscan_rect')
    if not os.path.exists(path_dbscan_rect):
        os.makedirs(path_dbscan_rect)
            
    #cluster_img.save(os.path.join(path_scann_line_dbscan, '{}.png'.format(nid)))
    #cv2.imwrite(os.path.join(path_dbscan_rect, '{}.png'.format(nid)), im_outer_rect) 
    cv2.imwrite(os.path.join(path_refer_mask_refine, '{}.png'.format(nid)), reference_image_2) 
    cv2.imwrite(os.path.join(path_refer_mask, '{}.png'.format(nid)), reference_image) 
    geo_img_h.save(os.path.join(path_scann_line_in_image_h, '{}.png'.format(nid)))
    geo_img_v.save(os.path.join(path_scann_line_in_image_v, '{}.png'.format(nid)))

    cv2.imwrite(os.path.join(path_scann_line_h, '{}.png'.format(nid)), scann_line_img_h) 
    cv2.imwrite(os.path.join(path_fg_mask, '{}.png'.format(nid)), scann_line_bg) 

    cv2.imwrite(os.path.join(path_morph, '{}.png'.format(nid)), close)   
    cv2.imwrite(os.path.join(path_contours, '{}.png'.format(nid)), im_cv_contour)    
    cv2.imwrite(os.path.join(path_inner_rect_cand, '{}.png'.format(nid)), im_inner_rect_cand)    
    cv2.imwrite(os.path.join(path_inner_rect_pred, '{}.png'.format(nid)), im_inner_rect_pred)    

    cv2.imwrite(os.path.join(path_mask, '{}.png'.format(nid)), mask)   
    cv2.imwrite(os.path.join(path_scann_line_v, '{}.png'.format(nid)), scann_line_img_v)  
    cv2.imwrite(os.path.join(path_morph_h, '{}.png'.format(nid)), close_h)   
    cv2.imwrite(os.path.join(path_morph_v, '{}.png'.format(nid)), close_v)   

    return tp, fp, fn, pred_height, actual_height, gdf

    #cv2.destroyAllWindows()
    


def load_label(root, nid):
    path_label = os.path.join(root, 'labels', '{}.txt'.format(nid))
    res = list()
    with open(path_label, 'r') as file:
        tmp = file.readlines()
        if len(tmp)==0:
            return None
        for line in tmp:
            a = list(map(float, line.split(' ')[1:]))
            res.append(np.array([a]))
    res = np.concatenate(res, axis=0)
    return res

    
    

def process_all(root):
    dat_path_root = os.path.join(root, 'depth_dat')
    geo_path_root = os.path.join(root, 'geometry_image')
    depth_path_root = os.path.join(root, 'depth_image')
    density_path_root = os.path.join(root, 'density_image')

    info_root = os.path.join(root, 'info')
    dat_files = os.listdir(dat_path_root)
    dat_files = sorted(dat_files, key=lambda x:int(x.split('_')[0]))
    true_positive = 0
    false_positive = 0
    flase_negative = 0
    total_gdf = pd.DataFrame()
    pred_hs = list()
    actual_hs = list()
    for idx, file in enumerate(dat_files):
        nid = int(file.split('_')[0])
        print(idx, '... ', nid)
        #if nid !=198:
        #    continue
        path = os.path.join(dat_path_root, file)
        geo_path = os.path.join(geo_path_root, '{}.png'.format(nid))
        depth_path = os.path.join(depth_path_root, '{}.png'.format(nid))
        density_path = os.path.join(depth_path_root, '{}.png'.format(nid))
        info_path = os.path.join(info_root, '{}_info.dat'.format(nid))
        tp, fp, fn, pred_height, actual_height, gdf = processing(path, geo_path, depth_path,density_path, info_path, save_path_root, nid)
        if not gdf.empty:
            total_gdf = total_gdf.append(gdf)
            total_gdf = total_gdf.reset_index(drop=True)
            total_gdf['id'] = total_gdf.index
        true_positive += tp
        false_positive += fp
        flase_negative += fn
        if pred_height is not None:
            pred_hs.append(pred_height)
            actual_hs.append(actual_height)
        print("")
    P = true_positive/(true_positive+false_positive)
    R = true_positive/(true_positive+flase_negative)
    print("precision:", P)
    print('recall:', R)
    F1 = (2*P*R)/(P+R)
    print("F1:", F1)
    #print('num windows:', true_positive+flase_negative)
    pred_hs = np.concatenate(pred_hs)#+(config.scann_height-1)
    actual_hs = np.concatenate(actual_hs)
    assert true_positive==len(pred_hs)
    
    error_h = (pred_hs-actual_hs)*0.02

    abs_error_h = np.abs(error_h)
    print('height error: ', abs_error_h.mean(), error_h.mean())
    print('height std: ', abs_error_h.std(), error_h.std())
    with open(os.path.join(save_path_root, 'error_height_scanline'), 'wb') as f:
        pickle.dump(error_h, f)
    
    '''
    error_h = np.round(error_h, 2)
    plt.hist(error_h, bins=int((error_h.max()-error_h.min())/0.02), facecolor="blue", edgecolor="black", alpha=0.7)
    plt.xlabel("error")
    # 显示纵轴标签
    plt.ylabel("num")
    # 显示图标题
    plt.title("num/error")
    plt.show()
    plt.savefig(os.path.join(save_path_root, 'hist_num_error.png'))
    '''
    total_gdf.to_file('./total_windows.geojson', driver='GeoJSON')


    
process_all(root)
#processing('./56_z_ransac.dat')

'''
precision: 0.8861538461538462
recall: 0.7619047619047619
F1: 0.8193456614509246
num windows: 756
height error:  0.07034722222222223

precision: 0.873546511627907
recall: 0.7992021276595744
F1: 0.8347222222222221
num windows: 752
height error:  0.07091514143094842


precision: 0.881203007518797
recall: 0.7710526315789473
F1: 0.8224561403508771
num windows: 760
height error:  0.07334470989761092 -0.014368600682593857
height std:  0.12362585556770712 0.14302566884736617

'''


