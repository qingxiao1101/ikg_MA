# -*- coding: utf-8 -*-
import pickle
import numpy as np

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

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
from sklearn.neighbors import KDTree
import geopandas
from shapely import geometry
from shapely.geometry import LineString
import pandas as pd
from points_find_fn import Point_Find_Fn
from lib.ply import write_points_ddd

save_path = './tmp/'
ply_update_value_root = './aligned_update_values/'
ply_ground_als_root = './aligned_abc/'
dets_root = '/Volumes/Qing Xiao/ikg/4_detection/all_dataset/train/final_train_predictions/txt/' 
depth_root = '/Volumes/Qing Xiao/ikg/4_detection/all_dataset/train/depth_dat/'
info_root = '/Volumes/Qing Xiao/ikg/4_detection/all_dataset/train/info/'

test_dets_root = '/Volumes/Qing Xiao/ikg/4_detection/all_dataset/test/prediction/fpn50_txt/' 
test_depth_root = '/Volumes/Qing Xiao/ikg/4_detection/all_dataset/test/depth_dat/'
test_info_root = '/Volumes/Qing Xiao/ikg/4_detection/all_dataset/test/info/'

save_win_ply = False
gps2als_shift = 43.5341477257

ply_update_values = Point_Find_Fn(ply_update_value_root)
ply_ground_als = Point_Find_Fn(ply_ground_als_root)

path_window_geojson = os.path.join(save_path, 'window_geojson_ALS')
if not os.path.exists(path_window_geojson):
   os.makedirs(path_window_geojson)
   
path_window_geojson_test = os.path.join(save_path, 'window_geojson_ALS_test')
if not os.path.exists(path_window_geojson_test):
   os.makedirs(path_window_geojson_test)
   
path_window_ply = os.path.join(save_path, 'window_ply')
if not os.path.exists(path_window_ply):
   os.makedirs(path_window_ply)

invalid_nids = list()
outlier_window = list()
def load_prediction(path_preds):
    res = list()
    with open(path_preds, 'r') as file:
        tmp = file.readlines()
        if len(tmp)==0:
            return None
        for line in tmp:
            #if int(line.split(' ')[0])==3:
            #    return None
            a = list(map(float, line.split(' ')))
            res.append(np.array([a]))
    res = np.concatenate(res, axis=0)
    return res

def generate_window_line_ply(start, end):
    tmp = list()
    for s, e in zip(start, end):
        x = np.arange(s[0], e[0], 0.001, dtype=np.float32)
        y = np.array([s[1]]*len(x))
        z = np.array([s[2]]*len(x))
        aug = np.array([1]*len(x))
        one_win = np.stack([x,y,z, aug]).T
        one_win_add = one_win.copy()
        one_win_add[:, 1] += 0.005
        one_win_add2 = one_win.copy()
        one_win_add2[:, 1] -= 0.005
        tmp.append(one_win)
        tmp.append(one_win_add)
        tmp.append(one_win_add2)
    tmp = np.concatenate(tmp, axis=0)
    return tmp
        

def window_lines_per_image(preds, scores, depth_path, info_path, nid):
    """
    preds: predicted boxes: [n, 4]
    scores: box confidence: [n,]
    depth_path: path of saved depth information. NOT depth image
    info_path: record the infos for each image, such as trainsformation matrix, 
                size and so on. 
    ground_heights: the ground height of current facade. 
    """
    global gps2als_shift
    if preds is None:
        return pd.DataFrame()
    
    with open(info_path, 'rb') as f:
        info = pickle.load(f)
    with open(depth_path, 'rb') as f:
        depth_dat = pickle.load(f)

    rows, cols = info['size']
    buttom_edge = info['buttom_edge']
    left_edge = info['left_edge']
    Trans = info['trans_i2o']
    basex = info['original_x']
    basey = info['original_y']
    basez = info['original_z']
    
    #print(left_edge, buttom_edge)
    X0 = 1/config.SCALE*preds[:, 0]+left_edge
    X1 = 1/config.SCALE*preds[:, 2]+left_edge
    Z = 1/config.SCALE*(rows-preds[:, 3]) + buttom_edge

    #center = list(zip((preds[:, 0]+preds[:, 2])//2, preds[:, 3]))
    bg_depth = depth_dat.min()
    Y= list()
    bias = 2
    for e in preds:
        tmp = list(depth_dat[min(int(e[3])+bias, rows-1), int(e[0]):int(e[2])])
        tmp += list(depth_dat[max(int(e[1])-bias, 0), int(e[0]):int(e[2])])
        tmp += list(depth_dat[int(e[1]):int(e[3]), max(int(e[0])-bias, 0)])
        tmp += list(depth_dat[int(e[1]):int(e[3]), min(cols-1, int(e[2])+bias)])
        tmp = [i for i in tmp if i!=bg_depth]
        avg_y = sum(tmp)/(len(tmp)+1e-6)
        Y.append(avg_y)
        
        #dists, inds = tree.query([[(e[0]+e[2])/2.0+basex, avg_y+basey]], k=1)
        #dist = dists.mean(axis=1)
        #ground_height = values[inds].mean(axis=1)
        #win_ground_height.append(values[inds].mean(axis=1))
        
    Y = np.array(Y)
    start = np.zeros((len(X0), 4))
    end = np.zeros((len(X0), 4))
    start[:, 3] = 1
    end[:, 3] = 1
    start[:, 2] = Z.copy()
    end[:, 2] = Z.copy()
    start[:, 0] = X0
    start[:, 1] = Y.copy()
    end[:, 0] = X1
    end[:, 1] = Y.copy()
    if save_win_ply:
        win_line_points = generate_window_line_ply(start, end)
        win_line_points = Trans @ win_line_points.T
        win_line_points = win_line_points.T[:, 0:3]
        win_line_points[:, 0] += basex
        win_line_points[:, 1] += basey
        win_line_points[:, 2] += basez
        write_points_ddd(win_line_points, os.path.join(path_window_ply, '{}_detection.ply'.format(nid)))
    
    o_start = Trans @ start.T
    o_start = o_start.T[:, 0:3]
    o_end = Trans @ end.T
    o_end = o_end.T[:, 0:3]
    # o_start, o_end in GPS coordinate system.
    lines = list()
    widths = X1 - X0
    
    tmp_all_window_points = np.concatenate([o_start,o_end], axis=0)
    tmp_all_window_points[:, 0] += basex
    tmp_all_window_points[:, 1] += basey
    tmp_all_window_points[:, 2] += basez
    fns = ply_update_values.get_fns(tmp_all_window_points[:, :2])
    if len(fns)==0:
        invalid_nids.append(nid)
        return pd.DataFrame()
    u_search_space, u_values = ply_update_values.limit_search_space(tmp_all_window_points[:, :2], fns)
    
    tree = KDTree(u_search_space, leaf_size=2)
    dists, inds = tree.query(tmp_all_window_points[:,:2], k=5)
    dist = dists.mean(axis=1)
    update_values = u_values[inds].mean(axis=1)
    #print("window: ", tmp_all_window_points)
    #print("update_values: ", update_values)
    
    g_search_space, g_values = ply_ground_als.limit_search_space(tmp_all_window_points[:, :2], fns)
    tree = KDTree(g_search_space, leaf_size=2)
    dists, inds = tree.query(tmp_all_window_points[:, :2], k=5)
    dist = dists.mean(axis=1)
    ground_height_als = g_values[inds].mean(axis=1)
    window_height_als = tmp_all_window_points[:, 2] - gps2als_shift
    window_height_als = window_height_als - update_values
    #print("window_height_als :", window_height_als)
    #print("ground_height_als: ", ground_height_als)
    
    ref_ground_height = (window_height_als - ground_height_als).reshape(2, -1)
    window_ref_ground_height = ref_ground_height.mean(axis=0)
    outlier_tmp = np.abs(ref_ground_height[0, :] - ref_ground_height[1, :])
    inlier_inds = np.where(outlier_tmp<0.5, 1, 0).nonzero()[0]
    if len(inlier_inds) > 0:
        if len(inlier_inds) < len(window_ref_ground_height):
            outlier_window.append(nid)
            
        tmp_all_window_points = tmp_all_window_points.reshape(2, -1, 3).transpose(1,0,2)
        tmp_all_window_points = tmp_all_window_points[inlier_inds]
        tmp_all_window_points = tmp_all_window_points.transpose(1,0,2)
        win_start = tmp_all_window_points[0]
        win_end = tmp_all_window_points[1]
        window_ref_ground_height = window_ref_ground_height[inlier_inds]
        widths = widths[inlier_inds]
        scores = scores[inlier_inds]
    else:
        outlier_window.append(nid)
        return pd.DataFrame()
    #widths = list()
    for i in range(len(win_start)):
        #widths.append(o_end[i, 0] - o_start[i, 0])
        line = LineString([(win_start[i, 0], win_start[i, 1]),
                           (win_end[i, 0], win_end[i, 1])])
        lines.append(line)

    line_id = np.arange(0, len(win_start))
    df = pd.DataFrame(line_id,columns=['id'])
    nids = [nid]*len(win_start)
    #df_fns = [fns for _ in range(len(nids))]
    
    df['rel_ground_h'] = window_ref_ground_height
    df['gps_h'] = tmp_all_window_points[:, :, 2].mean(axis=0)
    df['nid'] = np.array(nids)
    df['width'] = np.array(widths)
    df['score'] = scores
    df['fn'] = [','.join(fns) for _ in range(len(nids))]
    gdf = geopandas.GeoDataFrame(df, geometry=lines)
    return gdf

def find_nids(root):
    img_names = os.listdir(root)
    nids = [int(name.split('.')[0]) for name in img_names]
    return sorted(nids)

test_dets_nids = find_nids(test_dets_root)
total_gdf = pd.DataFrame()


print('processing test...')
for nid in test_dets_nids:
    print(nid)
    #if nid != 13:
    #    continue
    
    dets = load_prediction(os.path.join(test_dets_root, '{}.txt'.format(nid)))
    if dets is None:
        continue
    pred_boxes = dets[:, :-1]
    scores = dets[:, -1]
    info_path = os.path.join(test_info_root, '{}_info.dat'.format(nid))
    depth_path = os.path.join(test_depth_root, '{}_depth.dat'.format(nid))
    gdf = window_lines_per_image(pred_boxes, scores, depth_path, info_path, nid)
    if not gdf.empty:
        gdf.to_file(os.path.join(path_window_geojson_test, '{}.geojson'.format(nid)), driver='GeoJSON')
        total_gdf = total_gdf.append(gdf)
        total_gdf = total_gdf.reset_index(drop=True)
        total_gdf['id'] = total_gdf.index


print('processing train...')
dets_nids = find_nids(dets_root)
for nid in dets_nids:
    print(nid)

    dets = load_prediction(os.path.join(dets_root, '{}.txt'.format(nid)))
    if dets is None:
        continue
    pred_boxes = dets[:, :-1]
    scores = dets[:, -1]
    info_path = os.path.join(info_root, '{}_info.dat'.format(nid))
    depth_path = os.path.join(depth_root, '{}_depth.dat'.format(nid))
    #ground_heights_path = os.path.join(ground_height_root, '{}.dat'.format(nid))
    gdf = window_lines_per_image(pred_boxes, scores, depth_path, info_path, nid)
    if not gdf.empty:
        gdf.to_file(os.path.join(path_window_geojson, '{}.geojson'.format(nid)), driver='GeoJSON')
        total_gdf = total_gdf.append(gdf)
        total_gdf = total_gdf.reset_index(drop=True)
        total_gdf['id'] = total_gdf.index

print('invalid nids:', invalid_nids)
print('outlier_window nids:', outlier_window)
total_gdf.to_file('./tmp/total_windows.geojson', driver='GeoJSON')

'''
outlier_window nids: [524, 833, 1196, 1643, 1647, 1670, 1827, 1828, 
                      1887, 1912, 1941, 2127, 2376, 2391]

'''


