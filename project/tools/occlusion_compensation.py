# -*- coding: utf-8 -*-

import xlrd
import sys
sys.path.insert(0, '../all_segmetations')
import geopandas
from shapely import geometry
from shapely.geometry import LineString
import pandas as pd
import geojson
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import utils 
from tqdm import tqdm
import pickle
import math
import config
from skimage import io
import matplotlib.pyplot as plt

from sklearn import linear_model
import pcl
import random
import scann_line.ransac as ransac

parse_root = '/Volumes/Qing Xiao/ikg/1_parse/parse_dumpfiles_adjusted/'
root = '/Volumes/Qing Xiao/ikg/6_tmp/intermediate_splitover30/'
geo_root = '/Volumes/Qing Xiao/ikg/6_tmp/geojson_splitover30_total_overlap/'
save_root_2d = '/Volumes/Qing Xiao/ikg/4_detection/tmp/'


df = geopandas.read_file(open(os.path.join(geo_root, 'merge_post.geojson')))
with open(os.path.join(geo_root, 'merge_post.dat'), 'rb') as f:
    match = pickle.load(f)


file = 'qgis_rows_v2.xls'

def read_excel_id(file):
    wb = xlrd.open_workbook(filename=file)
    sheet1 = wb.sheet_by_index(0)
    cols = sheet1.col_values(1)
    ids = [int(x) for x in cols[1:]]
    return ids    





def calculate_matrix(inliers, m, norm_0, ref_z, direction_z):
    a, b, c, d = m
    x_max = inliers[np.argmax(inliers[:, 0])]
    x_min = inliers[np.argmin(inliers[:, 0])]
            
    y_max = inliers[np.argmax(inliers[:, 1])]
    y_min = inliers[np.argmin(inliers[:, 1])]
    
    if utils.l2_distance_2d(x_max[:2], x_min[:2]) < utils.l2_distance_2d(y_max[:2], y_min[:2]):
        b_min, b_max =  y_min, y_max
    else:
        b_min, b_max =  x_min, x_max
    plane_norm = np.array([a, b, c])/np.linalg.norm(np.array([a, b, c]))
    ref_x = b_min[0] 
    ref_y = b_min[1]
    ref_original = np.array([ref_x, ref_y, ref_z])
    x_dir = np.array([b_max[0],  b_max[1], ref_z])
    dot = np.dot(plane_norm, norm_0)
    b1 = (x_dir - ref_original) / np.linalg.norm(x_dir - ref_original) 
    b2 = plane_norm if dot >0 else - plane_norm
    b3 = np.cross(b2, b1)
    dot2 = np.dot(b3, direction_z)
    b3 = b3 if dot2> 0 else -b3
    T = np.zeros((4,4), dtype=np.float32)
    T[0:3, 0] = b1
    T[0:3, 1] = b2
    T[0:3, 2] = b3
    T[0:3, 3] = ref_original
    T[3,3] = 1
    inv_T = np.linalg.inv(T)
    return T, inv_T, b_min, b_max


def query_nid(num_id):
    segs = match[num_id]
    project_points = list()
    remainder_points = list()
    base_line = segs[0]
    index=df[df['id']==num_id].index[0]
    df_data = df.loc[index]
    #df_data = df[df['id']==num_id] 
    #inv_T = np.zeros((4,4))
    direction = False
    norm_0 = None
    direction_z = None
    ref_z = None
    
    for i, (name, nid) in enumerate(segs):
        print('\rseg: {}/{}'.format(i, len(segs)-1), end='')
        with open(os.path.join(root, name, "tmp_building.dat"), 'rb') as f:
            seg, bounding = pickle.load(f)
        with open(os.path.join(parse_root, name, "coordinate.dat"), 'rb') as f:
            coor = pickle.load(f)
        with open(os.path.join(parse_root, name, "head_info.dat"), 'rb') as f:
            header = pickle.load(f)
        norm, bounding2d, bounding3d, _, _ = bounding[nid]
        
        bg = np.zeros(coor.shape)
        bg = np.where(coor==bg, 0, 1)
        bg_mask = (bg==0).all(axis=2)
        bg_mask = np.logical_not(bg_mask)  #背景处为0， 其余为1
        _, bounding2d, _, _, _ = bounding[nid]
        
        if i == 0:
            base_x, base_y, base_z = header["original_x"], header["original_y"], header["original_z"]
            if bounding3d['x_min'][2] > bounding3d['x_max'][2]:
                direction_z = bounding3d['x_min'] - bounding3d['x_max']
            else:
                direction_z = bounding3d['x_max'] - bounding3d['x_min']
            ref_z = df_data['ref_h'] - header["original_z"]

            mask_seg = np.where(seg==nid, 1, 0).reshape(-1)
            idx_seg = mask_seg.nonzero()[0]
            points_seg = coor.reshape(-1, 3)[idx_seg, :]
            model = ransac.RANSAC()
            m, inlier_mask, _ = model.run(points_seg, inlier_thres=config.inliers_thres, max_iterations=config.max_iterations)
            norm_0 = norm
            inliers = points_seg[inlier_mask]
            
            T0, inv_T0, _, _ = calculate_matrix(inliers, m, norm_0, ref_z, direction_z)
            '''
            homo_inliers = np.ones((inliers.shape[0],4), dtype=np.float32)
            homo_inliers[:, 0:3] = inliers
            proj_3d = inv_T0 @ homo_inliers.T 
            proj_3d = proj_3d.T[:, 0:3]
            xyz0 = T0[:3]
            xyz0[:, 0] += xyz0[:, 3]
            xyz0[:, 1] += xyz0[:, 3]
            xyz0[:, 2] += xyz0[:, 3]
            xyz0 = xyz0.T
            xyz0_aug = np.ones((xyz0.shape[0], 4))
            xyz0_aug[:, 0:3] = xyz0
            pro_xyz0 = inv_T0 @ xyz0_aug.T
            pro_xyz0 = pro_xyz0.T[:,:3]
            
            fig = plt.figure('graph')
            
            ax = fig.add_subplot(111, projection='3d') 
            
            ax.set_title(r'3d')
            plt.plot(proj_3d[:,0],  proj_3d[:,1], proj_3d[:,2], '.', color='coral')
            plt.plot([pro_xyz0[0, 0], pro_xyz0[3, 0]], [pro_xyz0[0, 1], pro_xyz0[3, 1]], [pro_xyz0[0, 2], pro_xyz0[3, 2]], '-r', label='x')
            plt.plot([pro_xyz0[1, 0], pro_xyz0[3, 0]], [pro_xyz0[1, 1], pro_xyz0[3, 1]], [pro_xyz0[1, 2], pro_xyz0[3, 2]], '-g', label='y')   
            plt.plot([pro_xyz0[2, 0], pro_xyz0[3, 0]], [pro_xyz0[2, 1], pro_xyz0[3, 1]], [pro_xyz0[2, 2], pro_xyz0[3, 2]], '-b', label='z') 
        
            ax.legend()
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.text(0, 0, 0, 'x')  # (0,0,0)
            ax.view_init(60,0)
        
            plt.show()
            raise InterruptedError()
            '''

        mask_seg = np.where(seg==nid, 1, 0).reshape(-1)
        idx_seg = mask_seg.nonzero()[0]
        points_seg = coor.reshape(-1, 3)[idx_seg, :]
        points_seg[:, 0] += header["original_x"] - base_x
        points_seg[:, 1] += header["original_y"] - base_y
        points_seg[:, 2] += header["original_z"] - base_z
        
        home_points_seg = np.ones((points_seg.shape[0],4), dtype=np.float32)

        home_points_seg[:, 0:3] = points_seg
        proj_points_seg = inv_T0 @ home_points_seg.T
        proj_points_seg = proj_points_seg.T[:, 0:3]

        
        
        mask_bb = np.zeros(seg.shape)
        
        #mask_bb[bounding2d['x_min'].x:bounding2d['x_max'].x, 
        #         bounding2d['y_min'].y:bounding2d['y_max'].y] = 1
        adjust = 100
        mask_bb[max(0, bounding2d['x_min'].x-adjust):min(bounding2d['x_max'].x+adjust, seg.shape[0]), 
                 max(0, bounding2d['y_min'].y-adjust):min(bounding2d['y_max'].y+adjust, seg.shape[1])] = 1

        mask_bb = np.logical_and(mask_bb, bg_mask).reshape(-1)
        idx_bb = mask_bb.nonzero()[0]
        points_bb = coor.reshape(-1, 3)[idx_bb, :]
        points_bb[:, 0] += header["original_x"] - base_x
        points_bb[:, 1] += header["original_y"] - base_y
        points_bb[:, 2] += header["original_z"] - base_z
        
        home_points_bb = np.ones((points_bb.shape[0],4), dtype=np.float32)
        
        home_points_bb[:, 0:3] = points_bb
        proj_points_bb = inv_T0 @ home_points_bb.T
        proj_points_bb = proj_points_bb.T[:, 0:3]
        
        
        range_max = np.max(proj_points_seg[:, 0])
        range_min = np.min(proj_points_seg[:, 0])
        mask_effect = np.where(proj_points_bb[:,0]<=range_max, 1, 0)

        mask_effect = np.where(proj_points_bb[:, 0]>= range_min, mask_effect, 0)
        idx_effect = mask_effect.nonzero()[0]
        proj_points = proj_points_bb[idx_effect]
        remainder = points_bb[idx_effect]
        
        #cloud = pcl.PointCloud()
        #loud.from_array(remainder)
        #pcl.save(cloud, os.path.join(save_root_2d, 'ply', '{}_sub_{}.ply'.format(num_id, i)), format="ply")
    

        remainder_points.append(remainder)
        project_points.append(proj_points)
    print(" ")
    project_points = np.concatenate(project_points, axis=0)
    remainder_points = np.concatenate(remainder_points, axis=0)
    
    path_ply_original = os.path.join(save_root_2d, 'ply_original')
    if not os.path.exists(path_ply_original):
        os.makedirs(path_ply_original)
    cloud = pcl.PointCloud()
    '''
    tmp_remainder_points = remainder_points.copy()
    tmp_remainder_points[:, 0] += base_x
    tmp_remainder_points[:, 1] += base_y
    tmp_remainder_points[:, 2] += base_z
    '''
    cloud.from_array(remainder_points)
    pcl.save(cloud, os.path.join(path_ply_original, '{}_original.ply'.format(num_id)), format="ply")
    

    proj_3d = project_points[:, 0:3]
    average_height = np.median(proj_3d[:, 1])


    index_front_threshold = np.where(proj_3d[:, 1] > average_height-1.0, 1, 0).nonzero()[0]
    #print("idx: ", index_front_threshold.shape)
    proj_3d = proj_3d[index_front_threshold]
    remainder_points = remainder_points[index_front_threshold]
    
    index_under_threshold = np.where(remainder_points[:, 2] < np.min(remainder_points[:, 2])+2.0, 1, 0).nonzero()[0]
    index_above_threshold = np.where(remainder_points[:, 2] >= np.min(remainder_points[:, 2])+2.0, 1, 0).nonzero()[0]
    under_points = remainder_points[index_under_threshold]
    above_points = remainder_points[index_above_threshold]
    under_proj_3d = proj_3d[index_under_threshold]
    above_proj_3d = proj_3d[index_above_threshold]
    
    index_under_behind_threshold = np.where(under_proj_3d[:, 1] < average_height+0.5, 1, 0).nonzero()[0]
    under_proj_3d = under_proj_3d[index_under_behind_threshold]
    under_points = under_points[index_under_behind_threshold]
    index_above_behind_threshold = np.where(above_proj_3d[:, 1] < average_height+2.5, 1, 0).nonzero()[0]
    above_proj_3d = above_proj_3d[index_above_behind_threshold]
    above_points = above_points[index_above_behind_threshold]
    remainder_points = np.concatenate((under_points, above_points), axis=0)
    
    path_ply_in_depth = os.path.join(save_root_2d, 'ply_in_depth')
    if not os.path.exists(path_ply_in_depth):
        os.makedirs(path_ply_in_depth)
    cloud = pcl.PointCloud()
    '''
    tmp_remainder_points = remainder_points.copy()
    tmp_remainder_points[:, 0] += base_x
    tmp_remainder_points[:, 1] += base_y
    tmp_remainder_points[:, 2] += base_z
    '''
    cloud.from_array(remainder_points)
    pcl.save(cloud, os.path.join(path_ply_in_depth, '{}_in_depth.ply'.format(num_id)), format="ply")
    
    m, inlier_mask, flag = model.run(remainder_points, inlier_thres=config.inliers_thres, max_iterations=config.max_iterations)
    inliers = remainder_points[inlier_mask]
    #if not flag:
    #    T, inv_T = T0, inv_T0
    #else:
    T_final, inv_T_final, b_min, b_max = calculate_matrix(inliers, m, norm_0, ref_z, direction_z)
    
    path_ply_inliers = os.path.join(save_root_2d, 'ply_inliers')
    if not os.path.exists(path_ply_inliers):
        os.makedirs(path_ply_inliers)
    '''
    tmp_inliers = inliers.copy()
    tmp_inliers[:, 0] += base_x
    tmp_inliers[:, 1] += base_y
    tmp_inliers[:, 2] += base_z
    '''
    cloud.from_array(inliers)
    pcl.save(cloud, os.path.join(path_ply_inliers, '{}_inliers.ply'.format(num_id)), format="ply")
    
    homo_inliers = np.ones((inliers.shape[0],4), dtype=np.float32)
    homo_inliers[:, 0:3] = inliers
    proj_3d_inliers = inv_T_final @ homo_inliers.T 
    proj_3d_inliers = proj_3d_inliers.T[:, 0:3]
    ref_plane = np.mean(proj_3d_inliers[:, 1])
    homo_remainder_points = np.ones((remainder_points.shape[0],4), dtype=np.float32)
    homo_remainder_points[:, 0:3] = remainder_points
    proj_3d = inv_T_final @ homo_remainder_points.T 
    proj_3d = proj_3d.T[:, 0:3]
    #proj_3d = np.concatenate((under_proj_3d, above_proj_3d), axis=0)

    range_x_min = np.min(proj_3d[:,0])
    range_x_max = np.max(proj_3d[:,0])
    range_y_min = np.min(proj_3d[:,2])
    range_y_max = np.max(proj_3d[:,2])
    

    #print(range_x_min, range_x_max, range_y_min, range_y_max)
    bg = np.min(proj_3d[:,1]) - 0.1
    img_2d = np.ones((int(math.ceil(range_y_max-range_y_min)*config.SCALE)+1, 
                     int(math.ceil((range_x_max-range_x_min)*config.SCALE))+1), dtype=np.float32)*bg
    fill_flag = np.zeros(img_2d.shape)
    density_img = np.zeros(img_2d.shape, dtype=np.uint16)
    
    density = float(len(proj_3d)) / (img_2d.shape[0]*img_2d.shape[1])
    for i, (x,z,y) in enumerate(proj_3d):
            
            x = int(round((x-range_x_min)*config.SCALE))
            y = int(round((y-range_y_min)*config.SCALE))

            if fill_flag[img_2d.shape[0]-y-1,x] == 0.5 and \
                abs(img_2d[img_2d.shape[0]-y-1,x]-average_height) < abs(z-average_height):
                    pass
            else:
                img_2d[img_2d.shape[0]-y-1,x] = z 
            fill_flag[img_2d.shape[0]-y-1,x] = 0.5
            density_img[img_2d.shape[0]-y-1,x] += 1
    
    path_depth_dat = os.path.join(save_root_2d, 'depth_dat')
    if not os.path.exists(path_depth_dat):
        os.makedirs(path_depth_dat)    
    with open(os.path.join(path_depth_dat, '{}_depth.dat'.format(num_id)), 'wb') as f:
        pickle.dump(img_2d.copy(), f)
    path_info = os.path.join(save_root_2d, 'info')
    if not os.path.exists(path_info):
        os.makedirs(path_info)        
    with open(os.path.join(path_info, '{}_info.dat'.format(num_id)), 'wb') as f:
        pickle.dump({
            'size': img_2d.shape,
            'density': density,  # density of point cloud
            'resolution': 1.0/config.SCALE,
            'left_edge':range_x_min,
            'buttom_edge':range_y_min,
            'trans_i2o': T_final,
            'trans_o2i': inv_T_final,
            'original_x':base_x, 
            'original_y':base_y, 
            'original_z':base_z,
            'ref_road':ref_z,
            'ref_plane': ref_plane
            }, f)
        
    #cloud = pcl.PointCloud()
    #cloud.from_array(remainder_points)
    #pcl.save(cloud, os.path.join(save_root_2d, 'ply', '{}.ply'.format(num_id)), format="ply")
    
    img_2d = utils.rescale(img_2d)
    img_2d = np.round(img_2d*255).astype(np.uint8)
    
    density_img = utils.rescale(density_img)
    density_img = np.round(density_img*255).astype(np.uint8)
    
    #img_2d = np.where(img_2d<1.0, img_2d, 0.999)
    #img_2d = np.where(img_2d>0.001, img_2d, 0.001)
    #with open(os.path.join(save_root_2d, '{}_project3d.dat'.format(num_id)), 'wb') as f:
    #    pickle.dump(proj_3d, f)
    path_depth_img = os.path.join(save_root_2d, 'depth_image')
    if not os.path.exists(path_depth_img):
        os.makedirs(path_depth_img)    
    path_density_img = os.path.join(save_root_2d, 'density_image')
    if not os.path.exists(path_density_img):
        os.makedirs(path_density_img) 
    path_geometry_image = os.path.join(save_root_2d, 'geometry_image')
    if not os.path.exists(path_geometry_image):
        os.makedirs(path_geometry_image)    
    io.imsave(os.path.join(path_depth_img, '{}.png'.format(num_id)), img_2d)
    io.imsave(os.path.join(path_density_img, '{}.png'.format(num_id)), density_img)
    io.imsave(os.path.join(path_geometry_image, '{}.png'.format(num_id)), fill_flag)
    print("done...{}".format(num_id))


query_nid(28)
raise InterruptedError()

    