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
import random
import lib.ransac as ransac

MODE = "test"  # "test"
PATH = "/Volumes/Qing Xiao/ikg"

parse_root = PATH + '/1_parse/parse_dumpfiles_adjusted/'
root = PATH + '/6_tmp/intermediate_splitover30/'
geo_root = PATH + '/6_tmp/geojson_splitover30/'
save_root_2d = PATH + '/3_image_generation/{}/'.format(MODE)


df = geopandas.read_file(open(os.path.join(geo_root, 'merge_post.geojson')))
with open(os.path.join(geo_root, 'merge_post.dat'), 'rb') as f:
    match = pickle.load(f)

#print(df.shape)

test_df = geopandas.read_file(open(os.path.join(geo_root, '{}_facades_nid.geojson'.format(MODE))))

test_facades_nids = list(test_df['id'])
print(len(test_facades_nids))
facade_lines = list()

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

def generate_depth_image(coor, head):
    tmp = coor - head
    tmp = tmp**2
    tmp = tmp.sum(axis=2)
    dis = np.sqrt(tmp)
    return dis


def intersection_lines_plane(plane_equation, head, point, p0):
    n = plane_equation[:3]
    #d = plane_equation[3]
    l0 = head
    tmp = np.sqrt(np.sum((point - head)**2, axis=1))[:, None]
    l = (point - head)/tmp
    t = np.dot((p0[None, :]-l0), n)/np.dot(l, n)
    positive_inds = np.where(t>0, 1, 0)#.nonzero()[0]
    limit_inds = np.where(t>10, 0, positive_inds).nonzero()[0]
    #saved_inds = np.logical_and(positive_inds, limit_inds).nonzero()[0]
    t = t[limit_inds]
    l0 = l0[limit_inds]
    l = l[limit_inds]
    p = l0+l*t[:, None]
    return p.astype(np.float32)


def query_2d(num_id):
    segs = match[num_id]
    project_points = list()
    remainder_points = list()
    remainder_heads = list()
    base_line = segs[0]
    index=df[df['id']==num_id].index[0]
    df_data = df.loc[index]
    #df_data = df[df['id']==num_id] 
    #inv_T = np.zeros((4,4))
    direction = False
    norm_0 = None
    direction_z = None
    ref_z = None
    reflectances_effect = list()
    normals_effect = list()
    depths_effect = list()
    for i, (name, nid) in enumerate(segs):
        print('\rseg: {}/{}'.format(i, len(segs)-1), end='')
        with open(os.path.join(root, name, "tmp_building.dat"), 'rb') as f:
            seg, bounding = pickle.load(f)
        with open(os.path.join(parse_root, name, "coordinate.dat"), 'rb') as f:
            coor = pickle.load(f)
        with open(os.path.join(parse_root, name, "reflectance.dat"), 'rb') as f:
            o_reflectance = pickle.load(f)
        with open(os.path.join(parse_root, name, "normal.dat"), 'rb') as f:
            o_normal = pickle.load(f)
        with open(os.path.join(parse_root, name, "head.dat"), 'rb') as f:
            o_head = pickle.load(f)
        o_depth = generate_depth_image(coor, o_head) 
        
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
            
            T0, inv_T0, _b_min, _b_max = calculate_matrix(inliers, m, norm_0, ref_z, direction_z)
            facade_line = LineString([(_b_min[0]+base_x,_b_min[1]+base_y), (_b_max[0]+base_x,_b_max[1]+base_y)])
            facade_lines.append(facade_line)
            with open(os.path.join('./','tmp_facade_line_{}.txt'.format(MODE)),'a+') as file_handle:   # .txt可以不自己新建,代码会自动新建
                file_handle.write("{}  {}".format(num_id, facade_line))     # 写入
                file_handle.write('\n')
      
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
        adjust = 0
        mask_bb[max(0, bounding2d['x_min'].x-adjust):min(bounding2d['x_max'].x+adjust, seg.shape[0]), 
                 max(0, bounding2d['y_min'].y-adjust):min(bounding2d['y_max'].y+adjust, seg.shape[1])] = 1

        mask_bb = np.logical_and(mask_bb, bg_mask).reshape(-1)
        idx_bb = mask_bb.nonzero()[0]
        points_bb = coor.reshape(-1, 3)[idx_bb, :]
        
        reflectance_bb = o_reflectance.reshape(-1,)[idx_bb]
        normal_bb = o_normal.reshape(-1, 3)[idx_bb, :]
        depth_bb = o_depth.reshape(-1, )[idx_bb]
        
        points_bb[:, 0] += header["original_x"] - base_x
        points_bb[:, 1] += header["original_y"] - base_y
        points_bb[:, 2] += header["original_z"] - base_z

        heads_bb = o_head.reshape(-1, 3)[idx_bb, :]
        heads_bb[:, 0] += header["original_x"] - base_x
        heads_bb[:, 1] += header["original_y"] - base_y
        heads_bb[:, 2] += header["original_z"] - base_z        
                
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
        remainder_head = heads_bb[idx_effect]

        reflectances_effect.append(reflectance_bb[idx_effect])
        normals_effect.append(normal_bb[idx_effect])
        depths_effect.append(depth_bb[idx_effect])        
        remainder_heads.append(remainder_head)
        remainder_points.append(remainder)
        project_points.append(proj_points)
    print(" ")
    project_points = np.concatenate(project_points, axis=0)
    remainder_points = np.concatenate(remainder_points, axis=0)
    remainder_heads = np.concatenate(remainder_heads, axis=0)

    reflectances_effect = np.concatenate(reflectances_effect, axis=0)
    normals_effect = np.concatenate(normals_effect, axis=0)
    depths_effect = np.concatenate(depths_effect, axis=0)
    
    path_ply_original = os.path.join(save_root_2d, 'ply_original')
    if not os.path.exists(path_ply_original):
        os.makedirs(path_ply_original)
    #cloud.from_array(remainder_points)
    #pcl.save(cloud, os.path.join(path_ply_original, '{}_original.ply'.format(num_id)), format="ply")
    

    proj_3d = project_points[:, 0:3]
    average_height = np.median(proj_3d[:, 1])


    index_front_threshold = np.where(proj_3d[:, 1] > average_height-1.0, 1, 0).nonzero()[0]
    #print("idx: ", index_front_threshold.shape)
    proj_3d = proj_3d[index_front_threshold]
    remainder_points = remainder_points[index_front_threshold]
    remainder_heads = remainder_heads[index_front_threshold]  # remove points which behind facade with distance large than 1m.

    reflectances_effect = reflectances_effect[index_front_threshold]
    normals_effect = normals_effect[index_front_threshold]
    depths_effect = depths_effect[index_front_threshold]
    
    index_under_threshold = np.where(remainder_points[:, 2] < np.min(remainder_points[:, 2])+2.0, 1, 0).nonzero()[0]
    index_above_threshold = np.where(remainder_points[:, 2] >= np.min(remainder_points[:, 2])+2.0, 1, 0).nonzero()[0]
    under_points = remainder_points[index_under_threshold]
    under_reflectances = reflectances_effect[index_under_threshold]
    under_normals = normals_effect[index_under_threshold]
    under_depths = depths_effect[index_under_threshold]
    
    under_heads = remainder_heads[index_under_threshold]
    above_heads = remainder_heads[index_above_threshold]
    
    above_points = remainder_points[index_above_threshold]
    above_reflectances = reflectances_effect[index_above_threshold]
    above_normals = normals_effect[index_above_threshold]
    above_depths = depths_effect[index_above_threshold]
    
    
    under_proj_3d = proj_3d[index_under_threshold]
    above_proj_3d = proj_3d[index_above_threshold]
    
    index_under_behind_threshold = np.where(under_proj_3d[:, 1] < average_height+0.5, 1, 0).nonzero()[0]
    index_under_front_threshold = np.where(under_proj_3d[:, 1] >= average_height+0.5, 1, 0).nonzero()[0]

    under_proj_3d = under_proj_3d[index_under_behind_threshold]
    occlusion_points_1 = under_points[index_under_front_threshold]
    occlusion_heads_1 = under_heads[index_under_front_threshold]
    
    under_points = under_points[index_under_behind_threshold]
    under_reflectances = under_reflectances[index_under_behind_threshold]
    under_normals = under_normals[index_under_behind_threshold]
    under_depths = under_depths[index_under_behind_threshold]
    
    
    index_above_behind_threshold = np.where(above_proj_3d[:, 1] < average_height+2.5, 1, 0).nonzero()[0]
    index_above_front_threshold = np.where(above_proj_3d[:, 1] >= average_height+2.5, 1, 0).nonzero()[0]

    occlusion_points_2 = above_points[index_above_front_threshold]
    occlusion_heads_2 = above_heads[index_above_front_threshold]

    above_proj_3d = above_proj_3d[index_above_behind_threshold]
    above_points = above_points[index_above_behind_threshold]
    above_reflectances = above_reflectances[index_above_behind_threshold]
    above_normals = above_normals[index_above_behind_threshold]
    above_depths = above_depths[index_above_behind_threshold]
    
    
    remainder_points = np.concatenate((under_points, above_points), axis=0)
    reflectances = np.concatenate((under_reflectances, above_reflectances), axis=0)
    normals = np.concatenate((under_normals, above_normals), axis=0)
    depths = np.concatenate((under_depths, above_depths), axis=0)
    
    m, inlier_mask, flag = model.run(remainder_points, inlier_thres=config.inliers_thres, max_iterations=config.max_iterations)
    inliers = remainder_points[inlier_mask]

    T_final, inv_T_final, b_min, b_max = calculate_matrix(inliers, m, norm_0, ref_z, direction_z)
    #facade_line = LineString([(b_min[0]+base_x,b_min[1]+base_y), (b_max[0]+base_x,b_max[1]+base_y)])
    #facade_lines.append(facade_line)

    
    if len(occlusion_points_1) > 0 and len(occlusion_points_2)>0:
        occlusion_points = np.concatenate([occlusion_points_1, occlusion_points_2])
        occlusion_heads = np.concatenate([occlusion_heads_1, occlusion_heads_2])
    elif len(occlusion_points_1) > 0:
        occlusion_points = occlusion_points_1
        occlusion_heads = occlusion_heads_1
    else:
        occlusion_points = occlusion_points_2
        occlusion_heads = occlusion_heads_2
    
    '''
    path_ply_occlusion = os.path.join(save_root_2d, 'ply_occlusion')
    if not os.path.exists(path_ply_occlusion):
        os.makedirs(path_ply_occlusion)
    cloud = pcl.PointCloud()
    if len(occlusion_points) > 0:
        cloud.from_array(occlusion_points)
        pcl.save(cloud, os.path.join(path_ply_occlusion, '{}_occlusion.ply'.format(num_id)), format="ply")

    path_ply_o_head = os.path.join(save_root_2d, 'ply_o_head')
    if not os.path.exists(path_ply_o_head):
        os.makedirs(path_ply_o_head)
    cloud = pcl.PointCloud()
    if len(occlusion_heads) > 0:
        cloud.from_array(occlusion_heads)
        pcl.save(cloud, os.path.join(path_ply_o_head, '{}_o_head.ply'.format(num_id)), format="ply")
    
    '''
    intersection_points = intersection_lines_plane(m, occlusion_heads, occlusion_points, p0=inliers[0])
    a, b, c, d = m
    dis_indices = intersection_points[:, 0]*a + intersection_points[:, 1]*b + intersection_points[:, 2]*c +d 
    dis_inds = np.where(dis_indices<0.2, 1, 0).nonzero()[0]
    intersection_points = intersection_points[dis_inds]
    '''
    path_ply_intersection = os.path.join(save_root_2d, 'ply_intersection')
    if not os.path.exists(path_ply_intersection):
        os.makedirs(path_ply_intersection)
    cloud = pcl.PointCloud()
    if len(intersection_points) > 0:
        cloud.from_array(intersection_points)
        pcl.save(cloud, os.path.join(path_ply_intersection, '{}_intersection.ply'.format(num_id)), format="ply")
    '''
    #remainder_points = np.concatenate([remainder_points, intersection_points])
    
    path_ply_in_depth = os.path.join(save_root_2d, 'ply_in_depth')
    if not os.path.exists(path_ply_in_depth):
        os.makedirs(path_ply_in_depth)
    '''
    tmp_remainder_points = remainder_points.copy()
    tmp_remainder_points[:, 0] += base_x
    tmp_remainder_points[:, 1] += base_y
    tmp_remainder_points[:, 2] += base_z
    '''
    #cloud.from_array(remainder_points)
    #pcl.save(cloud, os.path.join(path_ply_in_depth, '{}_in_depth.ply'.format(num_id)), format="ply")
    
    #m, inlier_mask, flag = model.run(remainder_points, inlier_thres=config.inliers_thres, max_iterations=config.max_iterations)
    #inliers = remainder_points[inlier_mask]
    #if not flag:
    #    T, inv_T = T0, inv_T0
    #else:
    #T_final, inv_T_final, b_min, b_max = calculate_matrix(inliers, m, norm_0, ref_z, direction_z)

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
    
    homo_intersection_points = np.ones((intersection_points.shape[0],4), dtype=np.float32)
    homo_intersection_points[:, 0:3] = intersection_points
    proj_intersection_3d = inv_T_final @ homo_intersection_points.T 
    proj_intersection_3d = proj_intersection_3d.T[:, 0:3]

    range_x_min = np.min(proj_3d[:,0])
    range_x_max = np.max(proj_3d[:,0])
    range_y_min = np.min(proj_3d[:,2])
    range_y_max = np.max(proj_3d[:,2])
    

    #print(range_x_min, range_x_max, range_y_min, range_y_max)
    bg = np.min(proj_3d[:,1]) - 0.1
    img_2d = np.ones((int(math.ceil(range_y_max-range_y_min)*config.SCALE)+1, 
                     int(math.ceil((range_x_max-range_x_min)*config.SCALE))+1), dtype=np.float32)*bg
    fill_flag = np.zeros(img_2d.shape)
    occlusion_mask = np.zeros(img_2d.shape)
    density_img = np.zeros(img_2d.shape, dtype=np.uint16)
    img_reflectance = np.ones(img_2d.shape)*(np.min(reflectances)-1)
    img_norm = np.zeros((*img_2d.shape, 3))
    img_original_depth = np.zeros(img_2d.shape)
    
    density = float(len(proj_3d)) / (img_2d.shape[0]*img_2d.shape[1])
    
    
    for i, (x,z,y) in enumerate(proj_3d):
            
            x = int(round((x-range_x_min)*config.SCALE))
            y = int(round((y-range_y_min)*config.SCALE))

            if fill_flag[img_2d.shape[0]-y-1,x] == 0.5 and \
                abs(img_2d[img_2d.shape[0]-y-1,x]-average_height) < abs(z-average_height):
                    pass
            else:
                img_2d[img_2d.shape[0]-y-1,x] = z 
                img_reflectance[img_2d.shape[0]-y-1,x] = reflectances[i]
                img_norm[img_2d.shape[0]-y-1,x, :] = normals[i]
                img_original_depth[img_2d.shape[0]-y-1,x] = depths[i]
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
    
    img_2d_save1 = utils.rescale(img_2d)
    img_2d_save1 = np.round(img_2d_save1*255).astype(np.uint8)
    
    density_img_save1 = utils.rescale(density_img)
    density_img_save1 = np.round(density_img_save1*255).astype(np.uint8)
    
    
    path_camera_depth = os.path.join(save_root_2d, 'camera_depth')
    if not os.path.exists(path_camera_depth):
        os.makedirs(path_camera_depth) 
    with open(os.path.join(path_camera_depth, '{}.dat'.format(num_id)), 'wb') as f:
        pickle.dump(img_original_depth, f)    
        
    path_normal = os.path.join(save_root_2d, 'normal')
    if not os.path.exists(path_normal):
        os.makedirs(path_normal) 
    io.imsave(os.path.join(path_normal, '{}.png'.format(num_id)), img_norm)
    
    path_reflectance = os.path.join(save_root_2d, 'reflectance')
    if not os.path.exists(path_reflectance):
        os.makedirs(path_reflectance) 
    with open(os.path.join(path_reflectance, '{}.dat'.format(num_id)), 'wb') as f:
        pickle.dump(img_reflectance, f)    
        
    path_depth_img = os.path.join(save_root_2d, 'depth_image')
    if not os.path.exists(path_depth_img):
        os.makedirs(path_depth_img)    
    path_density_img = os.path.join(save_root_2d, 'density_image')
    if not os.path.exists(path_density_img):
        os.makedirs(path_density_img) 
    path_geometry_image = os.path.join(save_root_2d, 'geometry_image')
    if not os.path.exists(path_geometry_image):
        os.makedirs(path_geometry_image)    
    io.imsave(os.path.join(path_depth_img, '{}.png'.format(num_id)), img_2d_save1)
    io.imsave(os.path.join(path_density_img, '{}.png'.format(num_id)), density_img_save1)
    io.imsave(os.path.join(path_geometry_image, '{}.png'.format(num_id)), fill_flag)

    path_occl_mask = os.path.join(save_root_2d, 'occlusion_mask')
    if not os.path.exists(path_occl_mask):
        os.makedirs(path_occl_mask)    
    ##### for occlusion compensation
    
    avg_reflectance = reflectances.mean()
    avg_depth = depths.mean()
    avg_norm = normals.mean(axis=0)
    for i, (x,z,y) in enumerate(proj_intersection_3d):
            if z < -1.0:
                continue
            if x < range_x_min or x > range_x_max:
                continue
            if y < range_y_min or y > range_y_max:
                continue
            
            x = int(round((x-range_x_min)*config.SCALE))
            y = int(round((y-range_y_min)*config.SCALE))
            occlusion_mask[img_2d.shape[0]-y-1,x] = 0.5
            '''
            #if (fill_flag[img_2d.shape[0]-y-1,x] != 0 or fill_flag[img_2d.shape[0]-y-1,max(0, x-1)] != 0 or 
            #    fill_flag[img_2d.shape[0]-y-1,min(x-1, img_2d.shape[1]-1)]) != 0:
            if fill_flag[img_2d.shape[0]-y-1,x] != 0:
                continue
            
            if fill_flag[img_2d.shape[0]-y-1,x] == 0.5 and \
                abs(img_2d[img_2d.shape[0]-y-1,x]-average_height) < abs(z-average_height):
                    pass
            else:
                img_2d[img_2d.shape[0]-y-1,x] = z 
                img_reflectance[img_2d.shape[0]-y-1,x] = avg_reflectance
                img_norm[img_2d.shape[0]-y-1,x, :] = avg_norm
                img_original_depth[img_2d.shape[0]-y-1,x] = avg_depth
            fill_flag[img_2d.shape[0]-y-1,x] = 0.5
            density_img[img_2d.shape[0]-y-1,x] += 1
            '''
    io.imsave(os.path.join(path_occl_mask, '{}.png'.format(num_id)), occlusion_mask)
    
    '''    
    path_depth_dat = os.path.join(save_root_2d, 'occl_comp', 'depth_dat')
    if not os.path.exists(path_depth_dat):
        os.makedirs(path_depth_dat)    
    with open(os.path.join(path_depth_dat, '{}_depth.dat'.format(num_id)), 'wb') as f:
        pickle.dump(img_2d.copy(), f)      
        
    img_2d_save2 = utils.rescale(img_2d)
    img_2d_save2 = np.round(img_2d_save2*255).astype(np.uint8)
    
    density_img_save2 = utils.rescale(density_img)
    density_img_save2 = np.round(density_img_save2*255).astype(np.uint8)
    
    
    path_camera_depth = os.path.join(save_root_2d, 'occl_comp', 'camera_depth')
    if not os.path.exists(path_camera_depth):
        os.makedirs(path_camera_depth) 
    with open(os.path.join(path_camera_depth, '{}.dat'.format(num_id)), 'wb') as f:
        pickle.dump(img_original_depth, f)    
        
    path_normal = os.path.join(save_root_2d, 'occl_comp', 'normal')
    if not os.path.exists(path_normal):
        os.makedirs(path_normal) 
    io.imsave(os.path.join(path_normal, '{}.png'.format(num_id)), img_norm)
    
    path_reflectance = os.path.join(save_root_2d, 'occl_comp', 'reflectance')
    if not os.path.exists(path_reflectance):
        os.makedirs(path_reflectance) 
    with open(os.path.join(path_reflectance, '{}.dat'.format(num_id)), 'wb') as f:
        pickle.dump(img_reflectance, f)    
        
    path_depth_img = os.path.join(save_root_2d, 'occl_comp', 'depth_image')
    if not os.path.exists(path_depth_img):
        os.makedirs(path_depth_img)    
    path_density_img = os.path.join(save_root_2d, 'occl_comp', 'density_image')
    if not os.path.exists(path_density_img):
        os.makedirs(path_density_img) 
    path_geometry_image = os.path.join(save_root_2d, 'occl_comp', 'geometry_image')
    if not os.path.exists(path_geometry_image):
        os.makedirs(path_geometry_image)    
    io.imsave(os.path.join(path_depth_img, '{}.png'.format(num_id)), img_2d_save2)
    io.imsave(os.path.join(path_density_img, '{}.png'.format(num_id)), density_img_save2)
    io.imsave(os.path.join(path_geometry_image, '{}.png'.format(num_id)), fill_flag)
    '''
    print("done...{}".format(num_id))


'''
for idx, nid in enumerate(ids):
    print('{} / {} ...'.format(idx, len(ids)-1))
    query_2d(nid)
'''
'''
train_ids = [x for x in df['id'] if x not in ids]
random.seed(0)
part_train_ids = random.sample(train_ids, 400)
part_train_ids.sort()
'''

for idx, nid in enumerate(test_facades_nids):
    print('{} / {} ...'.format(idx, len(test_facades_nids)))
    query_2d(nid) 
    #if idx > 2:
    #    break
    
line_id = np.arange(0, len(test_facades_nids))
new_df = pd.DataFrame(line_id,columns=['id'])
new_df['nid'] = np.array(test_facades_nids)
gdf = geopandas.GeoDataFrame(new_df, geometry=facade_lines)
gdf.to_file('../tmp/facade_lines_{}.geojson'.format(MODE), driver='GeoJSON')

