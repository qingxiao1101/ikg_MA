# -*- coding: utf-8 -*-
import numpy as np
import struct
import pickle
from collections import Counter    
from skimage import io
import cv2
import matplotlib.pyplot as plt
#from tqdm import tqdm
from region_growing import RegionGrowing
import itertools
import time
from numba import jit
import math
from PIL import Image, ImageDraw, ImageFont
import config
import os
import csv
import utils
import os

import geopandas
from shapely import geometry
from shapely.geometry import LineString
import pandas as pd

import warnings

warnings.filterwarnings('ignore')



def load_file(name, path):
    with open(os.path.join(path, "{}.dat".format(name)), 'rb') as f:
        tmp = pickle.load(f)
    return tmp


def save_file(data, path):
    if os.path.exists(path):
        os.remove(path)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    
    

def extract_scene(path):
    '''
    extract the objects which perpendicular to ground
    '''
    norm = load_file('normal', path)
    
    norm_x = norm[:,:,0]
    norm_y = norm[:,:,1]
    norm_xy = np.sqrt(norm_x**2+norm_y**2)
    
    norm_xy = np.expand_dims(norm_xy, 2).repeat(3, axis=2)
    building = np.where(norm_xy>0.95, norm, 0)
    return building

def generate_depth_image(path):
    coor = load_file('coordinate', path)
    head = load_file('head', path)

    tmp = coor - head
    tmp = tmp**2
    tmp = tmp.sum(axis=2)
    dis = np.sqrt(tmp)
    return dis




def slope(p1, p2):
    return (p1[1]-p2[1])/(p1[0]-p2[0])    


def region_merge(seg, bounding, search_range, split_long_seg=False):
    rows, columns = seg.shape[0:2]
    #search_range = config.search_range
    search_mask = set()
    #print("region merging ...")
    for idx_r in range(rows):
        print("\rregion merging: {:.2f}%".format(100 * idx_r / float(rows)), end='')

        for idx_c in range(columns):
            cur_id = seg[idx_r, idx_c]
            if cur_id == -1 or cur_id in search_mask:
                continue
            cur_norm, cur_bounding, cur_bounding_3d, cur_depth, cur_ref = bounding[cur_id]
            
            y_max = cur_bounding['y_max'].y
            y_min = cur_bounding['y_min'].y
            x_max = cur_bounding['x_max'].x
            x_min = cur_bounding['x_min'].x
            range_seg = seg[max(0, x_min-search_range):min(rows, x_max+search_range),
                            max(0, y_min-search_range):min(columns, y_max+search_range)]
            
            new_seg_id = set(range_seg.reshape(-1))
            for nid in new_seg_id:
                    if nid==-1 or nid==cur_id:
                        continue
                    v1 = cur_norm
                    v2 = bounding[nid][0]
                    
                    similarity = np.dot(v1,v2)/(np.linalg.norm(v1)*(np.linalg.norm(v2)))

                    if similarity > config.threshold_similarity:
                        
                        
                        n_3d = bounding[nid][2]
                        
                        d = abs(cur_depth-bounding[nid][3])    
                        d_lines = utils.l2_distance_lines(cur_bounding_3d['y_min'], cur_bounding_3d['y_max'],
                                              n_3d['y_min'], n_3d['y_max'])
                        
                        ymin_coor,ymax_coor, cover_rate, _ = utils.line2line_project(bounding[cur_id][2]['y_min'][:2], 
                                                                           bounding[cur_id][2]['y_max'][:2],
                                                                           bounding[nid][2]['y_min'][:2], 
                                                                           bounding[nid][2]['y_max'][:2])

                        if not split_long_seg:
                            if (cover_rate < 0.7 and d_lines < 2.5 )   or \
                                (cover_rate > 0.7 and cover_rate < 0.95 and d < config.threshold_dis*(1+0.1*search_range/float(config.search_range)) and abs(cur_ref-bounding[nid][4])<0.1 and d_lines < 0.5):
                                if bounding[nid][1]['x_max'].x > bounding[cur_id][1]['x_max'].x:
                                    bounding[cur_id][1]['x_max'] = bounding[nid][1]['x_max']
                                    bounding[cur_id][2]['x_max'] = bounding[nid][2]['x_max']
                                if bounding[nid][1]['y_max'].y > bounding[cur_id][1]['y_max'].y:
                                    bounding[cur_id][1]['y_max'] = bounding[nid][1]['y_max']
                                    #bounding[cur_id][2]['y_max'] = bounding[nid][2]['y_max']
                                if bounding[nid][1]['x_min'].x < bounding[cur_id][1]['x_min'].x:
                                    bounding[cur_id][1]['x_min'] = bounding[nid][1]['x_min']
                                    bounding[cur_id][2]['x_min'] = bounding[nid][2]['x_min']
                                if bounding[nid][1]['y_min'].y <  bounding[cur_id][1]['y_min'].y:
                                    bounding[cur_id][1]['y_min'] = bounding[nid][1]['y_min']
                                    #bounding[cur_id][2]['y_min'] = bounding[nid][2]['y_min']
                                bounding[cur_id][2]['y_min'] = ymin_coor
                                bounding[cur_id][2]['y_max'] = ymax_coor
                                seg = np.where(seg==nid, cur_id, seg)
                        else:
                            if (cover_rate < 0.7 and d_lines < 2.5 ) :  
                                if bounding[nid][1]['x_max'].x > bounding[cur_id][1]['x_max'].x:
                                    bounding[cur_id][1]['x_max'] = bounding[nid][1]['x_max']
                                    bounding[cur_id][2]['x_max'] = bounding[nid][2]['x_max']
                                if bounding[nid][1]['y_max'].y > bounding[cur_id][1]['y_max'].y:
                                    bounding[cur_id][1]['y_max'] = bounding[nid][1]['y_max']
                                    #bounding[cur_id][2]['y_max'] = bounding[nid][2]['y_max']
                                if bounding[nid][1]['x_min'].x < bounding[cur_id][1]['x_min'].x:
                                    bounding[cur_id][1]['x_min'] = bounding[nid][1]['x_min']
                                    bounding[cur_id][2]['x_min'] = bounding[nid][2]['x_min']
                                if bounding[nid][1]['y_min'].y <  bounding[cur_id][1]['y_min'].y:
                                    bounding[cur_id][1]['y_min'] = bounding[nid][1]['y_min']
                                    #bounding[cur_id][2]['y_min'] = bounding[nid][2]['y_min']
                                bounding[cur_id][2]['y_min'] = ymin_coor
                                bounding[cur_id][2]['y_max'] = ymax_coor
                                seg = np.where(seg==nid, cur_id, seg)

            search_mask.add(cur_id)
   
    return seg, bounding




def split_building_fence(path, save_path):
    '''
    separate the facades and fences... 
    '''
    seg, bounding = load_file("tmp_regiongmerge", save_path)
    coor = load_file("coordinate", path)
    rows, columns = seg.shape
    bounding_fence = dict()
    bounding_building = dict()
    seg_fence = -np.ones(seg.shape)
    seg_building = -np.ones(seg.shape)
    fence_id = 0
    building_id = 0
    print("\rsplit building fence...", end='')

    for nid in set(seg.reshape(-1)):
        if nid == -1:
            continue
        x_min3d = bounding[nid][2]['x_min']
        y_min3d = bounding[nid][2]['y_min']
        x_max3d = bounding[nid][2]['x_max']
        y_max3d = bounding[nid][2]['y_max']
        ref_z = np.mean(coor[rows//2, bounding[nid][1]['y_min'].y:bounding[nid][1]['y_max'].y,2])
        length = math.sqrt((y_max3d[0]-y_min3d[0])**2 + (y_max3d[1]-y_min3d[1])**2)
        if max(x_min3d[2], x_max3d[2]) - ref_z >config.split_height and \
            length < config.block_width: 
            # 块宽度小于2米，过滤
            continue
        if max(x_min3d[2], x_max3d[2]) - ref_z >config.split_height \
            and abs(x_min3d[2]-x_max3d[2]) < config.block_height:
            # 高于1.5米，不是矮墙，然后块高度差小于2米，过滤
            continue
        if min(x_min3d[2], x_max3d[2]) - ref_z > config.threshold_height:
            continue
        if max(x_min3d[2], x_max3d[2]) - ref_z < config.split_height or length > 180:
            #最大高度小于1.5米，设置为矮墙
            seg_fence = np.where(seg==nid, fence_id, seg_fence)
            bounding_fence[fence_id] = bounding[nid]
            fence_id += 1
        elif max(x_min3d[2], x_max3d[2]) - ref_z > config.split_height and \
            length > config.block_width:
            seg_building = np.where(seg==nid, building_id, seg_building)
            bounding_building[building_id] = bounding[nid]
            building_id += 1
    print("    done!")        
    save_file((seg_fence, bounding_fence), os.path.join(save_path, 'tmp_fence.dat'))
    save_file((seg_building, bounding_building), os.path.join(save_path, 'tmp_building.dat'))
    utils.add_boundbox(save_path, seg_building, bounding_building, flag='building')
    #io.imsave(os.path.join(save_path, 'tmp_building.png'), utils.random_render(seg_building))
    io.imsave(os.path.join(save_path, 'tmp_fence.png'), utils.random_render(seg_fence))
    generate_geojson(path, save_path, bounding_building, name="building")
    print("    done!")    
    return seg_fence, seg_building, bounding_fence, bounding_building


def generate_csv(bounding, scanner="1", flag="mergebefore"):
    print('generate csv... ', end='')
    if os.path.exists("tmp/scanner_{}/segment_line_{}_{}.csv".format(scanner, scanner, flag)):
        os.remove("tmp/scanner_{}/segment_line_{}_{}.csv".format(scanner, scanner, flag))
        
    with open("tmp/scanner_{}/head_info.dat".format(scanner), 'rb') as f:
        data = pickle.load(f)
    o_x = data["original_x"]
    o_y = data["original_y"]
    count=0
    for nid, data in bounding.items():
        _, _, bounding_3d, _ , _= data
        info = "LINESTRING ({} {}, {} {})".format(bounding_3d['y_min'][0]+o_x,bounding_3d['y_min'][1]+o_y, 
                bounding_3d['y_max'][0]+o_x,bounding_3d['y_max'][1]+o_y)
        #info = [bounding_3d['y_min'][0]+o_x,bounding_3d['y_min'][1]+o_y, 
        #        bounding_3d['y_max'][0]+o_x,bounding_3d['y_max'][1]+o_y]
        with open("tmp/scanner_{}/segment_line_{}_{}.csv".format(scanner, scanner, flag),"a+") as csvfile: 
            writer = csv.writer(csvfile)
            writer.writerow([info])
        count += 1
    print("    done!")


def generate_geojson(path, save_path, bounding, name=None):
    print('generate geojson... ', end='')
    data = load_file('head_info', path)
    coor = load_file('coordinate', path)
    o_x = data["original_x"]
    o_y = data["original_y"]
    o_z = data["original_z"]
    lines = list()
    line_id = list()
    min_h = list()
    max_h = list()
    ref_h = list()
    dis = list()
    depth = list()
    for nid, data in bounding.items():
        _, _, bounding_3d, deep , _= data
        line = LineString([(bounding_3d['y_min'][0]+o_x,bounding_3d['y_min'][1]+o_y),
                           (bounding_3d['y_max'][0]+o_x,bounding_3d['y_max'][1]+o_y)])
        lines.append(line)
        line_id.append(nid)
        depth.append(deep)
        min_h.append(min(bounding_3d['x_min'][2], bounding_3d['x_max'][2])+o_z)
        max_h.append(max(bounding_3d['x_min'][2], bounding_3d['x_max'][2])+o_z)
        dis.append(utils.l2_distance_2d(bounding_3d['y_min'][0:2],bounding_3d['y_max'][0:2]))
        ref_h.append(np.mean(coor[coor.shape[0]//2, bounding[nid][1]['y_min'].y:bounding[nid][1]['y_max'].y,2])+o_z)
    df = pd.DataFrame(line_id,columns=['id'])
    df['min_h'] = min_h
    df['max_h'] = max_h
    df['ref_h'] = ref_h
    df['dis'] = dis
    df['depth'] = depth
    gdf = geopandas.GeoDataFrame(df, geometry=lines)
    if not gdf.empty:
        gdf.to_file(os.path.join(save_path, "segment_line_{}.geojson".format(name)), driver='GeoJSON')
    print("    done!")    
        
    



def run(path, save_path, split_long_seg=False):
    if  not os.path.exists(save_path):
        os.makedirs(save_path) 
    depth_image = generate_depth_image(path)
    coor = load_file("coordinate", path)
    reflectance = load_file("reflectance", path)
    
    reflectance = (reflectance-np.min(reflectance))/float(np.max(reflectance)-np.min(reflectance))
    
    scene = extract_scene(path)
    
    rows, columns = depth_image.shape
    ground = np.zeros((rows, columns, 3))         
    distance = np.expand_dims(depth_image, 2).repeat(3, axis=2) 
    building = np.where(distance > config.threshold_depth, scene, ground)
    region_growing = RegionGrowing(threshold=config.threshold_similarity, 
                                   if_4N=config.if_4N, num_filter=config.num_filter)
    seg, avg_norms, bounding = region_growing.run(building, coor, depth_image, reflectance, split_long_seg) # bounding (avg_norm, bounding_2d, bounding_3d)
    save_file((seg, bounding), os.path.join(save_path, 'tmp_regiongrowing.dat'))
    
    seg, bounding = region_merge(seg, bounding, search_range=config.search_range, split_long_seg=split_long_seg)
    
    save_file((seg, bounding), os.path.join(save_path, 'tmp_regiongmerge.dat'))
    io.imsave(os.path.join(save_path, 'tmp_regiongmerge.png'), utils.random_render(seg))
    generate_geojson(path, save_path, bounding, name="mergeed")

  
def run_once(path, save_path, file):
    path = os.path.join(path, file)
    save_path = os.path.join(save_path, file)
    run(path, save_path)
    split_building_fence(path, save_path)
    print("done!")


''' 
path = '/Volumes/Qing Xiao/ikg/parse_dumpfiles/'
save_path = '/Volumes/Qing Xiao/ikg/tmp_dumpfiles/'
run_once(path, save_path, '190906_094646_Scanner_2')
'''
#run_once(path, save_path, '190906_094925_Scanner_2')


'''
path = '/Volumes/Qing Xiao/ikg/parse_dumpfiles/190906_074826_Scanner_1'
save_path = '/Volumes/Qing Xiao/ikg/tmp_dumpfiles/190906_074826_Scanner_1'
 
seg, bounding = load_file('tmp_regiongrowing', '/Volumes/Qing Xiao/ikg/tmp_dumpfiles/190906_074826_Scanner_1')    
#utils.add_boundbox('../', seg, bounding, flag='190906_074826_Scanner_1')
seg, bounding = region_merge(seg, bounding, search_range=config.search_range, split_long_seg=False)
#utils.add_boundbox('../', seg, bounding, flag='190906_074826_Scanner_1')
#plt.imshow(utils.random_render(seg))
    
save_file((seg, bounding), os.path.join(save_path, 'tmp_regiongmerge.dat'))
io.imsave(os.path.join(save_path, 'tmp_regiongmerge.png'), utils.random_render(seg))
generate_geojson(path, save_path, bounding, name="mergeed")

split_building_fence(path, save_path)
'''













