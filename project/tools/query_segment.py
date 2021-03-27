# -*- coding: utf-8 -*-
import os
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

root = '/Volumes/Qing Xiao/ikg/6_tmp/intermediate_splitover30/'
save_root = os.path.join('/Volumes/Qing Xiao/ikg/6_tmp/', 'geojson_splitover30_total_overlap')

#parse_root = '/Volumes/Qing Xiao/ikg/parse_dumpfiles/'
#root = '/Volumes/Qing Xiao/ikg/tmp_dumpfiles/'
#save_root = '/Volumes/Qing Xiao/ikg/tmp_segments/'

with open(os.path.join(save_root, 'merge_post.dat'), 'rb') as f:
    match = pickle.load(f)

#df = geopandas.read_file(open('../tmp/merge_tmp_v2.geojson'))

g_df = pd.DataFrame()
query_id = 9
for name, nid in match[query_id]:
    df = geopandas.read_file(os.path.join(root, name, 'segment_line_building.geojson'))
    g_df = g_df.append(df[df['id']==nid])
    
g_df.to_file(os.path.join(save_root, "{}_segment_line.geojson".format(query_id)), driver='GeoJSON')

print("done!")