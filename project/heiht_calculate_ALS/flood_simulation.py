# -*- coding: utf-8 -*-
import pandas as pd
import os 
from sklearn.neighbors import KDTree
import numpy as np
import geopandas

'''
max water depth: 
    1.66142786984
    1.178758033309
    1.4942301973449998
    0.8043575130559001
    1.554369782226
    1.6838579311739998
    1.4046720208030001
    1.70271838826
    1.635200073343
    0.2867653124033
    1.6014694911660001
    0.2853841409056
'''

CSV_FILE_PATH = './tmp/ResultsForIKGWebviewer/'
risk_path = './tmp/risk_region/'
if not os.path.exists(risk_path):
   os.makedirs(risk_path)  
   
window_path = './tmp/according_times_geojson/'
if not os.path.exists(window_path):
   os.makedirs(window_path)  


csv_names = os.listdir(CSV_FILE_PATH)
csv_names.remove('Readme.txt')
window_lines_o = geopandas.read_file(open('./tmp/less2m_windows.geojson'))

#print(csv_names)
#df = pd.read_csv(CSV_FILE_PATH)
for names in csv_names:
    print(names)
    window_lines = window_lines_o.copy()
    window_lines['water_depth'] = -1.0
    flood_df = pd.read_csv(os.path.join(CSV_FILE_PATH, names))
    search_space = np.array(list(zip(flood_df['Points:0'].tolist(), flood_df['Points:1'].tolist())))
    values = np.array(flood_df['WDEPTH'].tolist())
    tree = KDTree(search_space, leaf_size=2)
    for i in range(len(window_lines)):
        line = np.array(window_lines.iloc[i]['geometry']).reshape(-1, )
        pos_line = np.array([[(line[0]+line[2])/2.0, (line[1]+line[3])/2.0]])
        dists, inds = tree.query(pos_line, k=1)
        
        water_depth = values[inds].mean(axis=1)
        if dists.mean() > 5.0:
            window_lines.loc[i,'water_depth' ] = -1.5
        else:
            window_lines.loc[i,'water_depth' ] = water_depth

    save_name = names.split('_')[0]
    window_lines.to_file(os.path.join(window_path, 'window_{}.geojson'.format(save_name)), driver='GeoJSON')
    risk_df = window_lines[window_lines['rel_ground_h']<=window_lines['water_depth']]
    risk_df.to_file(os.path.join(risk_path, 'risk_{}.geojson'.format(save_name)), driver='GeoJSON')
        





