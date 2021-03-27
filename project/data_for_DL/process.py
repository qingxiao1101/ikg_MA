# -*- coding: utf-8 -*-
import os 
from shapely.geometry import LineString
import pandas as pd
import geopandas as gpd

MODE = 'test'
#df = pd.DataFrame(columns=['nid'])
df = pd.DataFrame(columns=['nid'])

lines = list()
with open(os.path.join('./','tmp_facade_line_{}.txt'.format(MODE)),'r') as file_handle:
   for i, a in enumerate(file_handle.readlines()):
       coor = a.split('(')[-1].split(')')[0]
       coor = coor.replace(',', ' ')
       coor = [x for x in coor.split(' ') if x !='']
       coor = list(map(float, coor))       
       facade_line = LineString([(coor[0], coor[1]), (coor[2], coor[3])])
       lines.append(facade_line)
       nid = a.split(' ')[0]
       df.loc[i] = [nid]

print(df.shape)       
gdf = gpd.GeoDataFrame(df, geometry=lines)
gdf.to_file('../tmp/facade_lines_{}_v2.geojson'.format(MODE), driver='GeoJSON')

       
