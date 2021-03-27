# -*- coding: utf-8 -*-
import geopandas
import numpy as np 
from sklearn.neighbors import KDTree

similar_thres = 0.1 # distance of center point + rel_ground_h

df = geopandas.read_file(open('./tmp/total_windows.geojson'))
print("num all windows: ", len(df))

df = df.reset_index(drop=True)

coors = [np.array(x).reshape(-1, ) for x in df['geometry']]
coors = np.stack(coors)
positions = np.stack([(coors[:, 2]+coors[:, 0])/2.0,  (coors[:, 3]+coors[:, 1])/2.0, np.array(df['rel_ground_h'])], axis=1)

scores = np.array(df['score'])
tree = KDTree(positions, leaf_size=2)
dists, inds = tree.query(positions, k=5)
removed_inds = list()
for itr in range(len(positions)):
    dist, ind = dists[itr], inds[itr]
    score = scores[ind]
    idx = np.where(dist<similar_thres, 1, 0).nonzero()[0]
    idx = [i for i in idx if i!=itr]
    score = score[idx]
    _idx = np.where(score<scores[itr], 1, 0).nonzero()[0]
    if len(_idx)>0:
        idx = np.array(idx).astype(np.int16)
        tmp_idx = idx[_idx]
        removed_idx = list(ind[tmp_idx])
        removed_inds += removed_idx

removed_inds = list(set(removed_inds))
df = df.drop(index=removed_inds)
df = df.reset_index(drop=True)

print("after remove: ", len(df))
df.to_file('./tmp/remove_duplicate_total_windows.geojson', driver='GeoJSON')

df = df[df['rel_ground_h']<2.0]
df.to_file('./tmp/less2m_windows.geojson', driver='GeoJSON')
