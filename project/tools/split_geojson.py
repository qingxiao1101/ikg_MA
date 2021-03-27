# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
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






df = geopandas.read_file(open('../tmp/final_merge_tmp.geojson'))
aim = df[df['dis']>4.0]
aim.to_file('../tmp/merge_remove_less4.0', driver='GeoJSON')

print(aim.shape)

