# -*- coding: utf-8 -*-


threshold_depth = 6.0  # filter most of object cars
modality_reference_average = 'mode' # median , mode, mean
threshold_split_long_seg = 30.0
# for region growing
threshold_similarity = 0.95 # the similarity between two norms.
if_4N = False  # using 4 neighboors or 8 neighboors for region growing
num_filter = 100 # if the pixel of region less than this threshold, then remove it.

cover_rate = 0.1
# for region merging 
search_range = 20 
using_depth = False # if true, then using depth as threshold for merging, otherwise using space distance
threshold_dis = 1.5
inliers_thres = 0.8
max_iterations = 50
iou_thres = 0.5
scann_height = 5
# for split fence and building 
block_height = 2.0
block_width = 3.0
split_height = 3.0 
threshold_height = 5.0  

image_overlap = 100 # for split the long image


SCALE = 50
