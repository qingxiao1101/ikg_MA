# -*- coding: utf-8 -*-

import os
from generate_segment  import run, split_building_fence




root = '/Volumes/Qing Xiao/ikg/1_parse/parse_dumpfiles_adjusted/'
if  not os.path.exists(root):
    raise InterruptedError("ssd not exist!!")

save_root = '/Volumes/Qing Xiao/ikg/6_tmp/tmp_dumpfiles_post0.75/'
if  not os.path.exists(save_root):
    os.makedirs(save_root)    
    
files = os.listdir(root)

for idx, file in enumerate(files):
    print("{}:  {} / {}".format(file, idx+1, len(files)))
    path_parse_dumpfile = os.path.join(root, file)
    
    save_path = os.path.join(save_root, file)
    
    #if os.path.exists(save_path):
    #    continue
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    run(path_parse_dumpfile, save_path, split_long_seg=True)
    split_building_fence(path_parse_dumpfile, save_path)

'''  
path_parse_dumpfile = os.path.join(root, '190906_092658_Scanner_2')
save_path = os.path.join(save_root, '190906_092658_Scanner_2')
run(path_parse_dumpfile, save_path, split_long_seg=True)
split_building_fence(path_parse_dumpfile, save_path)
'''
