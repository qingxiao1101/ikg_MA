# -*- coding: utf-8 -*-

import numba as nb
import numpy as np
from tqdm import tqdm
from numba import jit
import warnings
import time

warnings.filterwarnings('ignore')

class Point(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y

@jit
def quick_sort(array, left, right):
    if left >= right:
        return
    low = left
    high = right
    key = array[low]
    while left < right:
        while left < right and array[right] > key:
            right -= 1
        array[left] = array[right]
        while left < right and array[left] <= key:
            left += 1
        array[right] = array[left]
    array[right] = key
    quick_sort(array, low, left - 1)
    quick_sort(array, left + 1, high)


@jit
def median(lis):
    quick_sort(lis, 0, len(lis)-1)
    return lis[(len(lis)-1)//2]
    




#@jit(parallel=True,nogil=True)
@jit
def median_filter(img, rows, columns):
    three_channel = False
    if img.shape[-1]==3:
        three_channel = True
    tmp = np.zeros((rows, columns, 3)) if three_channel else np.zeros((rows, columns))
    connect = [(-1,-1), (0, -1),(1, -1),(1, 0), (0, 1),(-1, 1), (-1, 0)]
    #connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), \
    #                    Point(0, 1), Point(-1, 1), Point(-1, 0)]
    
    for x in range(1, rows-1):
        for y in range(1, columns-1):
           
           r = np.zeros((len(connect),))
           if three_channel:
               g = np.zeros((len(connect),))
               b = np.zeros((len(connect),))
           
           for i in range(len(connect)):
               if three_channel:
                   r[i] = img[x+connect[i][0], y+connect[i][1], 0]
                   g[i] = img[x+connect[i][0], y+connect[i][1], 1]
                   b[i] = img[x+connect[i][0], y+connect[i][1], 2]
               else:
                  r[i] = img[x+connect[i][0], y+connect[i][1]]
           if three_channel:
               tmp[x,y,0] = median(r)
               tmp[x,y,1] = median(g)
               tmp[x,y,2] = median(b)
           else:
               tmp[x,y] = median(r)
    return tmp
