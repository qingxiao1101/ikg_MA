# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.font_manager import FontProperties
import csv
import os
import matplotlib.ticker as ticker

file_root = '/Users/xiaoqing/Desktop/tfboard-1'

'''读取csv文件'''
def readcsv(files):
    csvfile = open(files, 'r')
    plots = csv.reader(csvfile, delimiter=',')
    x = []
    y = []
    for row in plots:
        y.append((row[2])) 
        x.append(row[1])
    return x[1:] ,y[1:]

plt.figure()

'''
x,y1=readcsv(os.path.join(file_root, "precision.csv"))
precision = [round(float(i), 2)*100 for i in y1]
plt.plot(x, precision, '-*', color='red', label='precision')

_,y2=readcsv(os.path.join(file_root, "recall.csv"))
recall = [round(float(i), 2)*100 for i in y2]
plt.plot(x, recall, '-*', color='blue', label='recall')

_,y3=readcsv(os.path.join(file_root, "F1.csv"))
F1 = [round(float(i), 2)*100 for i in y3]
plt.plot(x, F1, '-*', color='green', label='F1')


plt.xlabel('Epochs',fontsize=12)
plt.ylabel('Score (%)',fontsize=12)
plt.title('evaluation curves',fontsize=12)
plt.legend(fontsize=12)


raise SystemError()
'''
x,y1=readcsv(os.path.join(file_root, "loss.csv"))
x = [int(i) for i in x]
loss = [round(float(i), 3) for i in y1]
plt.plot(x, loss, color='red', label='loss')


'''
_,y2=readcsv(os.path.join(file_root, "loss_rpn_cls.csv"))
loss_rpn_cls = [round(float(i), 3) for i in y2]
plt.plot(x, loss_rpn_cls, color='green', label='loss_rpn_cls')

_,y3=readcsv(os.path.join(file_root, "loss_rpn_box.csv"))
loss_rpn_box = [round(float(i), 3) for i in y3]
plt.plot(x, loss_rpn_box, color='yellow', label='loss_rpn_box')

_,y4=readcsv(os.path.join(file_root, "loss_rcnn_cls.csv"))
loss_rcnn_cls = [round(float(i), 3) for i in y4]
plt.plot(x, loss_rpn_box, color='blue', label='loss_rcnn_cls')

_,y5=readcsv(os.path.join(file_root, "loss_rcnn_box.csv"))
loss_rcnn_box = [round(float(i), 3) for i in y5]
plt.plot(x, loss_rpn_box, color='gray', label='loss_rcnn_box')
'''

plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(5000))

plt.xlabel('Iterations',fontsize=12)
plt.ylabel('Loss',fontsize=12)
plt.title('loss curves',fontsize=12)
plt.legend(fontsize=12)


