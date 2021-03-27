# -*- coding: utf-8 -*-

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

with open('./tmp/error_height_w2', 'rb') as f:
    w2 = pickle.load(f)
with open('./tmp/error_height_w5', 'rb') as f:
    w5 = pickle.load(f)
with open('./tmp/error_height_wd1', 'rb') as f:
    wd = pickle.load(f)
with open('./tmp/error_height_w7', 'rb') as f:
    w7 = pickle.load(f)

print(np.mean(np.abs(wd)))

raise SystemError

print(max(w2))
min_len = min(min(min(len(w2), len(w5)), len(wd)), len(w7))
data = {
'width=2': w2[:min_len],
'width=5': w5[:min_len],
'width=7': w7[:min_len],
'dynamic width': wd[:min_len]
}

df = pd.DataFrame(data)
df.plot.box(title="distributions of height errors with different Scan Line's width", xlabel='different Scan Line\'s width', ylabel='height error (m)')
plt.grid(linestyle="--", alpha=0.3)
plt.show()


with open('./tmp/error_height_fpn50_False', 'rb') as f:
    w1 = pickle.load(f)
with open('./tmp/error_height_fpn50_True', 'rb') as f:
    w2 = pickle.load(f)
with open('./tmp/error_height_res50_False', 'rb') as f:
    w3 = pickle.load(f)
with open('./tmp/error_height_res50_True', 'rb') as f:
    w4 = pickle.load(f)

max_len = max(max(max(len(w1), len(w2)), len(w3)), len(w4))

data = {
'fpn50': list(w1)+[0]*(max_len-len(w1)),
'fpn50+post': list(w2)+[0]*(max_len-len(w2)),
'res50': list(w3)+[0]*(max_len-len(w3)),
'res50+post': list(w4)+[0]*(max_len-len(w4))
}

df = pd.DataFrame(data)
df.plot.box(title="distributions of height errors with different backbones", ylabel='height error (m)', showmeans=False)
plt.grid(linestyle="--", alpha=0.3)
plt.ylim(-0.5, 0.5)
plt.show()
