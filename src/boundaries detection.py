# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 19:09:30 2021

@author: gtsal
"""
# conda install -c conda-forge pcl



import numpy as np
import pcl


p = pcl.PointCloud(10)  # "empty" point cloud
a = np.asarray(p)       # NumPy view on the cloud
a[:] = 0                # fill with zeros
print(p[3])             # prints (0.0, 0.0, 0.0)
a[:, 0] = 1             # set x coordinates to 1
print(p[3])             # prints (1.0, 0.0, 0.0)