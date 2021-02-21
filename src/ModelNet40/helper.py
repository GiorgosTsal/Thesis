# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 22:03:57 2021

@author: root
"""
import numpy as np
import pandas as pd
import open3d as o3d

file = "functionA_3.ply"
 # file2 = "functionA_9.off"
path = '../mydataset/functionA/test/' + file




def read_ply_file(filename):
    pcd = o3d.io.read_point_cloud(filename) # Read the point cloud
    # Convert open3d format to numpy array
    # Here, you have the point cloud in numpy format. 
    point_cloud_in_numpy = np.asarray(pcd.points) 
    
    return point_cloud_in_numpy 

points = read_ply_file(path)