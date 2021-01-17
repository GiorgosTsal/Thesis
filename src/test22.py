# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 20:55:33 2021

@author: root
"""
import numpy as np
import open3d as o3d
import os

# Pass numpy array to Open3D.o3d.geometry.PointCloud and visualize
xyz = np.random.rand(100, 3)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
o3d.io.write_point_cloud("./data.ply", pcd, True)

o3d.visualization.draw_geometries([pcd])