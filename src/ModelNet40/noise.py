# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 22:01:34 2021

@author: root
"""

  
import numpy as np 
import os
import open3d as o3d
#import matplotlib.pyplot as plt


# %% Open plys as np arrays


def read_off(file):
    pcd = o3d.io.read_point_cloud(file) # Read the point cloud
    # Convert open3d format to numpy array
    # Here, you have the point cloud in numpy format. 
    point_cloud_in_numpy = np.asarray(pcd.points) 
    
    return point_cloud_in_numpy # return N*3 array 


input_path="mydataset/"
extra_path = "functionA/test/"
output_path="output/"
dataname = "functionA_2.ply"



myfile = input_path + extra_path + dataname

if not os.path.exists(output_path):
    os.makedirs(output_path)
    
# bunny_output_path = output_path+"bpa_mesh.ply"

pcd = o3d.io.read_point_cloud(myfile) # Read the point cloud
# Convert open3d format to numpy array
# Here, you have the point cloud in numpy format. 
point_cloud_in_numpy = np.asarray(pcd.points) 

print("1")
# o3d.visualization.draw_geometries([pcd])


# # %% Find Euclidean distance between points  

# # using sum() and square() 
# # intializing points in 
# # numpy arrays 
# point1 = np.array((1, 2, 3)) 
# point2 = np.array((1, 1, 1)) 
  
# # finding sum of squares 
# sum_sq = np.sum(np.square(point1 - point2)) 
  
# # Doing squareroot and 
# # printing Euclidean distance 
# print(np.sqrt(sum_sq)) 



# # %% create and add noise to vector

# # mean is the mean of the normal distribution you are choosing from
# # std is the standard deviation of the normal distribution
# # nElements is the number of elements you get in array noise
# # pososto is the percentage of noise we want to add multiplied by meanD
# # meanD is the mean euclidean distance between all points of the input vector

# pososto = 0.5
# meanD = 0.7
# mean = 0
# std = 1
# nElements = 1000


# noise = np.random.normal(mean,std,nElements)

# print(noise)
