# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 22:14:47 2021

@author: gtsal
"""

from pyntcloud import PyntCloud 
import numpy as np
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt 
import pandas as pd
import os
import sys
import pdb

pcd1 = PyntCloud.from_file("other/bunny.ply")


output_dir = "./exports/"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    

pcd_np = np.zeros((len(pcd1.points),6))



x = pcd1.points['x'].values 
y = pcd1.points['y'].values 
z = pcd1.points['z'].values 


v=np.column_stack((x,y))
vec = np.column_stack((v,z))

row, col = vec.shape

mylist = []

for i in range(0,row):
    # print(vec[i,0])
    if((vec[i,0] > 0.0442 and vec[i,0]<0.0623) and (vec[i,1] > 0.0387 and vec[i,1]<0.0757) and (vec[i,2] > -0.011 and vec[i,2]<0.035)):
        # print("1")
        mylist.append(vec[i,:])
        # print(vec[i,:])
        
# for i in range(0,row):
#     # print(vec[i,0])
#     if((vec[i,0] > min(x) and vec[i,0]<max(x)) and (vec[i,1] > min(y) and vec[i,1]<max(y)) and (vec[i,2] > min(z) and vec[i,2]<max(z))):
#         # print("1")
#         mylist.append(vec[i,:])
#         print(i)


newnp = np.array(mylist)

fig = plt.figure()
fig.suptitle('Function A: z=2+sinx+cosy')
ax = fig.add_subplot(111, projection='3d')
ax.scatter(newnp[:,0],newnp[:,1],newnp[:,2])
plt.show()

fig = plt.figure()
fig.suptitle('Function A: z=2+sinx+cosy')
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,z)
plt.show()

