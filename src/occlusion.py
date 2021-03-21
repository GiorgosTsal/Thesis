# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 20:29:18 2021

@author: root
"""

# %matplotlib qt
import random as rand
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import os
import shutil
import open3d as o3d

# %% clean forlder
def deleteFilesRecurs():
    # delete OFF files
    root_dir = 'mydataset/'
    for root, dirs, files in os.walk(root_dir):
        for name in files:
                os.remove(os.path.join(root, name)) #change to os.remove once sure
    print("Old files removed...")
    
deleteFilesRecurs()
# %% generate point cloud - set gloabal vars

#number of points to Generate
T = 400


sigma = 8
alpha = 4

# # BIG domain of x and y
# XdomainStart = -1000
# XdomainEnd = 1000
# YdomainStart = -1000
# YdomainEnd = 1000

# #test domain of x and y as geogebra
# XdomainStart = -12 
# XdomainEnd = 12
# YdomainStart = -12
# YdomainEnd = 12


# #domain of x and y as petridis gave me
# XdomainStart = -1
# XdomainEnd = 1
# YdomainStart = 1
# YdomainEnd = 3

# %% OCCLUSION ON X AXIS
# XdomainStart = -1
# XdomainEnd = -0.5
# YdomainStart = 1
# YdomainEnd = 3

# x = []
# y = []

# #populate x and y with random numbers into and range/domain
# for i in range(0,T):
#       y.append(rand.uniform(YdomainStart,YdomainEnd))
      
# #populate x and y with random numbers into and range/domain
# for i in range(0,int(T/2)):
#       x.append(rand.uniform(XdomainStart, XdomainEnd))
      
     

# # new cut domain of x and y
# XdomainStart = 0.5
# XdomainEnd = 1


# #populate x and y with random numbers into and range/domain
# for i in range(0,int(T/2)):
#       x.append(rand.uniform(XdomainStart, XdomainEnd))

 # %%OCCLUSION ON Y AXIS
# XdomainStart = -1
# XdomainEnd = 1
# YdomainStart = 1
# YdomainEnd = 1.5

# x = []
# y = []

# #populate x and y with random numbers into and range/domain
# for i in range(0,T):
#       x.append(rand.uniform(XdomainStart, XdomainEnd))
      
# #populate x and y with random numbers into and range/domain
# for i in range(0,int(T/2)):
#       y.append(rand.uniform(YdomainStart,YdomainEnd))
      
     

# # new cut domain of x and y
# YdomainStart = 2.5
# YdomainEnd = 3


# #populate x and y with random numbers into and range/domain
# for i in range(0,int(T/2)):
#       y.append(rand.uniform(YdomainStart,YdomainEnd))

 # %%OCCLUSION ON BOTH AXIS -case 1
# XdomainStart = -1
# XdomainEnd = -0.5
# YdomainStart = 1
# YdomainEnd = 1.5

# x = []
# y = []

      
# #populate x and y with random numbers into and range/domain
# for i in range(0,int(T/2)):
#       y.append(rand.uniform(YdomainStart,YdomainEnd))
#       x.append(rand.uniform(XdomainStart, XdomainEnd))
      
     

# # new cut domain of x and y
# YdomainStart = 2.5
# YdomainEnd = 3
# XdomainStart = 0.5
# XdomainEnd = 1


# #populate x and y with random numbers into and range/domain
# for i in range(0,int(T/2)):
#       y.append(rand.uniform(YdomainStart,YdomainEnd))
#       x.append(rand.uniform(XdomainStart, XdomainEnd))
  
# %%OCCLUSION ON BOTH AXIS -case 2
XdomainStart = -1
XdomainEnd = -0.5
YdomainStart = 2
YdomainEnd = 3

x = []
y = []


#populate x and y with random numbers into and range/domain
for i in range(0,T):
      y.append(rand.uniform(YdomainStart,YdomainEnd))
      
#populate x and y with random numbers into and range/domain
for i in range(0,int(T/2)):
      x.append(rand.uniform(XdomainStart, XdomainEnd))
      
     
XdomainStart = 0.5
XdomainEnd = 1


#populate x and y with random numbers into and range/domain
for i in range(0,int(T/2)):
      x.append(rand.uniform(XdomainStart, XdomainEnd))
# %%
#transorm lists to numpy arrays for ease
X=np.array(x)
Y=np.array(y)

#Function A: z=2+sinx+cosy
funA = 2 + np.sin(X) + np.cos(Y)

#Function B: z=a*exp(-(x2+y2)/σ)      σ=8,   a=4
funB = alpha*np.exp(-(pow(X,2)+pow(Y,2))/sigma)

#Function C: z=x2+0.25*y2
funC = pow(X,2) + 0.25*pow(Y,2)

# round for 5 decimal format
# X = [np.round(x, 5) for x in X]
# Y = [np.round(y, 5) for y in Y]
# funA = [np.round(a, 5) for a in funA]
# funB = [np.round(b, 5) for b in funB]
# funC = [np.round(c, 5) for c in funC]

X = np.round(X, 5)
Y = np.round(Y, 5)
funA = np.round(funA, 5)
funB = np.round(funB, 5)
funC = np.round(funC, 5)

# %% Plot Point Cloud 

#Plot Function A: z=2+sinx+cosy
fig = plt.figure()
fig.suptitle('Function A: z=2+sinx+cosy')
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X,Y,funA)
plt.show()

#Plot Function B: z=a*exp(-(x2+y2)/σ)      σ=8,   a=4
fig = plt.figure()
fig.suptitle('Function B: z=a*exp(-(x2+y2)/σ)   σ=8, a=4')
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X,Y,funB)
plt.show()

#Plot Function C : z=x2+0.25*y2
fig = plt.figure()
fig.suptitle('Function C : z=x2+0.25*y2')
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X,Y,funC)
plt.show()
# %% Rotation region


v=np.column_stack((X,Y))

vec_funA = np.column_stack((v,funA))
vec_funB = np.column_stack((v,funB))
vec_funC = np.column_stack((v,funC))




def rotateVec(vector, degrees, axis, x_rot=0, y_rot=0, z_rot=0):  
    if axis == 'x':
        rotation_axis = np.array([1, 0, 0])
    elif axis=='y':
        rotation_axis = np.array([0, 1, 0])
    elif axis=='z':
        rotation_axis = np.array([0, 0, 1])
    elif axis=='all':
        rotation_axis = np.array([1, 1, 1]) 
        
    rotation_degrees = degrees
    rotation_radians = np.radians(rotation_degrees)    
    rotation_vector = rotation_radians * rotation_axis
    rotation = R.from_rotvec(rotation_vector)
    rotated_vec = rotation.apply(vector)
    return rotated_vec

def rotateVecMultiple(vector, x_rot=0, y_rot=0, z_rot=0):  
    #x rotation  
    rotation_axis_x = np.array([1, 0, 0])     
    rotation_degrees = x_rot
    rotation_radians = np.radians(rotation_degrees)    
    rotation_vector = rotation_radians * rotation_axis_x
    rotation = R.from_rotvec(rotation_vector)
    rotated_vec = rotation.apply(vector)
    #y rotation
    rotation_axis_y = np.array([0, 1, 0])
    rotation_degrees = y_rot
    rotation_radians = np.radians(rotation_degrees)    
    rotation_vector = rotation_radians * rotation_axis_y
    rotation = R.from_rotvec(rotation_vector)
    rotated_vec = rotation.apply(rotated_vec)
    #z 
    rotation_axis_z = np.array([0, 0, 1])
    rotation_degrees = z_rot
    rotation_radians = np.radians(rotation_degrees)    
    rotation_vector = rotation_radians * rotation_axis_z
    rotation = R.from_rotvec(rotation_vector)
    rotated_vec = rotation.apply(rotated_vec)
    return rotated_vec

def deleteFilesRecurs():
    # delete OFF files
    root_dir = 'mydataset/'
    for root, dirs, files in os.walk(root_dir):
        for name in files:
                os.remove(os.path.join(root, name)) #change to os.remove once sure
    print("All files removed...")
                
def splitFilesToFolders(percentage, count): 
    forrmat = '.ply'
    stop =int(percentage * count)
    samplelist = rand.sample(range(0, count), stop)    
    for n in samplelist:
        #split functionA folders
        src_a = 'mydataset/functionA/train/functionA_'+ str(n) + forrmat
        dst_a = 'mydataset/functionA/test/functionA_'+ str(n) + forrmat
        shutil.move(src_a, dst_a)
        #split functionB folders
        src_b = 'mydataset/functionB/train/functionB_'+ str(n) + forrmat
        dst_b = 'mydataset/functionB/test/functionB_'+ str(n) + forrmat
        shutil.move(src_b, dst_b)
        #split functionA folders
        src_c = 'mydataset/functionC/train/functionC_'+ str(n) + forrmat
        dst_c = 'mydataset/functionC/test/functionC_'+ str(n) + forrmat
        shutil.move(src_c, dst_c)  
    print('Files splited...')
            
# %% Create .off files from vector with rotation 

# 3 variables (x,y,z)
# 8 values of degrees
# Total points = 8 ^3 = 512, so 512 files per function
degrees = []
for x in range(0, 360, 45):
    degrees.append(x)

# print(degreez)
# # [0, 45, 90, 135, 180, 225, 270, 315]
print('Creation of dataset begins...')
print('Starting to rotate and write files...')
count = 0 
frmt = '.ply'
for x_rot in degrees:
    for y_rot in degrees:
        for z_rot in degrees:
            rotated_vec_funA = rotateVecMultiple(vec_funA,x_rot, y_rot, z_rot)
            rotated_vec_funB = rotateVecMultiple(vec_funB,x_rot, y_rot, z_rot)
            rotated_vec_funC = rotateVecMultiple(vec_funC,x_rot, y_rot, z_rot)
           
            pcdA = o3d.geometry.PointCloud()
            pcdA.points = o3d.utility.Vector3dVector(rotated_vec_funA)
            o3d.io.write_point_cloud('mydataset/functionA/train/functionA_'+ str(count) + frmt, pcdA, True)
            
            pcdB = o3d.geometry.PointCloud()
            pcdB.points = o3d.utility.Vector3dVector(rotated_vec_funB)
            o3d.io.write_point_cloud('mydataset/functionB/train/functionB_'+ str(count) + frmt, pcdB, True)
            
            pcdC = o3d.geometry.PointCloud()
            pcdC.points = o3d.utility.Vector3dVector(rotated_vec_funC)
            o3d.io.write_point_cloud('mydataset/functionC/train/functionC_'+ str(count) + frmt, pcdC, True)
            

            count +=1

print('All files created succesfully...')

    
# %% Split into train-test ===> Copy 30% of the files as test
percentage = 0.3
splitFilesToFolders(percentage, count)