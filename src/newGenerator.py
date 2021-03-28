# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 17:57:13 2021

@author: gtsal
"""

# %matplotlib qt
import random as rand
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import os
import open3d as o3d

# with this, we generate in each execution the same data
np.random.seed(2021)
rand.seed = 2021
plots = True

def deleteFilesRecurs():
    # delete OFF files
    root_dir = 'mydataset/'
    for root, dirs, files in os.walk(root_dir):
        for name in files:
                os.remove(os.path.join(root, name)) #change to os.remove once sure
    print("Old files removed...")
    
deleteFilesRecurs()
# %% generate point cloud

#number of points to Generate
T = 400


sigma = 8
alpha = 4
# %% Function A: z=2+sinx+cosy

# new cut domain of x and y
XadomainStart = -1
XadomainEnd = 3
YadomainStart = -1
YadomainEnd = 3

xa = []
ya = []
#populate x and y with random numbers into and range/domain
for i in range(0,T):
      xa.append(rand.uniform(XadomainStart, XadomainEnd))
      ya.append(rand.uniform(YadomainStart,YadomainEnd))

#transorm lists to numpy arrays for ease
Xa=np.array(xa)
Ya=np.array(ya)


funA = 2 + np.sin(Xa) + np.cos(Ya)


Xa = np.round(Xa, 5)
Ya = np.round(Ya, 5)
funA = np.round(funA, 5)
# %% Function B: z=a*exp(-(x2+y2)/σ)      σ=8,   a=4

# new cut domain of x and y
XbdomainStart = -2.4
XbdomainEnd = 1.6
YbdomainStart = -0.5
YbdomainEnd = 3

xb = []
yb = []
#populate x and y with random numbers into and range/domain
for i in range(0,T):
      xb.append(rand.uniform(XbdomainStart, XbdomainEnd))
      yb.append(rand.uniform(YbdomainStart,YbdomainEnd))

#transorm lists to numpy arrays for ease
Xb=np.array(xb)
Yb=np.array(yb)


funB = alpha*np.exp(-(pow(Xb,2)+pow(Yb,2))/sigma)

Xb = np.round(Xb, 5)
Yb = np.round(Yb, 5)
funB = np.round(funB, 5)

# %% Function C: z=x2+0.25*y2
# new cut domain of x and y
XcdomainStart = -0.5
XcdomainEnd = 3
YcdomainStart = -1.8
YcdomainEnd = 4.5

xc = []
yc = []
#populate x and y with random numbers into and range/domain
for i in range(0,T):
      xc.append(rand.uniform(XcdomainStart, XcdomainEnd))
      yc.append(rand.uniform(YcdomainStart,YcdomainEnd))

#transorm lists to numpy arrays for ease
Xc=np.array(xc)
Yc=np.array(yc)
funC = pow(Xc,2) + 0.25*pow(Yc,2)


Xc = np.round(Xc, 5)
Yc = np.round(Yc, 5)
funC = np.round(funC, 5)

# %% Plot Point Cloud 
if(plots):
    #Plot Function A: z=2+sinx+cosy
    fig = plt.figure()
    fig.suptitle('Function A: z=2+sinx+cosy')
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Xa,Ya,funA)
    plt.show()
    
    #Plot Function B: z=a*exp(-(x2+y2)/σ)      σ=8,   a=4
    fig = plt.figure()
    fig.suptitle('Function B: z=a*exp(-(x2+y2)/σ)   σ=8, a=4')
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Xb,Yb,funB)
    plt.show()
    
    #Plot Function C : z=x2+0.25*y2
    fig = plt.figure()
    fig.suptitle('Function C : z=x2+0.25*y2')
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Xc,Yc,funC)
    plt.show()
       
# %% Rotation region


va=np.column_stack((Xa,Ya))
vec_funA = np.column_stack((va,funA))

vb=np.column_stack((Xb,Yb))
vec_funB = np.column_stack((vb,funB))

vc=np.column_stack((Xc,Yc))
vec_funC = np.column_stack((vc,funC))



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

# %% Create noizy dataset 

# 0.15 0.22 0.28 0.35 0.42 0.5

pososto = 0
meanD = 0.7
mean = 0
std = pososto * meanD
nElements = 400

noise = np.random.normal(mean,std,nElements)

# print(noise)
# 3 variables (x,y,z)
# 8 values of degrees
# Total points = 8 ^3 = 512, so 512 files per function
degrees = []
for x in np.arange(22.5, 360, 45):
    degrees.append(x)

print(degrees)
# print(degrees)
# # [0, 45, 90, 135, 180, 225, 270, 315]
# [22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5]

np.random.seed(2021)
rand.seed = 2021

total = 0 
frmt = '.ply'
for x_rot in degrees:
    for y_rot in degrees:
        for z_rot in degrees:
            total+=1
            
stop =int(0.3 * total)   
samplelist = rand.sample(range(0, total), stop)    
# converting list to array
sample_arr = np.array(samplelist)    
  
print('Creation of dataset begins...')
count = 0 
frmt = '.ply'
for x_rot in degrees:
    for y_rot in degrees:
        for z_rot in degrees:
            
            for i in range(0,stop):
                if(sample_arr[i] == count):
                    flag = True
                    break
                else:
                    flag = False
                    
                    
                    
            if(flag): # If flag true add pcd to test folder with NOISE
                # FunctionA
                rotated_vec_funA = rotateVecMultiple(vec_funA,x_rot, y_rot, z_rot)
                rotated_vec_funA[:,0] = rotated_vec_funA[:,0] + noise
                rotated_vec_funA[:,1] = rotated_vec_funA[:,1] + noise
                rotated_vec_funA[:,2] = rotated_vec_funA[:,2] + noise
                
                pcdA = o3d.geometry.PointCloud()
                pcdA.points = o3d.utility.Vector3dVector(rotated_vec_funA)
                o3d.io.write_point_cloud('mydataset/functionA/test/functionA_'+ str(count) + frmt, pcdA, True)
                
                # FunctionB
                rotated_vec_funB = rotateVecMultiple(vec_funB,x_rot, y_rot, z_rot)
                rotated_vec_funB[:,0] = rotated_vec_funB[:,0] + noise
                rotated_vec_funB[:,1] = rotated_vec_funB[:,1] + noise
                rotated_vec_funB[:,2] = rotated_vec_funB[:,2] + noise
                
                pcdB = o3d.geometry.PointCloud()
                pcdB.points = o3d.utility.Vector3dVector(rotated_vec_funB)
                o3d.io.write_point_cloud('mydataset/functionB/test/functionB_'+ str(count) + frmt, pcdB, True)
                
                #FunctionC
                rotated_vec_funC = rotateVecMultiple(vec_funC,x_rot, y_rot, z_rot)
                rotated_vec_funC[:,0] = rotated_vec_funC[:,0] + noise
                rotated_vec_funC[:,1] = rotated_vec_funC[:,1] + noise
                rotated_vec_funC[:,2] = rotated_vec_funC[:,2] + noise   
                
                pcdC = o3d.geometry.PointCloud()
                pcdC.points = o3d.utility.Vector3dVector(rotated_vec_funC)
                o3d.io.write_point_cloud('mydataset/functionC/test/functionC_'+ str(count) + frmt, pcdC, True)                
            else: # without NOISE
                # FunctionA
                rotated_vec_funA = rotateVecMultiple(vec_funA,x_rot, y_rot, z_rot)
                pcdA = o3d.geometry.PointCloud()
                pcdA.points = o3d.utility.Vector3dVector(rotated_vec_funA)
                o3d.io.write_point_cloud('mydataset/functionA/train/functionA_'+ str(count) + frmt, pcdA, True)
                
                # FunctionB
                rotated_vec_funB = rotateVecMultiple(vec_funB,x_rot, y_rot, z_rot)                
                pcdB = o3d.geometry.PointCloud()
                pcdB.points = o3d.utility.Vector3dVector(rotated_vec_funB)
                o3d.io.write_point_cloud('mydataset/functionB/train/functionB_'+ str(count) + frmt, pcdB, True)
                
                #FunctionC
                rotated_vec_funC = rotateVecMultiple(vec_funC,x_rot, y_rot, z_rot)
                pcdC = o3d.geometry.PointCloud()
                pcdC.points = o3d.utility.Vector3dVector(rotated_vec_funC)
                o3d.io.write_point_cloud('mydataset/functionC/train/functionC_'+ str(count) + frmt, pcdC, True)
            
            #Important
            count +=1

if(count == total):
    print("Dataset created.")

    
print("Program terminated succesfully.")
