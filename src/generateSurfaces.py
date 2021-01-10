# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 13:22:49 2021

@author: root
"""

import random as rand
import numpy as np
import matplotlib.pyplot as plt


#number of points to Generate
T = 10000

sigma = 8
alpha = 4

# # BIG domain of x and y
# XdomainStart = -1000
# XdomainEnd = 1000
# YdomainStart = -1000
# YdomainEnd = 1000

#test domain of x and y as geogebra
XdomainStart = -12 
XdomainEnd = 12
YdomainStart = -12
YdomainEnd = 12


# #domain of x and y as petridis gave me
# XdomainStart = -1
# XdomainEnd = 1
# YdomainStart = 1
# YdomainEnd = 3

x = []
y = []
#populate x and y with random numbers into and range/domain
for i in range(0,T):
      x.append(rand.uniform(XdomainStart, XdomainEnd))
      y.append(rand.uniform(YdomainStart,YdomainEnd))

#transorm lists to numpy arrays for ease
X=np.array(x)
Y=np.array(y)

#Function A: z=2+sinx+cosy
funA = 2 + np.sin(X) + np.cos(Y)

#Function B: z=a*exp(-(x2+y2)/σ)      σ=8,   a=4
funB = alpha*np.exp(-(pow(X,2)+pow(Y,2))/sigma)

#Function C: z=x2+0.25*y2
funC = pow(X,2) + 0.25*pow(Y,2)

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

# Rotation region
from scipy.spatial.transform import Rotation as R

v=np.column_stack((X,Y))
vec=np.column_stack((v,funB))

def rotateVec(vector, degrees, axis):  
    if axis == 'x':
        rotation_axis = np.array([1, 0, 0])
    elif axis=='y':
        rotation_axis = np.array([0, 1, 0])
    elif axis=='z':
        rotation_axis = np.array([0, 0, 1])      
    rotation_degrees = degrees
    rotation_radians = np.radians(rotation_degrees)    
    rotation_vector = rotation_radians * rotation_axis
    rotation = R.from_rotvec(rotation_vector)
    rotated_vec = rotation.apply(vector)
    return rotated_vec

degrees = 90
axis = 'x'
points = 20
#apply rotation 90 degrees on x axis
rotated_vec = rotateVec(vec, degrees, axis)

#Plot rotated
fig = plt.figure()
fig.suptitle('Rotated '+ str(degrees)+' degrees on ' + axis + ' axis: Function B : z=a*exp(-(x2+y2)/σ)   σ=8, a=4')
ax = fig.add_subplot(111, projection='3d')
ax.scatter(rotated_vec[:,0],rotated_vec[:,1],rotated_vec[:,2])
plt.show()


def translateVec(vector, points, axis):
    translated_vector=vector;
    if axis == 'x':
        translated_vector[:,0] = translated_vector[:,0] + points
    elif axis=='y':
        translated_vector[:,1] = translated_vector[:,1] + points
    elif axis=='z':
        translated_vector[:,2] = translated_vector[:,2] + points 
    return translated_vector

#apply translate for 20 points on x axis
translated_vec = translateVec(vec, points, axis)

vec=np.column_stack((v,funB))#i run this command for my help

#Plot translated
fig = plt.figure()
fig.suptitle('Translated for ' + str(points) + ' points on ' + axis +' axis: Function B : z=a*exp(-(x2+y2)/σ)   σ=8, a=4')
ax = fig.add_subplot(111, projection='3d')
ax.scatter(translated_vec[:,0],translated_vec[:,1],translated_vec[:,2])
plt.show()

    
    

