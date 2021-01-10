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
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X,Y,funA)
plt.show()

#Plot Function A: z=2+sinx+cosy
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X,Y,funB)
plt.show()

#Plot Function C : z=x2+0.25*y2
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X,Y,funC)
plt.show()