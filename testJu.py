# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:33:44 2020

@author: coco1
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def vec_Ju(t):
    a = 5.202603 #au
    e = 0.048498
    i = np.deg2rad(-1.303)
    Om = np.deg2rad(-100.46)
    om = np.deg2rad(86.13)
    
    n = 0.01720209895/np.sqrt(a**3)
    M0 = np.deg2rad(20.02)
    M = M0 + n*t
    E = Newton(M,fE,fdE,e)
    X = a*(np.cos(E)-e)
    Y = a*np.sqrt(1-(e**2))*np.sin(E)
    A = np.array([X, Y, 0])
    x = np.matmul(R3(Om),np.matmul(R1(i),np.matmul(R3(om),A)))
    return(x,E,M)

def fE(E,e,M):
    return(E - e*np.sin(E) - M)

def fdE(E,e):
    return(1 - e*np.cos(E))

def R1(a):
    A = np.zeros([3,3])
    A[0,0] = 1
    A[1,1] = np.cos(a)
    A[2,2] = np.cos(a)
    A[2,1] = -np.sin(a)
    A[1,2] = np.sin(a)
    return(A)

def R3(a):
    A = np.zeros([3,3])
    A[2,2] = 1
    A[0,0] = np.cos(a)
    A[1,1] = np.cos(a)
    A[1,0] = -np.sin(a)
    A[0,1] = np.sin(a)
    return(A)

def Newton(M,f,fd,e,eps=0.0001):
    dif = 2*eps
    x = M
    while (dif > eps):
        x1 = x - (f(x,e,M)/fd(x,e))
        dif = np.abs(x1-x)
        x = x1
    return(x)


end_a = 100
End = end_a*365.25
h = 5
t = np.arange(0,End+1,h)
X, Y, R, Z = [], [], [], []
E, M = [], []

for j in t:
    print(j/t[-1])
    x, e, m = vec_Ju(j)
    X.append(x[0])
    Y.append(x[1])
    Z.append(x[2])
    E.append(e)
    M.append(m)
    R.append(np.sqrt(x[0]**2+x[1]**2+x[2]**2))

plt.figure(1)
plt.plot(t,E)
plt.show()


plt.figure(2)
plt.plot(t,M)
plt.show()


fig = plt.figure(3)
ax = fig.add_subplot(111, projection='3d')

ax.plot(X,Y,Z)
ax.set_zlim(-1,1)

fig.show()