# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 14:33:47 2020

@author: coco1
"""
import matplotlib.pyplot as plt
import numpy as np

def f(R,t):
    r = np.sqrt(R[0]**2 + R[1]**2 + R[2]**2)
    G = 0.000295824
    c = -(G)/(r**3)
    return(np.array([R[3], R[4], R[5], c*R[0], c*R[1], c*R[2]]))

def RK4(h,y,f,t=0):
    k1 = f(y, t)
    k2 = f(y+((h/2)*k1), t+(h/2))
    k3 = f(y+((h/2)*k2), t+(h/2))
    k4 = f(y+(h*k3), t+h)
    return(y + h*((1/6)*(k1+(2*k2)+(2*k3)+k4)))

a = 2
k = 0.01720209895

R0 = [a, 0, 0, 0, (k/(np.sqrt(a))), 0]

h = 0.1
r = R0
pr = 0
t, x, y, z, R, Rp = [], [], [], [], [], []
for i in np.arange(0,(100*365),h):
    t.append(i)
    Npr = int((100*i)/(100*365))
    if Npr != pr:
        print(Npr)
        pr = Npr
    r = RK4(h,r,f)
    x.append(r[0])
    y.append(r[1])
    z.append(r[2])
    Rp.append(np.sqrt(r[3]**2 + r[4]**2 + r[5]**2))
    R.append(np.sqrt(r[0]**2 + r[1]**2 + r[2]**2))
    
plt.figure(1)
plt.plot(x,y)
plt.axis('equal')
plt.show()

plt.figure(2)
plt.plot(R,Rp)
plt.show()