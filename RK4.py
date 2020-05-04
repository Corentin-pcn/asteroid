# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 17:27:28 2020

@author: coco1
"""

import matplotlib.pyplot as plt
import numpy as np

def f(y,t):
    return(-y)

def RK4(h,y,f,t=0):
    k1 = f(y, t)
    k2 = f(y+((h/2)*k1), t+(h/2))
    k3 = f(y+((h/2)*k2), t+(h/2))
    k4 = f(y+(h*k3), t+h)
    return(y + h*((1/6)*(k1+(2*k2)+(2*k3)+k4)))
    
def error(ye,ya):
    return(np.abs((ye-ya)/ya))

H = [0.1, 0.01, 0.001]
c = ['g--','r--','b--']
l = [str(h) for h in H]
T, Yp, E, Et = [], [], [], []
y0 = 1


for h in H:
    y = y0
    t, yp, e = [], [], []
    for i in np.arange(h,10,h):
        t.append(i)
        y = RK4(h,y,f)
        yp.append(y)
        e.append(error(np.exp(-i), y))
    Et.append(e)
    E.append(np.mean(e))
    T.append(t)
    Yp.append(yp)

plt.figure(1) 
for i in range(0,3):
    plt.plot(T[i], Yp[i],c[i],label = l[i])
plt.plot(T[-1], [np.exp(-i) for i in T[-1]], 'y-.', label='exact')
plt.legend()
plt.xlabel('time')
plt.ylabel('y')
plt.show()

plt.figure(2)
plt.plot(H,E)
plt.xlabel('h')
plt.ylabel('error')
plt.show()

plt.figure(3)
for i in range(0,3):
    plt.plot(T[i], Et[i],c[i],label = l[i])
plt.legend()
plt.xlabel('time')
plt.ylabel('error')
plt.show()