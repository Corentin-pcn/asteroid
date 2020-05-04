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

def D_time(h, end, s = "f"):
    t = np.arange(0,end+1,h)
    if s == "f":
        return(t)
    elif s == "b":
        return(np.flip(t,axis=0))
    else:
        return(None)

def calc_orb(h, R0, t):
    r = R0
    Rn, Rpn, R, Rp = [], [], [], []
    for i in t:
        r = RK4(h,r,f)
        R.append([r[0], r[1], r[2]])
        Rp.append([r[3], r[4], r[5]])
        Rpn.append(np.sqrt(r[3]**2 + r[4]**2 + r[5]**2))
        Rn.append(np.sqrt(r[0]**2 + r[1]**2 + r[2]**2))
    return([t,R,Rp,Rn,Rpn])

def conv_OE(R, Rp, mu):
    a = 1/((2/np.linalg.norm(R))-((np.linalg.norm(Rp)**2)/mu))
    e = np.linalg.norm(((np.cross(Rp, np.cross(R,Rp)))/mu)-(R/np.linalg.norm(R)))
    kx, ky, kz = np.cross(R,Rp)/(np.linalg.norm(R)*np.linalg.norm(Rp))
    i = np.arccos(kz)
    return([a,e,i])

a = 2
k = 0.01720209895
End = 1200 #100*362.25
h = 0.1

R0 = [a, 0, 0, 0, (k/(np.sqrt(a))), 0]

t, R, Rp, Rn, Rpn = calc_orb(h, R0, D_time(h, End))
t2, R2, Rp2, R2n, Rp2n = calc_orb(-h, [R[-1][0], R[-1][1], R[-1][2], Rp[-1][0], Rp[-1][1], Rp[-1][2]], D_time(h, End, s="b"))

plt.figure(1)
plt.plot([i[0] for i in R],[i[1] for i in R])
plt.axis('equal')
plt.show()

plt.figure(2)
plt.plot(Rn,Rpn)
plt.show()

print((R2n[-1]-Rn[0])*1.5*(10**8))

A,E,I = [], [], []
for i in range(0, len(R)):
    a,e,i = conv_OE(R[-1], Rp[-1], 0.000295824)
    A.append(a)
    E.append(e)
    I.append(i)
    
plt.figure(3)
plt.plot(t,A)
plt.show()

plt.figure(4)
plt.plot(t,E)
plt.show()

plt.figure(5)
plt.plot(t,I)
plt.show()