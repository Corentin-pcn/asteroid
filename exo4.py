# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 14:33:47 2020

@author: coco1
"""
import matplotlib.pyplot as plt
import numpy as np

def f(R,t):
    r = np.sqrt(R[0]**2 + R[1]**2 + R[2]**2)
    xi, yi, zi = vec_Ju(t)
    ri = np.sqrt(xi**2 + yi**2 + zi**2)
    di = np.sqrt((R[0]-xi)**2 + (R[1]-yi)**2 + (R[2]-zi)**2)
    G = 0.000295824
    MJ = 1/(1047.348625)
    c = -(G)/(r**3)
    
    xpp = c*R[0]-G*MJ*(((R[0]-xi)/(di**3))+(xi/(ri**3)))
    ypp = c*R[1]-G*MJ*(((R[1]-yi)/(di**3))+(yi/(ri**3)))
    zpp = c*R[2]-G*MJ*(((R[2]-zi)/(di**3))+(zi/(ri**3)))
    
    return(np.array([R[3], R[4], R[5], xpp, ypp, zpp]))

def fE(E,e,M):
    return(E - e*np.sin(E) - M)

def fdE(E,e):
    return(1 - e*np.cos(E))

def Newton(M,f,fd,e,eps=0.000001):
    dif = 2*eps
    x = M
    while (dif > eps):
        x1 = x - (f(x,e,M)/fd(x,e))
        dif = np.abs(x1-x)
        x = x1
    return(x)

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
        r = RK4(h,r,f,i)
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
    return(x[0], x[1], x[2])

a = 2
k = 0.01720209895
end_a = 100
End = end_a*365.25
h = 5

R0 = [a, 0, 0, 0, (k/(np.sqrt(a))), 0]

t, R, Rp, Rn, Rpn = calc_orb(h, R0, D_time(h, End))
t2, R2, Rp2, R2n, Rp2n = calc_orb(-h, [R[-1][0], R[-1][1], R[-1][2], Rp[-1][0], Rp[-1][1], Rp[-1][2]], D_time(h, End, s="b"))

plt.figure(1)
plt.scatter([i[0] for i in R],[i[1] for i in R],marker='.')
plt.axis('equal')
plt.show()

#plt.figure(2)
#plt.plot(Rn,Rpn)
#plt.show()

print((R2n[-1]-Rn[0])*1.5*(10**8))

A,E,I = [], [], []
for i in range(0, len(Rn)):
    a,e,i = conv_OE(R[i], Rp[i], 0.000295824)
    A.append(a)
    E.append(e)
    I.append(i)
    
plt.figure(3)
plt.plot(t,A)
plt.title("Semi-major axis")
plt.show()

plt.figure(4)
plt.plot(t,E)
plt.title("Eccentricity")
plt.show()

plt.figure(5)
plt.plot(t,I)
plt.title("Inclinesion")
plt.show()