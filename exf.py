# -*- coding: utf-8 -*-
"""
Created on Tue May  5 23:10:09 2020

@author: coco1
"""

import matplotlib.pyplot as plt #Importing library for plots
from mpl_toolkits.mplot3d import Axes3D
import numpy as np #Importing a math library

def f(R,t): #Function that represent the N-body equation with Jupiter for the asteroid
    r = np.sqrt(R[0]**2 + R[1]**2 + R[2]**2) #Cumpute norma of position x, y and z of Asteroid
    xi, yi, zi = vec_Ju(t) #Get the Elliptic position of Jupiter at time t
    ri = np.sqrt(xi**2 + yi**2 + zi**2) #Cumpute norma of position x, y and z of Jupiter
    di = np.sqrt((R[0]-xi)**2 + (R[1]-yi)**2 + (R[2]-zi)**2) #Compute norma of position difference between Asteroid and Jupiter
    G = 0.000295824 #Gravitationnal constant in au3 Sun mass-1 day-2
    MJ = 1/(1047.348625) #Mass of jupiter in Sun mass
    m = 0 #Mass of the asteroid
    
    xpp = (-(G*(1+m))/(r**3)*R[0])-G*MJ*(((R[0]-xi)/(di**3))+(xi/(ri**3))) #N-Body equation for each coordinate
    ypp = (-(G*(1+m))/(r**3)*R[1])-G*MJ*(((R[1]-yi)/(di**3))+(yi/(ri**3)))
    zpp = (-(G*(1+m))/(r**3)*R[2])-G*MJ*(((R[2]-zi)/(di**3))+(zi/(ri**3)))
    
    return(np.array([R[3], R[4], R[5], xpp, ypp, zpp])) #Sending back velocity and position in cartesian coordinate

def fE(E,e,M):
    return(E - e*np.sin(E) - M) #Function between mean and eccentric anomaly

def fdE(E,e):
    return(1 - e*np.cos(E)) #Derive of precedent function

def Newton(M,f,fd,e,eps=0.000001): #Function of the Newton method
    dif = 2*eps #Starting difference
    x = M #Starting x
    while (dif > eps): #while loop with stopping condition of epsilon
        x1 = x - (f(x,e,M)/fd(x,e)) #new x
        dif = np.abs(x1-x) #new difference between the two x
        x = x1 #replace by new x
    return(x)

def RK4(h,y,f,t=0): #RK4 integration method
    k1 = f(y, t)
    k2 = f(y+((h/2)*k1), t+(h/2))
    k3 = f(y+((h/2)*k2), t+(h/2))
    k4 = f(y+(h*k3), t+h)
    return(y + h*((1/6)*(k1+(2*k2)+(2*k3)+k4)))

def D_time(h, end, s = "f"):#Function that was used in the backward-forward method
    t = np.arange(0,end+1,h)#Create a time list
    if s == "f":
        return(t) #if forward return as it is
    elif s == "b":
        return(np.flip(t,axis=0)) #if backward reverse the list
    else:
        return(None) #if something return nothing

def calc_orb(h, R0, t): #Computation of the orbit 
    r = R0 #starting condition for integration
    Rn, Rpn, R, Rp = [], [], [], []
    for i in t: #Iteration on the time list
        r = RK4(h,r,f,i) #Find the r for a given time i
        R.append([r[0], r[1], r[2]]) #return list of x, y and z
        Rp.append([r[3], r[4], r[5]]) #return list of xp, yp and zp
        Rpn.append(np.sqrt(r[3]**2 + r[4]**2 + r[5]**2)) #Return list of the speed norma
        Rn.append(np.sqrt(r[0]**2 + r[1]**2 + r[2]**2)) #Return list of the position norma
        print(i/t[-1]) #Percentage time
    return([t,R,Rp,Rn,Rpn])

def conv_OE(R, Rp, mu): #Conversion between cartesian and orbital element
    a = 1/((2/np.linalg.norm(R))-((np.linalg.norm(Rp)**2)/mu)) #Semi-major axis
    e = np.linalg.norm(((np.cross(Rp, np.cross(R,Rp)))/mu)-(R/np.linalg.norm(R))) #Eccentricity
    kx, ky, kz = np.cross(R,Rp)/(np.linalg.norm(R)*np.linalg.norm(Rp))
    i = np.arccos(kz) #inclination
    return([a,e,i])

def R1(a): #First rotational matrix
    A = np.zeros([3,3])
    A[0,0] = 1
    A[1,1] = np.cos(a)
    A[2,2] = np.cos(a)
    A[2,1] = -np.sin(a)
    A[1,2] = np.sin(a)
    return(A)

def R3(a): #Second rotationnal matrix
    A = np.zeros([3,3])
    A[2,2] = 1
    A[0,0] = np.cos(a)
    A[1,1] = np.cos(a)
    A[1,0] = -np.sin(a)
    A[0,1] = np.sin(a)
    return(A)
    
def vec_Ju(t):
    #Orbital elements at J2000 for Jupiter
    a = 5.202603 #semi-major axis
    e = 0.048498 #Eccentricity
    i = np.deg2rad(-1.303) #Inclination
    Om = np.deg2rad(-100.46) #Longitude of ascending node
    om = np.deg2rad(86.13) #Perihelion distance
    
    n = 0.01720209895/np.sqrt(a**3) 
    M0 = np.deg2rad(20.02) #Starting Mean anomaly
    
    M = M0 + n*t #Mean anomaly formula
    E = Newton(M,fE,fdE,e) #Finding the Eccentric anomaly with Newton mehod using M as starting point
    
    X = a*(np.cos(E)-e)
    Y = a*np.sqrt(1-(e**2))*np.sin(E)
    
    A = np.array([X, Y, 0])
    x = np.matmul(R3(Om),np.matmul(R1(i),np.matmul(R3(om),A))) #Rotation of X and Y to give all the cartesian coordinate fo position
    return(x[0], x[1], x[2])

def pos_Ast(t):
    #Asteroid paramter at JD 2456600.5
    a = 5.454 #semi-major axis
    e = 0.3896 #Eccentricity
    n = 0.01720209895/np.sqrt(a**3)
    i = np.deg2rad(-108.358) #Inclination
    Om = np.deg2rad(-276.509) #Longitude of ascending node
    om = np.deg2rad(-226.107) #Perihelion distance
    
    t0 = 5055.5 #Difference between J2000 (starting point of Jupiter) and starting point of Asteroid
    
    M0 = np.deg2rad(146.88) #Starting Mean anomaly
    
    M = M0 + n*(t-t0) #Mean anomaly formula with time difference to get Asteroid and Jupiter position at same date
    
    E = Newton(M,fE,fdE,e) #Finding the Eccentric anomaly with Newton mehod using M as starting point
    X = a*(np.cos(E)-e)
    Y = a*np.sqrt(1-(e**2))*np.sin(E)
    r = a*(1-e*np.cos(E))
    Xd = -((n*(a**2))/r)*np.sin(E)
    Yd = ((n*(a**2))/r)*np.sqrt(1-(e**2))*np.cos(E)
    A = np.array([X, Y, 0])
    B = np.array([Xd, Yd, 0])
    x = np.matmul(R3(Om),np.matmul(R1(i),np.matmul(R3(om),A)))#Rotation of X and Y to give all the cartesian coordinate fo position
    y = np.matmul(R3(Om),np.matmul(R1(i),np.matmul(R3(om),B)))#Rotation of X and Y to give all the cartesian coordinate fo velocity
    return([x[0], x[1], x[2], y[0], y[1], y[2]])


#------------------------MAIN---------------------------------------------------



k = 0.01720209895
end_a = 1000 #End of the computation of the orbit in years
End = end_a*365.25 #Put in days
h = 1 #Step for integration

R0 = pos_Ast(0) #Get initial position of Asteroid

t, R, Rp, Rn, Rpn = calc_orb(h, R0, D_time(h, End)) #Launch the function for orbit integration
#t2, R2, Rp2, R2n, Rp2n = calc_orb(-h, [R[-1][0], R[-1][1], R[-1][2], Rp[-1][0], Rp[-1][1], Rp[-1][2]], D_time(h, End, s="b"))

m = 0
Xj, Yj, Zj = [], [], []

while(t[m] < 11.86*365.25): #Calculate the Jupiter position in 1 period of rotation for the plot 
    xj, yj, zj = vec_Ju(t[m])
    Xj.append(xj)
    Yj.append(yj)
    Zj.append(zj)
    m += 1

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d') #Creating 3D plot

ax.plot(Xj,Yj,Zj,label="Jupiter",color='b') #Ploting Jupiter orbit
ax.plot([i[0] for i in R],[i[1] for i in R],[i[2] for i in R],label="Asteroid",color='g') #Ploting Asteroid position
ax.legend()
fig.show()#show figure

#plt.figure(2)
#plt.plot(Rn,Rpn) #Print norma of speed depending on norma of position
#plt.show()

#print((R2n[-1]-Rn[0])*1.5*(10**8)) #Difference of the backward-forward

A,E,I = [], [], []
for i in range(0, len(Rn)): #Creating the list of orbital elements of Asteroid
    a,e,i = conv_OE(R[i], Rp[i], 0.000295824)
    A.append(a)
    E.append(e)
    I.append(i)
    
plt.figure(3) #Semi-major axis plot
plt.plot((t/365.25),A)
plt.title("Semi-major axis")
plt.xlabel("time (year)")
plt.ylabel("semi-major axis a (au)")
plt.show()

plt.figure(4) #Eccentricity plot
plt.plot((t/365.25),E)
plt.title("Eccentricity")
plt.xlabel("time (year)")
plt.ylabel("eccentricity e")
plt.show()

plt.figure(5) #Inclination plot in degrees
plt.plot((t/365.25),np.rad2deg(I))
plt.title("Inclination")
plt.xlabel("time (year)")
plt.ylabel("inclination i (°)")
plt.show()