#%%
import numpy as np
import matplotlib.pyplot as plt


#%%
def rk4(function, t, y0, h):
    y = np.zeros((len(t), len(y0)))
    y[0] = y0

    for i in range(len(t)-1):
        k1 = function(t[i], y[i], h, i)
        k2 = function(t[i] + h/2, y[i] + h/2 * k1, h, i)
        k3 = function(t[i] + h/2,  y[i] + h/2 * k2, h, i)
        k4 = function(t[i] + h, y[i] + h * k3, h, i)
        y[i+1] = y[i] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    return y


def motion_simplified(t, r, h, i):
    G = 0.000295824
    m = 0
    f = np.zeros(len(r))
    f[0:3] = r[3:6]
    norme = np.sqrt(r[0]**2 + r[1]**2 + r[2]**2)
    coef = -(G * (1+m) / norme**3)
    f[3:6] = coef * r[0:3]
    return f


#%% Initial conditions of asteroid

a = 2   # au
e = 0
i = 0

G = 0.000295824
k = np.sqrt(G)


#%%
r0 = np.array([a, 0, 0])
r0_dot = np.array([0, k/np.sqrt(a), 0])
R0 = np.concatenate((r0, r0_dot))

#%%
h = 0.1
t = np.arange(0, 1200, h)

y = rk4(motion_simplified, t, R0, h)


#%%
plt.figure('Orbit')
plt.plot(y[:, 0], y[:, 1])
plt.xlabel('x')
plt.ylabel('y')

plt.axis('equal')
plt.show()

# %% backward forward integration

def backward_forward(function, t, y0, h):
    y_forward = rk4(function, t, y0, h)
    new_y0 = y_forward[-1]
    y_backward = rk4(function, np.flip(t), new_y0, -h)
    return y_forward, y_backward


h_bf = 1
# to do h = [0.1, 1, 10] and calculate the computation time
n = 36525 # 100 ans
t_bf = np.arange(0, n, h_bf)

yf, yb = backward_forward(motion_simplified, t_bf, R0, h_bf)
error_bf = yf[0] - yb[-1]
error_distance = np.linalg.norm(error_bf[0:2]) * 150e6
print('error on distance =', error_distance)

# %% orbital elements calculation

def orbital_elements(y):
    a, e, inc = np.zeros(len(y)), np.zeros(len(y)), np.zeros(len(y))
    r, r_dot = y[:, 0:3], y[:, 3:6]

    for i in range(len(y)):
        a[i] = ((2 / np.linalg.norm(r[i])) - (np.linalg.norm(r_dot[i])**2 / G))**-1

        temp = (np.cross(r_dot[i], np.cross(r[i], r_dot[i])) / G) - (r[i] /np.linalg.norm(r[i]))
        e[i] = np.linalg.norm(temp)

        k = (np.cross(r[i], r_dot[i])) / (np.linalg.norm(r[i]) * np.linalg.norm(r_dot[i]))
        inc[i] = np.arccos(np.deg2rad(k[2]))
    return a, e, inc

a, e, i = orbital_elements(yf)
i = np.rad2deg(i)


# %% Jupiter perturbations

def motion_jupiter(t, r, h, i):
    G = 0.000295824
    m = 0
    f = np.zeros(len(r))
    mj = 1 / 1047.348625
    r_j = r_j_calculation(G, h, i)

    f[0:3] = r[3:6]
    norme = np.sqrt(r[0]**2 + r[1]**2 + r[2]**2)
    without_pertu = -(G * (1+m) / norme**3)
    jupiter_pertu = G * mj * ((r[0:3]-r_j) / np.linalg.norm(r[0:3]-r_j)**3 + r_j / np.linalg.norm(r_j)**3)
    # print(jupiter_pertu)
    f[3:6] = without_pertu * r[0:3] - jupiter_pertu
    return f


def r_j_calculation(G, h, i):
    aj = 5.2
    T = 11.86*365.25
    theta = (2 * np.pi) / T
    rj = np.zeros(3)

    rj[0] = aj * np.cos(h*(i+1) * theta)
    rj[1] = aj * np.sin(h*(i+1) * theta)
    # print('rj =', rj)
    return rj

h_j = 10
n_j = 365.25 * 12
t_j = np.arange(0, n_j, h_j)

y_j = rk4(motion_jupiter, t_j, R0, h_j)
# jupiter_orbit = 

plt.figure('Orbit')
plt.plot(y_j[:, 0], y_j[:, 1])
plt.xlabel('x')
plt.ylabel('y')

plt.axis('equal')
# plt.axis('off')
plt.show()


#%% Orbital element variation

a_j, e_j, i_j = orbital_elements(y_j)

plt.figure('Perturbations a')
plt.plot(t_j, a_j)

plt.figure('Pertubations e')
plt.plot(t_j, e_j)

plt.figure('Pertubations i')
plt.plot(t_j, i_j)

plt.show()



# %%
