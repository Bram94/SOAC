# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 11:34:29 2018

@author: bramv
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import twolayer_PV_inversion as pv

mpl.rcParams['axes.titlesize'] = 22
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['axes.labelsize'] = 22
mpl.rcParams['axes.labelweight'] = 'bold'
for j in ('xtick','ytick'):
    mpl.rcParams[j+'.labelsize'] = 18
mpl.rcParams['legend.fontsize'] = 18


#%%
L = 180000e3
I = 7200 #There are I grid points
dx = 2 * L / (I - 1)
x = np.arange(-L, L+dx, dx)
I = len(x) #Could differ by 1 from the value set above, depending on rounding effects

a = 800e3
Q_max = 0.1
Q = Q_max * np.exp(- (x / a)**2)
eps = 0.9

g = 10.
f = 1e-4
t_flux = 600.
H1 = H2 = 1000.
R_rossby = 1. / f * np.sqrt(2 * (1 - eps) / (1 + eps) * g * H1 * H2 / (H1 + H2))



t_max = 240 * 3600. #In seconds
dt = 1.
t = np.arange(0, t_max+dt, dt)
N = len(t)



def RK4(f, y, t, dt):
    #f = f(t, y)
    k1 = f(t, y)
    k2 = f(t + dt/2., y + k1/2.)
    k3 = f(t + dt/2., y + k2/2.)
    k4 = f(t + dt, y + k3)
    return dt/6. * (k1 + 2.*k2 + 2.*k3 + k4) # = dy

def f_derivatives(t, y): #y = np.array([zeta1, zeta_2, delta1, delta2, phi_1, phi_2, h_1, h_2])
    zeta1, zeta2, delta1, delta2, phi1, phi2, h1, h2 = y
    
    if t <= t_flux: 
        M1 = -Q; M2 = Q/eps
    else:
        M1 = M2 = 0.
    
    dzeta1dt = -f * delta1
    dzeta2dt = -f * delta2
    
    ddelta1dt = f * zeta1
    ddelta2dt = f * zeta2
    ddelta1dt[1:-1] -= (phi1[2:] + phi1[:-2] - 2. * phi1[1:-1]) / dx**2
    ddelta2dt[1:-1] -= (phi2[2:] + phi2[:-2] - 2. * phi2[1:-1]) / dx**2
    
    #Use a backward/forward difference scheme to approximate the second derivative at the boundaries.
    #https://en.wikipedia.org/wiki/Finite_difference_coefficient#Forward_and_backward_finite_difference
    #All these schemes lead however to clearly different solutions compared to simulations in which boundary effects
    #don't play a role.
    c = {1: [1, -2, 1], 2: [2, -5, 4, -1], 3: [35./12., -26./3., 19./2., -14./3., 11./12.],
         4:[15./4., -77./6., 107./6., -13., 61./12., -5./6.],
         5: [203./45., -87./5., 117./4., -254./9., 33./2., -27./5., 137./180.]}
    
    order = 3
    for j in range(len(c[order])):
        ddelta1dt[0] -= c[order][j] * phi1[j] / dx**2
        ddelta2dt[0] -= c[order][j] * phi2[j] / dx**2
        ddelta1dt[-1] -= c[order][j] * phi1[- (j+1)] / dx**2
        ddelta2dt[-1] -= c[order][j] * phi2[- (j+1)] / dx**2
        
    dphi1dt = -g * (H1*delta1 + eps*H2*delta2) + g * (M1 + eps*M2)
    dphi2dt = -g * (H1*delta1 + H2*delta2) + g * (M1 + M2)
    dh1dt = -H1 * delta1 + M1
    dh2dt = -H2 * delta2 + M2
    
    return np.array([dzeta1dt, dzeta2dt, ddelta1dt, ddelta2dt, dphi1dt, dphi2dt, dh1dt, dh2dt])
    
zeta1 = np.zeros(I)
zeta2 = zeta1.copy()
delta1 = zeta1.copy()
delta2 = zeta2.copy()
h1 = H1 * np.ones(I)
h2 = H2 * np.ones(I)
phi1 = g * (h1 + eps*h2)
phi2 = g * (h1 + h2)

y = np.array([zeta1, zeta2, delta1, delta2, phi1, phi2, h1, h2])
for n in range(N):
    y += RK4(f_derivatives, y, t[n], dt)
    print(n, y[-1].min(), y[-1].max())
zeta1, zeta2, delta1, delta2, phi1, phi2, h1, h2 = y



x /= 1000.
#%%
pv.L = 10000e3
pv.I = 400 #There are I grid points
pv.dx = 2 * pv.L / pv.I
pv.x = np.arange(-pv.L, pv.L+pv.dx, pv.dx)
pv.I = len(pv.x) #Could differ by 1 from the value set above, depending on rounding effects

x_pv = pv.x / 1000.
v1_pv, v2_pv, h1_pv, h2_pv, _, _ = pv.calculate_v(a)
zeta1_pv = (v1_pv[2:] - v1_pv[:-2]) / (2. * pv.dx)
zeta2_pv = (v2_pv[2:] - v2_pv[:-2]) / (2. * pv.dx)



#%%
zeta1_g = np.zeros(I); zeta2_g = zeta1_g.copy()
zeta1_g[1:-1] = 1./ (f * dx**2) * (phi1[2:] + phi1[:-2] - 2. * phi1[1:-1])
zeta2_g[1:-1] = 1./ (f * dx**2) * (phi2[2:] + phi2[:-2] - 2. * phi2[1:-1])

plt.figure(figsize = (10, 7))
#plt.plot(x, h1, x, h2)
data_plot = (x >= -10000) & (x <= 10000)
plt.plot(x[data_plot], zeta1[data_plot], 'b-', x[data_plot], zeta2[data_plot], 'r-')
plt.plot(x[data_plot], zeta1_g[data_plot], 'b--', x[data_plot], zeta2_g[data_plot], 'r--')
plt.plot(x_pv[1:-1], zeta1_pv, 'cyan', x_pv[1:-1], zeta2_pv, 'orange')
plt.ticklabel_format(axis = 'y', style = 'sci', scilimits = (-2, 2))
#plt.ylim([-1e-6, 1e-6])
plt.xlabel('x (km)'); plt.ylabel('$\zeta$ (s$^{-1}$)')
n_hours = int(t_max / 3600.)
plt.title(str(n_hours) + (' hours' if n_hours != 1 else ' hour'))
plt.legend(['layer 1', 'layer 2', 'layer 1 geostrophic', 'layer 2 geostrophic', 'layer 1 PV inversion', 'layer 2 PV inversion'], loc = (1.01, 0.56))
plt.savefig('2layer_'+str(n_hours)+'hours.jpg', dpi = 240, bbox_inches = 'tight')
plt.show()

"""
fig, ax = plt.subplots(2, 1, figsize = (10, 14))
ax[0].plot(x[data_plot], phi1[data_plot], 'b-')
ax[1].plot(x[data_plot], phi2[data_plot], 'r-')
plt.show()
"""