# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 13:54:58 2018

@author: bramv
"""
import numpy as np
import matplotlib.pyplot as plt



L = 2500e3
I = 400 #There are I + 1 grid points
dx = 2 * L / I
x = np.arange(-L, L+dx, dx)

eta_0 = -50.
x_0 = 0.
g = 10.
f = 5e-4
h_mean = 1e3
R_rossby = np.sqrt(g * h_mean) / f




def calculate_v(a):
    eta = eta_0 * np.exp(- ((x - x_0) / a)**2)
    h = h_mean + eta
    zeta_pot = f / h

    v = np.zeros(I+1)
    n = 0
    while True:
        #Assume that v remains zero at the boundaries x = -L and x = +L
        v_old = v.copy()
        for i in range(1, I):
            h[i] = ((v[i+1] - v[i-1]) / (2 * dx) + f) / zeta_pot[i]
            
            A = f / g * zeta_pot[i]
            B = h[i] * (zeta_pot[i+1] - zeta_pot[i-1]) / (2 * dx)
            
            R_i = v[i] - (v[i+1] + v[i-1] - dx**2 * B) / (2 + A * dx**2)
            v[i] -= R_i
            
        if n == 0:
            initial_max_v_diff = np.max(np.abs(v - v_old))
            print('initial_max_v_diff = ', initial_max_v_diff)
        elif np.max(np.abs(v - v_old)) < 0.01 * initial_max_v_diff:
            break
        n += 1
    print(n, v.max(), v.min())
    return v, h, eta
    
v,_,_ = calculate_v(60e3)

plt.figure()
plt.plot(x / 1000., v)
plt.show()


def calculate_energy_conversion_ratio(v, h, eta):
    #Exclude the factor rho_1/2, as it is the same for every energy expression
    P_0 = g * np.sum(eta**2)
    P_g = g * np.sum((h - h_mean)**2)
    K_g = np.sum(h * v**2)
    
    print(P_0, P_g, K_g, np.mean(h), np.sum(v**2))
    return K_g / (P_0 - P_g)

scale_factors = np.power(10, np.arange(0, 2.001, 1))
a_array = scale_factors * R_rossby
energy_conversion_ratios = np.zeros(len(a_array))
for i in range(0, len(a_array)):
    L = a_array[i]*5
    dx = 2 * L / I
    x = np.arange(-L, L+dx, dx)

    v, h, eta = calculate_v(a_array[i])
    
    energy_conversion_ratios[i] = calculate_energy_conversion_ratio(v, h, eta)
    
plt.figure()
plt.semilogx(scale_factors, energy_conversion_ratios)
plt.show()