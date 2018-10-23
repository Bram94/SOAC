# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 11:19:12 2018

@author: bramv
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['axes.titlesize'] = 22
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['axes.labelsize'] = 22
mpl.rcParams['axes.labelweight'] = 'bold'
for j in ('xtick','ytick'):
    mpl.rcParams[j+'.labelsize'] = 18
mpl.rcParams['legend.fontsize'] = 18



#%%%
Q_max = 0.1
eps = 0.9

g = 10.
f = 1e-4
t_flux = 600.
H1 = H2 = 1000.
R_rossby = 1. / f * np.sqrt(2 * (1 - eps) / (1 + eps) * g * H1 * H2 / (H1 + H2))
print(R_rossby)
        


def calculate_v(a):
    Q = Q_max * np.exp(- (x / a)**2)
    M1 = -Q; M2 = Q/eps  
    h_p1 = M1 * t_flux
    h_p2 = M2 * t_flux

    h1 = H1 + h_p1 ; h2 = H2 + h_p2
    zeta_pot1 = f / h1; zeta_pot2 = f / h2
    
    v1 = np.zeros(I); v2 = np.zeros(I)
    n = 0
    convergence_fac = 1e-3
    while True:
        #Assume that v remains zero at the boundaries x = -L and x = +L
        R1 = np.zeros(I); R2 = R1.copy()
        for i in range(1, I-1):
            h1[i] = ((v1[i+1] - v1[i-1]) / (2 * dx) + f) / zeta_pot1[i]
            h2[i] = ((v2[i+1] - v2[i-1]) / (2 * dx) + f) / zeta_pot2[i]
            
            A1 = f / (g * (1-eps)) * zeta_pot1[i]; A2 = f / (g * (1-eps)) * zeta_pot2[i]
            B1 = h1[i] * (zeta_pot1[i+1] - zeta_pot1[i-1]) / (2 * dx) - eps * f * v2[i] * zeta_pot1[i] / (g * (1 - eps))
            B2 = h2[i] * (zeta_pot2[i+1] - zeta_pot2[i-1]) / (2 * dx) - f * v1[i] * zeta_pot2[i] / (g * (1 - eps))
            
            R1[i] = v1[i] - (v1[i+1] + v1[i-1] - dx**2 * B1) / (2 + A1 * dx**2)
            R2[i] = v2[i] - (v2[i+1] + v2[i-1] - dx**2 * B2) / (2 + A2 * dx**2)
            v1[i] -= R1[i]; v2[i] -= R2[i]
            
        if n == 0:
            initial_max_v_diff1 = np.max(np.abs(R1)); initial_max_v_diff2 = np.max(np.abs(R2))
            print('initial_max_v_diff1 = ', initial_max_v_diff1, 'initial_max_v_diff2 = ', initial_max_v_diff2)
        elif np.max(np.abs(R1)) < convergence_fac * initial_max_v_diff1 and np.max(np.abs(R2)) < convergence_fac * initial_max_v_diff2:
            break
        n += 1
    #print(n, v1.max(), v1.min(), v2.max(), v2.min())
    return v1, v2, h1, h2, h_p1, h_p2

def calculate_energy_conversion_ratio(v1, v2, h1, h2, h_p1, h_p2):
    P_g = g * np.sum((h1 - H1)**2 + eps * (h2 - H2)**2 + 2 * eps * (h1 - H1) * (h2 - H2))
    #Exclude the factor rho_1/2, as it is the same for every energy expression
    h1_0 = h_p1; h2_0 = h_p2
    P_0 = g * np.sum(h1_0**2 + eps * h2_0**2 + 2 * eps * h1_0 * h2_0)
    K_g = np.mean(h1) * np.sum(v1**2) + eps * np.mean(h2) * np.sum(v2**2)
    return K_g / (P_0 - P_g), K_g



if __name__ == '__main__': #This part is only executed if this script runs as the main script, 
#not when it is imported from another script
    if True:
        #%%
        a = 800e3 / 1000.
        L = a * 5
        I = 400 #There are I grid points
        dx = 2 * L / I
        x = np.arange(-L, L+dx, dx)
        I = len(x) #Could differ by 1 from the value set above, depending on rounding effects
        
        v1, v2, h1, h2, h_p1, h_p2 = calculate_v(a)
        
        #%%
        fig, ax = plt.subplots(3, 1, figsize = (10, 20))
        x = x / 1000.
        data_plot = (x >= -100) & (x <= 100)
        ax[0].plot(x[data_plot][1:-1], (v1[data_plot][2:] - v1[data_plot][:-2]) / (2 * dx), x[data_plot][1:-1], (v2[data_plot][2:] - v2[data_plot][:-2]) / (2 * dx))
        ax[1].plot(x[data_plot], H1 + h_p1[data_plot], 'b--', x[data_plot], h1[data_plot], 'b-')
        ax[2].plot(x[data_plot], H2 + h_p2[data_plot], 'r--', x[data_plot], h2[data_plot], 'r-')
        plt.show()

    #%%
    I = 400
    scale_factors = np.power(10, np.arange(-2., 3.001, 0.25))
    a_array = scale_factors * R_rossby
    energy_conversion_ratios = np.zeros(len(a_array))
    K_g = np.zeros(len(a_array))
    for i in range(0, len(a_array)):
        L = a_array[i]*5
        dx = 2 * L / I
        x = np.arange(-L, L+dx, dx)
    
        v1, v2, h1, h2, h_p1, h_p2 = calculate_v(a_array[i])
        
        energy_conversion_ratios[i], K_g[i] = calculate_energy_conversion_ratio(v1, v2, h1, h2, h_p1, h_p2)
       
    #%%
    plt.figure(figsize = (10, 7))
    plt.semilogx(scale_factors, energy_conversion_ratios)
    plt.ylim([0.0, 0.5])
    plt.xlabel('a/R'); plt.ylabel('K / $\Delta$P')
    plt.title('Energy conversion ratio as a function of a/R')
    plt.savefig('2layer_energy_conversion_ratios.jpg', dpi = 240, bbox_inches = 'tight')
    plt.show()
    
    plt.figure(figsize = (10, 7))
    plt.semilogx(scale_factors, K_g/np.max(K_g))
    #plt.ylim([0.0, 0.5])
    plt.xlabel('a/R'); plt.ylabel(r'KE')
    plt.grid()
    plt.title('Kinetic Energy as a function of a/R')
    plt.savefig('2layer_KE.jpg', dpi = 240, bbox_inches = 'tight')
    plt.show()   