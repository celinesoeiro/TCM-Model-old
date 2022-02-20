# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 19:12:42 2022

@author: celin
"""

import numpy as np
import pylab as plt

C = 150
Vr = -75
Vl = -45
k = 1.2
a = 0.01
b = 5
c = -56
d = 130
Vpico = 50

# Modelo Izhikevich

def dvdt(V,I,u):
    return (k*(V - Vr)*(V - Vl) - u + I)/C 

def dudt(V,u):
    return a*(b*(V - Vr) - u)

tmin = 0
tmax = 500
dt = 1e-3
t = np.arange(tmin,tmax,dt)


J = np.zeros(len(t))
v = np.zeros(len(t))
u = np.zeros(len(t))

J_values = [300,370,500,550]

for j in range(len(J_values)):
    J[20000:450000] = J_values[j]
    for i in range (len(t) - 1):
        v[i + 1] = dvdt(v[i], J[i], u[i]) + dt*dvdt(v[i], J[i], u[i])
        u[i + 1] = dudt(v[i], u[i]) + dt*dudt(v[i], u[i])

    fig, axs = plt.subplots(2, sharex=False, figsize=(7,10))
    fig.suptitle(f'Questao 6 - letra a - ii', fontsize=15)
    axs[0].set_title('V')
    axs[0].set(ylabel='Voltagem (mV)')
    axs[0].set(xlabel='tempo (ms)')
    axs[0].grid()
    axs[0].plot(t, v, 'k')
    
    axs[1].set_title('Corrente')
    axs[1].set(ylabel='I')
    axs[1].set(xlabel='t')
    axs[1].grid()
    axs[1].plot(t, J )