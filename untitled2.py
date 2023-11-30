#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 16:33:22 2023

@author: celinesoeiro
"""

import numpy as np
import matplotlib.pyplot as plt
from random import seed, random

def u_after_spike(u, U, t):
    return u[0][t] + U*(1 - u[0][t])

def tm_eq():
    # Utilization parameter -> Represents the fraction of available resources ready for use (release probability)
    u[0][t + 1] = -u[0][t]/tau_f + U*(1 - u[0][t])*AP
    # Fraction of resources that remain available after neurotransmitter depletion
    x[0][t + 1] = (1 - x[0][t])/tau_d - u[0][t]*x[0][t]*AP
    # Post-synaptic current
    I[0][t + 1] = -I[0][t]/tau_s + A*u[0][t]*x[0][t]*AP
    

seed(1)
random_factor = random()
    
ms = 1000
dt = 1/ms
simulation_time = 500 # 500 s

# 1 second
time = np.arange(0, simulation_time + dt, dt)

current = 5
voltage_pick = 30

# RS
params = {'a': 0.02, 'b': 0.2, 'c': -65, 'd': 8}

# Excitatory
a = params['a']
b = params['b']
c = params['c'] + 15*random_factor**2
d = params['d'] - 6*random_factor**2

# Inhibitory
a = params['a'] + 0.08*random_factor
b = params['b'] - 0.05*random_factor
c = params['c']
d = params['d']
        
# Excitatory conditions
tau_f = [670, 17]
tau_d = [138, 671]
tau_s = [3, 3]
U = [0.09, 0.5]
A = [0.37, 0.63]

def dvdt(v, u, I):
    return 0.04*v**2 + 5*v + 140 - u + I

def dudt(v,u):
    return a*(b*v - u)

# membrane potential vector
v = np.zeros(len(time))    
v[0] = -65

# membrane recovery variable vector
u = np.zeros(len(time))    
u[0] = b*v[0]

# Current vector input
I = current*np.ones(len(time))

fired = []

x = np.zeros((2,len(time)))
y = np.zeros((2,len(time)))
I_s = np.zeros((2,len(time)))
PSC = np.zeros(len(time))
AP = 0

for t in range(1, len(time)):
    # Pre synaptic neuron
    v_aux = v[t - 1]
    u_aux = u[t - 1]
    I_aux = I[t - 1]
    
    if (v_aux >= voltage_pick):
        v_aux = v[t]
        v[t] = c
        u[t] = u_aux + d
        fired.append(t)
        AP = 1
       
    else:            
        # solve using Euler
        dv = dvdt(v_aux, u_aux, I_aux)
        du = dudt(v_aux, u_aux)
        v[t] = v_aux + dv*dt
        u[t] = u_aux + du*dt
        AP = 0
    
    if(AP == 1):        
        print('before ',x[t], AP)
    
    for s in range(2):    
        # Utilization parameter -> Represents the fraction of available resources ready for use (release probability)
        x[s][t] = -x[s][t - 1]/tau_f[s] + U[s]*(1 - x[s][t - 1])*AP
        # Fraction of resources that remain available after neurotransmitter depletion
        y[s][t] = (1 - y[s][t - 1])/tau_d[s] - x[s][t - 1]*y[s][t - 1]*AP
        # Post-synaptic current
        I_s[s][t] = -I_s[s][t - 1]/tau_s[s] + A[s]*x[s][t - 1]*y[s][t - 1]*AP
        
    i_sum = np.sum(I_s)

    PSC[t] = i_sum
    
    if(AP == 1):        
        print('after ',x[t], AP)
        
plt.figure(1)
plt.suptitle("Voltage")
plt.plot(v, 'b', label='voltage (mV)')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.show()

plt.figure(1)
plt.suptitle("PSC")
plt.plot(PSC, 'b', label='voltage (mV)')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.show()