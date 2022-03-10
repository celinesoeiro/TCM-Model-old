# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 16:41:28 2022

@author: celin
"""

import numpy as np
from random import seed, random
import matplotlib.pyplot as plt

params = {'a': 0.02, 'b': 0.2, 'c': -65, 'd': 8}
neuron_type = 'excit'
current_value = 10
current_start = 100
current_finish = 900
voltage_pick = 30
simulation_time = 100
time_step = 0.1
neurons = 1

seed(1)

# Parameters according Izhikevich article 
random_factor = random()

if (neuron_type == 'excitory' or 'excit'):
    a = params['a']
    b = params['b']
    c = params['c'] + 15*random_factor**2
    d = params['d'] - 6*random_factor**2

elif (neuron_type == 'inhibitory' or 'inib'):
    a = params['a'] + 0.08*random_factor
    b = params['b'] - 0.05*random_factor
    c = params['c']
    d = params['d']

# a = params['a']
# b = params['b']
# c = params['c']
# d = params['d']
    
time = np.arange(0, simulation_time + time_step, time_step) 

I = np.zeros(len(time))
I[current_start:current_finish] = current_value

v = np.zeros(len(time))    
v[0] = -65

u = np.zeros(len(time))    
u[0] = b*v[0]

# Izhikevich neuron equations
def dvdt(v, u, I):
    return 0.04*v**2 + 5*v + 140 - u + I

def dudt(v,u):
    return a*(b*v - u)

fired = []

for t in range(1, len(time)):     
    vc = v[t - 1]
    uc = u[t - 1]
    Ic = I[t - 1]
    
    if (vc >= voltage_pick):
        vc = v[t]
        v[t] = c
        u[t] = uc + d
        fired.append(t)
    
    else:
        # solve using RK 4th order

        # dv1 = time_step * dvdt(vc, uc, Ic)
        # dv2 = time_step * dvdt(vc + dv1 * 0.5, uc, Ic)
        # dv3 = time_step * dvdt(vc + dv2 * 0.5, uc, Ic)
        # dv4 = time_step * dvdt(vc + dv3, uc, Ic)
        # v[t] = 1/6 * (dv1 + 2*(dv2 + dv3) + dv4)
        
        # du1 = time_step * dudt(vc, uc)
        # du2 = time_step * dudt(vc, uc + du1 * 0.5)
        # du3 = time_step * dudt(vc, uc + du2 * 0.5)
        # du4 = time_step * dudt(vc, uc + du3)
        # u[t] = 1/6 * (du1 + 2*(du2 + du3) + du4)
        
         # solve using euler
        dv = dvdt(vc, uc, Ic)
        du = dudt(vc, uc)
        v[t] = vc + dv*time_step
        u[t] = uc + du*time_step


fig1, axs1 = plt.subplots(3)
fig1.suptitle("Regular Spiking")
axs1[0].plot(v)
axs1[0].set_title('V x t')
axs1[0].grid()

axs1[1].plot(I)
axs1[1].set_title('I x t')
axs1[1].grid()

axs1[2].plot(fired)
axs1[2].set_title('disparos')
axs1[2].grid()