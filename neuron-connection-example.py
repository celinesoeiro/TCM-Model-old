# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 21:46:31 2022

@description: Create two neurons and connect them

@author: Celine
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

# =============================================================================
# Equations
# =============================================================================
# Izhikevich neuron equations
def dvdt(v, u, I):
    return 0.04*v**2 + 5*v + 140 - u + I

def dudt(v,u):
    return a*(b*v - u)

# =============================================================================
# Simulation
# =============================================================================

# Time array    
t = 100
dt = 0.25
time = np.arange(0, t + dt, dt)

# Simulation parameters
initial_voltage = -65
initial_current = 20
peak_voltage = 30

# Neuron parameters - Regular spiking
neuron_params = {'a': 0.02, 'b': 0.2, 'c': -65, 'd': 8}

# neuron 1 variables
v1 = np.zeros(len(time))
u1 = np.zeros(len(time))
I1 = np.ones(len(time))*initial_current
fired1 = np.zeros(len(time))

# neuron 1 variables
v2 = np.zeros(len(time))
u2 = np.zeros(len(time))
I2 = np.concatenate((np.ones(20), np.ones(60)*initial_current, np.ones(20))) 
fired2 = np.zeros(len(time))

for i in range(1, len(time)):    
    # Izhikevich neuron - RS Neuron
    a = neuron_params['a']
    b = neuron_params['b']
    c = neuron_params['c'] 
    d = neuron_params['d'] 
    
    v1[0] = initial_voltage
    u1[0] = b*v1[0]
    
    v2[0] = initial_voltage
    u2[0] = b*v2[0]
    
    v1_aux = v1[i - 1]
    u1_aux = u1[i - 1]
    I1_aux = I1[i - 1]
    
    v2_aux = v2[i - 1]
    u2_aux = u2[i - 1]
    I2_aux = I2[i - 1]
    
    if (v1_aux >= peak_voltage):
        v1_aux = v1[i]
        v1[i] = c
        u1[i] = u1_aux + d
        fired1[i] = 1
       
    else:            
        # solve using Euler
        dv1 = dvdt(v1_aux, u1_aux, I1_aux)
        du1 = dudt(v1_aux, u1_aux)
        v1[i] = v1_aux + dv1*dt
        u1[i] = u1_aux + du1*dt
        
    if (v2_aux >= peak_voltage):
        v2_aux = v2[i]
        v2[i] = c
        u2[i] = u2_aux + d
        fired2[i] = 1
       
    else:            
        # solve using Euler
        dv2 = dvdt(v2_aux, u2_aux, I2_aux)
        du2 = dudt(v2_aux, u2_aux)
        v2[i] = v2_aux + dv2*dt
        u2[i] = u2_aux + du2*dt
        

fig, axs = plt.subplots(ncols=2)
fig.suptitle('neurons')
axs[0].set_title('neuron 1')
axs[0].plot(time, v1)
axs[0].plot(time, u1)
axs[0].legend('v1u1', ncol=1, loc='upper left')

axs[1].set_title('neuron 2')
axs[1].plot(time, v2)
axs[1].plot(time, u2)    
axs[1].legend('v2u2', ncol=1, loc='upper left');



