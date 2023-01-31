"""
Neuron connection

@author: Celine Soeiro

Goal: 
    Connect two or more Izhikevich neuron using Tsodyks and Markram synapse
    
Regular Spiking neuron parameters:
    'params': {'a': 0.02, 'b': 0.2, 'c': -65, 'd': 8}, 
    'current_value': 10,
    'current_start': 100,
    'current_finish':  900,
    'voltage_pick': 30,
    'simulation_time': 100,
    'time_step': 0.1,
    'neuron_type': 'excitatory',
    'initial_voltage': -65
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

# Tsodyks and Markram synapse equations
def x_eq(x, t_d, u, delta):
    # fraction of the neurotransmitter resources that remain available after synaptic transmission
    return -(1-x)/t_d - u*x*delta

def u_eq(u, t_f, U, delta):
    # fraction of available neurotransmitter resources ready to be used
    return -(u/t_f) + U*(1 - u)*delta
    
def I_eq(I, t_s, A, u, x, delta):
    # post-synaptic current
    return -(I/t_s) + A*u*x*delta

# =============================================================================
# Simulation
# =============================================================================

# Time array    
t = 100
dt = 0.25
time = np.arange(0, t + dt, dt)

# Simulation parameters
initial_voltage = -65
initial_current = 10
peak_voltage = 30

# Regular Spiking Neuron parameters
neuron_params = {'a': 0.02, 'b': 0.2, 'c': -65, 'd': 8}

v = np.zeros(len(time))
u = np.zeros(len(time))
I = np.ones(len(time))*initial_current
fired = np.zeros(len(time))

for i in range(1, len(time)):    
    # Izhikevich neuron - RS Neuron
    a = neuron_params['a']
    b = neuron_params['b']
    c = neuron_params['c'] 
    d = neuron_params['d'] 
    
    v[0] = initial_voltage
    u[0] = b*v[0]
    
    v_aux = v[i - 1]
    u_aux = u[i - 1]
    I_aux = I[i - 1]
    
    
    if (v_aux >= peak_voltage):
        v_aux = v[i]
        v[i] = c
        u[i] = u_aux + d
        fired[i] = 1
       
    else:            
        # solve using Euler
        dv = dvdt(v_aux, u_aux, I_aux)
        du = dudt(v_aux, u_aux)
        v[i] = v_aux + dv*dt
        u[i] = u_aux + du*dt
    
plt.plot(time, v)
plt.plot(time, u)    
plt.legend('vu', ncol=1, loc='upper left');
    
    

