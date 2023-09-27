#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 08:14:59 2023

@author: celinesoeiro
"""

import random
import numpy as np

random.seed(0)
random_factor = np.round(random.random(),2)

from cortex_functions import poisson_spike_generator, izhikevich_dvdt, izhikevich_dudt, plot_raster, plot_voltage, get_frequency

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================
ms = 1000                                       # 1 second = 1000 milliseconds
sim_time = 3                                    # seconds
dt = 1/ms                                      # seconds
fs = 1/dt                                       # Hz (Sampling Frequency)

# Voltage parameters
v_threshold = 30
v_resting = -65

# Calculate the number of time steps
num_steps = int(sim_time / dt)

n_D = 1
n_CI = 1
n_TC = 1
total_neurons = n_D + n_CI

spikes_D = np.zeros((1, num_steps))
spikes_CI = np.zeros((1, num_steps))

f_thalamus = 22                                  # Hz (Thalamus frequency)
c_thalamus = 40                                 # mA (Thalamus input value to the cortex)

# =============================================================================
# Izhikevich neuron parameters
# =============================================================================
#    0-RS  1-IB  2-FS 3-LTS 4-TC  5-TR 
a = [0.02, 0.02, 0.1, 0.02, 0.02, 0.02]
b = [0.2,  0.2,  0.2, 0.25, 0.25, 0.25]
c = [-65,  -55,  -65, -65,   -65,  -65]
d = [8,    4,      2,   2,  0.05, 2.05]

# D -> 1 RS neuron
a_D = np.c_[a[0]*np.ones((1, 1)), a[1]*np.ones((1, 0))]
b_D = np.c_[b[0]*np.ones((1, 1)), b[1]*np.ones((1, 0))]
c_D = np.c_[c[0]*np.ones((1, 1)), c[1]*np.ones((1, 0))] + 15*random_factor**2
d_D = np.c_[d[0]*np.ones((1, 1)), d[1]*np.ones((1, 0))] - 0.6*random_factor**2

# CI -> 1 FS neuron
a_CI = np.c_[a[2]*np.ones((1, 1)), a[3]*np.ones((1, 0))] + 0.008*random_factor
b_CI = np.c_[b[2]*np.ones((1, 1)), b[3]*np.ones((1, 0))] - 0.005*random_factor
c_CI = np.c_[c[2]*np.ones((1, 1)), c[3]*np.ones((1, 0))]
d_CI = np.c_[d[2]*np.ones((1, 1)), d[3]*np.ones((1, 0))]

v_D = np.zeros((1, num_steps))
u_D = np.zeros((1, num_steps))

v_CI = np.zeros((1, num_steps))
u_CI = np.zeros((1, num_steps))

v_D[0][0] = v_resting
u_D[0][0] = b_D*v_resting

v_CI[0][0] = v_resting
u_CI[0][0] = b_CI*v_resting

# =============================================================================
# Idc
# =============================================================================
Idc = [3.6, 3.7, 3.9, 0.5, 0.7]

I_D = np.c_[Idc[0]*np.ones((1, 1)), Idc[1]*np.ones((1, 0))]
I_CI = np.c_[Idc[2]*np.ones((1, 1)), Idc[3]*np.ones((1, 0))]
I_TC = np.c_[Idc[4]*np.ones((1, 1)), Idc[4]*np.ones((1, 0))]

# =============================================================================
# MAKING THALAMIC INPUT
# =============================================================================
Thalamus_spikes, I_Thalamus = poisson_spike_generator(
    num_steps = num_steps, 
    dt = dt, 
    num_neurons = n_TC, 
    thalamic_firing_rate = f_thalamus,
    current_value = c_thalamus)

get_frequency(I_Thalamus, sim_time)

plot_raster(title="Thalamus Raster Plot", num_neurons=1, spike_times=Thalamus_spikes, sim_time=sim_time, dt=dt)
plot_voltage(title="Thalamus spikes", y=I_Thalamus[0], dt=dt, sim_time=sim_time)

# =============================================================================
# LAYER D
# =============================================================================

for t in range(1, num_steps):
    v_aux = v_D[0][t - 1]
    u_aux = u_D[0][t - 1]
    
    if(v_aux >= v_threshold):
        v_aux = 1*v_D[0][t]
        v_D[0][t] = 1*c_D[0][0]
        u_D[0][t] = u_aux + d_D[0][0]
    else:
        dvdt = izhikevich_dvdt(v_aux, u_aux, I_D[0][0])
        dudt = izhikevich_dudt(v_aux, u_aux, a_D[0][0], b_D[0][0])
        v_D[0][t] = v_aux + dvdt*dt
        u_D[0][t] = u_aux + dudt*dt
    
plot_voltage(title="Layer D spikes", y=v_D[0], dt=dt, sim_time=sim_time)
    