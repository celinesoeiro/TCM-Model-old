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

f_thalamus = 8                                  # Hz (Thalamus frequency)
c_thalamus = 10                                 # mA (Thalamus input value to the cortex)

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
# Post Synaptic Currents
# =============================================================================
PSC_D = np.zeros((1, num_steps))
PSC_CI = np.zeros((1, num_steps))
PSC_TC = np.zeros((1, num_steps))

# =============================================================================
# DBS
# =============================================================================
connectivity_factor_normal = 2.5 
connectivity_factor_PD = 5
connectivity_factor = connectivity_factor_normal

# =============================================================================
# CONNECTION MATRIX
# =============================================================================
r_D = 0 + 1*np.random.rand(n_D, 1)
r_CI = 0 + 1*np.random.rand(n_CI, 1)

# D to D coupling
aee_d = -1e1/connectivity_factor;            
W_D = aee_d*r_D;

# D to CI Coupling
aei_D_CI = -7.5e3/connectivity_factor;   
W_D_CI = aei_D_CI*r_D;

# D to Thalamus coupling
aee_D_TC = 1e1/connectivity_factor;      
W_D_TC = aee_D_TC*r_D;

# CI to CI coupling
aii_ci = -5e2/connectivity_factor;          
W_CI = aii_ci*r_CI;

# CI to D coupling
aie_CI_D = 2e2/connectivity_factor;     
W_CI_D = aie_CI_D*r_CI;

# CI to Thalamus coupling
aie_CI_TC = 1e1/connectivity_factor;    
W_CI_TC = aie_CI_TC*r_CI;

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
# LAYER D & LAYER CI
# =============================================================================
for t in range(1, num_steps):
    # D
    v_D_aux = v_D[0][t - 1]
    u_D_aux = u_D[0][t - 1]
    
    if(v_D_aux >= v_threshold):
        v_D_aux = 1*v_D[0][t]
        v_D[0][t] = 1*c_D[0][0]
        u_D[0][t] = u_D_aux + d_D[0][0]
    else:
        dvdt_D = izhikevich_dvdt(v_D_aux, u_D_aux, I_D[0][0])
        dudt_D = izhikevich_dudt(v_D_aux, u_D_aux, a_D[0][0], b_D[0][0])
        
        # Self feedback - Inhibitory
        coupling_D = W_D*PSC_D[0][t]
        # Coupling D to CI - Excitatory 
        coupling_CI = W_D_CI*PSC_CI[0][t]
        
        v_D[0][t] = v_D_aux + dt*(dvdt_D + coupling_CI + coupling_D)
        u_D[0][t] = u_D_aux + dudt_D*dt
    
    # CI
    v_CI_aux = v_CI[0][t - 1]
    u_CI_aux = u_CI[0][t - 1]
    
    if(v_CI_aux >= v_threshold):
        v_CI_aux = 1*v_CI[0][t]
        v_CI[0][t] = 1*c_CI[0][0]
        u_CI[0][t] = u_CI_aux + d_CI[0][0]
    else:
        dvdt_CI = izhikevich_dvdt(v_CI_aux, u_CI_aux, I_CI[0][0])
        dudt_CI = izhikevich_dudt(v_CI_aux, u_CI_aux, a_CI[0][0], b_CI[0][0])
        
        # Self feeback - Inhibitory
        coupling_CI = W_CI*PSC_CI[0][t]
        # Coupling CI to D - Inhibitory
        coupling_D = W_CI_D*PSC_D[0][t]
        
        v_CI[0][t] = v_CI_aux + dt*(dvdt_CI + coupling_CI + coupling_D)
        u_CI[0][t] = u_CI_aux + dudt_CI*dt
    
plot_voltage(title="Layer D spikes", y=v_D[0], dt=dt, sim_time=sim_time)
plot_voltage(title="Layer CI spikes", y=v_CI[0], dt=dt, sim_time=sim_time)
