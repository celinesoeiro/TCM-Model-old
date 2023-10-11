#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 09:18:57 2023

@author: celinesoeiro
"""

"""
Created on Wed Sep 27 08:14:59 2023

@author: celinesoeiro
"""

import random
import numpy as np
import pandas as pd

random.seed(0)
random_factor = np.round(random.random(),2)

from cortex_functions import poisson_spike_generator, izhikevich_dvdt, izhikevich_dudt, plot_raster, plot_voltage, get_frequency, tm_synapse_eq
from cortex_functions import plot_heat_map

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================
ms = 1000                                       # 1 second = 1000 milliseconds
sim_time = 3                                    # seconds
dt = 1/ms                                       # seconds
fs = int(1/dt)                                       # Hz (Sampling Frequency)

# Voltage parameters
v_threshold = 30
v_resting = -65

# Calculate the number of time steps
num_steps = int(sim_time / dt)

n_TC = 100
n_TR = 40
total_neurons = n_TC + n_TR

# Distribution of neurons in each structure
n_TC_1 = int(0.7*n_TC)      # TC neurons
n_TC_2 = int(0.3*n_TC)      # TC neurons
n_TR_1 = int(0.5*n_TR)      # TR neurons
n_TR_2 = int(0.5*n_TR)      # TR neurons

spikes_TC = np.zeros((n_TC, num_steps))
spikes_TR = np.zeros((n_TR, num_steps))

f_thalamus = 8                                  # Hz (Thalamus frequency)

# =============================================================================
# Izhikevich neuron parameters
# =============================================================================
#    0-RS  1-IB  2-FS 3-LTS 4-TC  5-TR 
a = [0.02, 0.02, 0.1, 0.02, 0.02, 0.02]
b = [0.2,  0.2,  0.2, 0.25, 0.25, 0.25]
c = [-65,  -55,  -65, -65,   -65,  -65]
d = [8,    4,      2,   2,  0.05, 2.05]

# TC -> TC neuron
a_TC = np.c_[a[4]*np.ones((1, n_TC_1)), a[4]*np.ones((1, n_TC_2))]
b_TC = np.c_[b[4]*np.ones((1, n_TC_1)), b[4]*np.ones((1, n_TC_2))]
c_TC = np.c_[c[4]*np.ones((1, n_TC_1)), c[4]*np.ones((1, n_TC_2))] + 15*random_factor**2
d_TC = np.c_[d[4]*np.ones((1, n_TC_1)), d[4]*np.ones((1, n_TC_2))] - 0.6*random_factor**2

# TR -> TR neuron
a_TR = np.c_[a[2]*np.ones((1, n_TR_1)), a[5]*np.ones((1, n_TR_2))] + 0.008*random_factor
b_TR = np.c_[b[2]*np.ones((1, n_TR_1)), b[5]*np.ones((1, n_TR_2))] - 0.005*random_factor
c_TR = np.c_[c[2]*np.ones((1, n_TR_1)), c[5]*np.ones((1, n_TR_2))]
d_TR = np.c_[d[2]*np.ones((1, n_TR_1)), d[5]*np.ones((1, n_TR_2))]

v_TC = np.zeros((n_TC, num_steps))
u_TC = np.zeros((n_TC, num_steps))

v_TR = np.zeros((n_TR, num_steps))
u_TR = np.zeros((n_TR, num_steps))

v_TC[:, 0] = v_resting
u_TC[:, 0] = b_TC*v_resting

v_TR[:, 0] = v_resting
u_TR[:, 0] = b_TR*v_resting

# =============================================================================
# Idc
# =============================================================================
Idc = [3.6, 3.7, 3.9, 0.5, 0.7]
Idc = [value * 100 for value in Idc]

I_TC = np.c_[Idc[4]*np.ones((1, n_TC_1)), Idc[4]*np.ones((1, n_TC_2))]
I_TR = np.c_[Idc[4]*np.ones((1, n_TR_1)), Idc[4]*np.ones((1, n_TR_2))]

# =============================================================================
# Post Synaptic Currents
# =============================================================================
PSC_TC = np.zeros((1, num_steps))
PSC_TR = np.zeros((1, num_steps))
PSC_TC_Cortex = np.zeros((1, num_steps))
PSC_D_Thalamus = np.zeros((1, num_steps))

# =============================================================================
# DBS
# =============================================================================
connectivity_factor_normal = 2.5 
connectivity_factor_PD = 5
connectivity_factor = connectivity_factor_normal

# =============================================================================
# CONNECTION MATRIX
# =============================================================================
r_TC = 0 + 1*np.random.rand(n_TC, 1)
r_TR = 0 + 1*np.random.rand(n_TR, 1)

# TC COUPLINGS
## TC to TC
aee_tc = 0/connectivity_factor;          W_TC_TC = aee_tc*r_TC;
## TC to TR
aei_tctr = -5e2/connectivity_factor;     W_TC_TR = aei_tctr*r_TC;

# TR COUPLINGS
## TR to TR
aii_tr = -5e1/connectivity_factor;       W_TR_TR = aii_tr*r_TR;
## TR to TC
aie_trtc = 1e3/connectivity_factor;      W_TR_TC = aie_trtc*r_TR;

# Initialize matrix (2 structures -> 2x2 matrix)
matrix = np.zeros((2,2))

matrix[0][0] = np.mean(W_TC_TC)
matrix[0][1] = np.mean(W_TC_TR)
matrix[1][0] = np.mean(W_TR_TC)
matrix[1][1] = np.mean(W_TR_TR)

# normalizing Normal coupling matrix
matrix_norm = matrix/np.linalg.norm(matrix)

print("-- Printing the coupling matrixes")

CM_Normal = pd.DataFrame(matrix_norm, columns=['TC', 'TR'])
plot_heat_map(matrix_normal = CM_Normal, labels=['TC', 'TR'])

# =============================================================================
#     Noise terms
# =============================================================================
white_gaussian_add = 1.5; cn = 1 # additive white Gaussian noise strength
white_gaussian_thr = 0.5 # threshold white Gaussian noise strength

random_TR = np.random.randn(n_TR, fs)
random_TC = np.random.randn(n_TC, fs)

random_TR_diff = np.random.randn(n_TR, num_steps - fs)
random_TC_diff = np.random.randn(n_TC, num_steps - fs)

zeta_TR = white_gaussian_thr*np.c_[ random_TR, cn*random_TR_diff ]
zeta_TC = white_gaussian_thr*np.c_[ random_TC, cn*random_TC_diff ]

kisi_TC = white_gaussian_add*np.c_[ random_TC, cn*random_TC_diff ]
kisi_TR = white_gaussian_add*np.c_[ random_TR, cn*random_TR_diff ]

# =============================================================================
# TM synapse
# =============================================================================
r_TC = np.zeros((1, 3))
x_TC = np.zeros((1, 3))
Is_TC = np.zeros((1, 3))

r_TR = np.zeros((1, 3))
x_TR = np.zeros((1, 3))
Is_TR = np.zeros((1, 3))

t_f_E = [670, 17, 326]
t_d_E = [138, 671, 329]
U_E = [0.09, 0.5, 0.29]
A_E = [0.2, 0.63, 0.17]
A_T_D_E = [0, 1, 0] # Depressing (From Thalamus to Layer D)
A_D_T_E = [1, 0, 0] # Facilitating (From Layer D to Thalamus)
t_s_E = 3

t_f_I = [376, 21, 62]
t_d_I = [45, 706, 144]
U_I = [0.016, 0.25, 0.32]
A_I = [0.08, 0.75, 0.17]
t_s_I = 11

Ipost_TC = np.zeros((1,n_TC))
Ipost_TR = np.zeros((1,n_TR))

# =============================================================================
# MAKING THALAMIC INPUT
# =============================================================================
# TC_spikes, I_TC = poisson_spike_generator(
#     num_steps = num_steps, 
#     dt = dt, 
#     num_neurons = 1, 
#     thalamic_firing_rate = f_thalamus,
#     current_value = Idc[4])

# get_frequency(I_TC, sim_time)

# plot_raster(title="TC Raster Plot", num_neurons=1, spike_times=TC_spikes, sim_time=sim_time, dt=dt)
# plot_voltage(title="TC spikes", y=I_TC[0], dt=dt, sim_time=sim_time)

# TR_spikes, I_TR = poisson_spike_generator(
#     num_steps = num_steps, 
#     dt = dt, 
#     num_neurons = 1, 
#     thalamic_firing_rate = f_thalamus,
#     current_value = Idc[4])

# get_frequency(I_TR, sim_time)

# plot_raster(title="TR Raster Plot", num_neurons=1, spike_times=TR_spikes, sim_time=sim_time, dt=dt)
# plot_voltage(title="TR spikes", y=I_TR[0], dt=dt, sim_time=sim_time)

# =============================================================================
# THALAMIC NUCLEUS
# =============================================================================
for t in range(1, num_steps):
# =============================================================================
# TC
# =============================================================================
    for tc in range(n_TC):
        v_TC_aux = v_TC[tc][t - 1]
        u_TC_aux = u_TC[tc][t - 1]
        AP_TC = 0
        
        if(v_TC_aux >= v_threshold + zeta_TC[tc][t - 1]):
            v_TC_aux = 1*v_TC[tc][t]
            v_TC[tc][t] = 1*c_TC[0][tc]
            u_TC[tc][t] = 1*u_TC_aux + d_TC[0][tc]
            AP_TC = 1
            spikes_TC[tc][t] = t
        else:
            dvdt_TC = izhikevich_dvdt(v_TC_aux, u_TC_aux, I_TC[0][tc])
            dudt_TC = izhikevich_dudt(v_TC_aux, u_TC_aux, a_TC[0][tc], b_TC[0][tc])
            
            # Self feeback - Inhibitory
            coupling_TC_TC = W_TC_TC[tc][0]*PSC_TC[0][t]
            # Coupling TC to TR - Inhibitory
            coupling_TC_TR = W_TC_TR[tc][0]*PSC_TR[0][t]
            
            v_TC[tc][t] = v_TC_aux + dt*(dvdt_TC + coupling_TC_TC + coupling_TC_TR)
            u_TC[tc][t] = u_TC_aux + dudt_TC*dt
            
        # Synaptic connection - Within Thalamus
        syn_TC = tm_synapse_eq(r=r_TC, x=x_TC, Is=Is_TC, AP=AP_TC, tau_f=t_f_E, tau_d=t_d_E, tau_s=t_s_E, U=U_E, A=A_E, dt=dt)
        r_TC = syn_TC['r']
        x_TC = syn_TC['x']
        Is_TC = syn_TC['Is']
        Ipost_TC = syn_TC['Ipost']
        
        # Synaptic connection - Thalamus to Cortex
        syn_TC_Cortex = tm_synapse_eq(r=r_TC, x=x_TC, Is=Is_TC, AP=AP_TC, tau_f=t_f_E, tau_d=t_d_E, tau_s=t_s_E, U=U_E, A=A_E, dt=dt)
        r_TC_Cortex = syn_TC_Cortex['r']
        x_TC_Cortex = syn_TC_Cortex['x']
        Is_TC_Cortex = syn_TC_Cortex['Is']
        Ipost_TC_Cortex = syn_TC_Cortex['Ipost']
        
    PSC_TC[0][t] = np.sum(Ipost_TC)
    PSC_TC_Cortex[0][t] = np.sum(Ipost_TC_Cortex)
# =============================================================================
# TR
# =============================================================================
    for tr in range(n_TR):
        v_TR_aux = v_TR[tr][t - 1]
        u_TR_aux = u_TR[tr][t - 1]
        AP_TR = 0
        
        if (v_TR_aux >= v_threshold + zeta_TR[tr][t - 1]):
            v_TR_aux = 1*v_TR[tr][t]
            v_TR[tr][t] = 1*c_TR[0][tr]
            u_TR[tr][t] = 1*u_TR_aux + d_TR[0][tr]
            AP_TR = 1
            spikes_TR[tr][t] = t
        else:
            dvdt_TR = izhikevich_dvdt(v_TR_aux, u_TR_aux, I_TR[0][tr])
            dudt_TR = izhikevich_dudt(v_TR_aux, u_TR_aux, a_TR[0][tr], b_TR[0][tr])
            
            # Self feeback - Inhibitory
            coupling_TR_TR = W_TR_TR[tr][0]*PSC_TR[0][t]
            # Coupling TR to TC - Inhibitory
            coupling_TR_TC = W_TR_TC[tr][0]*PSC_TC[0][t]
            
            v_TC[tr][t] = v_TR_aux + dt*(dvdt_TR + coupling_TR_TC + coupling_TR_TR)
            u_TC[tr][t] = u_TR_aux + dudt_TR*dt
            
        # Synaptic connection - Within Thalamus
        syn_TR = tm_synapse_eq(r=r_TR, x=x_TR, Is=Is_TR, AP=AP_TR, tau_f=t_f_I, tau_d=t_d_I, tau_s=t_s_I, U=U_I, A=A_I, dt=dt)
        r_TR = syn_TR['r']
        x_TR = syn_TR['x']
        Is_TR = syn_TR['Is']
        Ipost_TR = syn_TR['Ipost']
            
    PSC_TR[0][t] = np.sum(Ipost_TR)

plot_voltage(title="Layer TC spikes", y=v_TC[0], dt=dt, sim_time=sim_time)
plot_voltage(title="Layer TR spikes", y=v_TR[0], dt=dt, sim_time=sim_time)

plot_voltage(title="LFP Layer TC", y=PSC_TC[0], dt=dt, sim_time=sim_time)
plot_voltage(title="LFP Layer TR", y=PSC_TR[0], dt=dt, sim_time=sim_time)


