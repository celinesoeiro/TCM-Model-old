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
from cortex_functions import plot_heat_map, plot_raster_cortex

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================
ms = 1000                                       # 1 second = 1000 milliseconds
sim_time = 3                                    # seconds
dt = 1/ms                                       # seconds
fs = int(1/dt)                                  # Hz (Sampling Frequency)
chop_till = int(0.1/dt)

# Voltage parameters
v_threshold = 30
v_resting = -65

# Calculate the number of time steps
num_steps = int(sim_time / dt)

n_S = 10
n_M = 10
n_D = 10
n_CI = 10
n_TC = 10
n_TR = 4
total_neurons = n_S + n_M + n_D + n_CI + n_TC + n_TR

# Distribution of neurons in each structure
n_S_1 = int(0.5*n_S)        # RS neurons
n_S_2 = int(0.5*n_S)        # IB neurons
n_M_1 = int(0.5*n_M)        # RS neurons
n_M_2 = int(0.5*n_M)        # IB neurons
n_D_1 = int(0.7*n_D)        # RS neurons
n_D_2 = int(0.3*n_D)        # IB neurons
n_CI_1 = int(0.5*n_CI)      # FS neurons
n_CI_2 = int(0.5*n_CI)      # LTS neurons
n_TR_1 = int(0.5*n_TR)      # TR neurons
n_TR_2 = int(0.5*n_TR)      # TR neurons
n_TC_1 = int(0.7*n_TC)      # TC neurons
n_TC_2 = int(0.3*n_TC)      # TC neurons

if (n_S%2 != 0 or n_M%2 != 0 or n_D%2 != 0 or n_CI%2 != 0 or n_TC%2 != 0 or n_TR%2 != 0):
    # Distribution of neurons in each structure
    n_S_1 = int(1*n_S)        # RS neurons
    n_S_2 = int(0*n_S)        # IB neurons
    n_M_1 = int(1*n_M)        # RS neurons
    n_M_2 = int(0*n_M)        # IB neurons
    n_D_1 = int(1*n_D)        # RS neurons
    n_D_2 = int(0*n_D)        # IB neurons
    n_CI_1 = int(1*n_CI)      # FS neurons
    n_CI_2 = int(0*n_CI)      # LTS neurons
    n_TR_1 = int(1*n_TR)      # TR neurons
    n_TR_2 = int(0*n_TR)      # TR neurons
    n_TC_1 = int(1*n_TC)      # TC neurons
    n_TC_2 = int(0*n_TC)      # TC neurons

spikes_S = np.zeros((n_S, num_steps))
spikes_M = np.zeros((n_M, num_steps))
spikes_D = np.zeros((n_D, num_steps))
spikes_CI = np.zeros((n_CI, num_steps))
spikes_TC = np.zeros((n_TC, num_steps))
spikes_TR = np.zeros((n_TR, num_steps))

f_thalamus = 8                # Hz (Thalamus frequency)

# =============================================================================
# Izhikevich neuron parameters
# =============================================================================
#    0-RS  1-IB  2-FS 3-LTS 4-TC  5-TR 
a = [0.02, 0.02, 0.1, 0.02, 0.02, 0.02]
b = [0.2,  0.2,  0.2, 0.25, 0.25, 0.25]
c = [-65,  -55,  -65, -65,   -65,  -65]
d = [8,    4,      2,   2,  0.05, 2.05]

# S -> RS and IB neuron
a_S = np.c_[a[0]*np.ones((1, n_S_1)), a[1]*np.ones((1, n_S_2))]
b_S = np.c_[b[0]*np.ones((1, n_S_1)), b[1]*np.ones((1, n_S_2))]
c_S = np.c_[c[0]*np.ones((1, n_S_1)), c[1]*np.ones((1, n_S_2))] + 15*random_factor**2
d_S = np.c_[d[0]*np.ones((1, n_S_1)), d[1]*np.ones((1, n_S_2))] - 0.6*random_factor**2

# M -> RS neuron
a_M = np.c_[a[0]*np.ones((1, n_M_1)), a[1]*np.ones((1, n_M_2))]
b_M = np.c_[b[0]*np.ones((1, n_M_1)), b[1]*np.ones((1, n_M_2))]
c_M = np.c_[c[0]*np.ones((1, n_M_1)), c[1]*np.ones((1, n_M_2))] + 15*random_factor**2
d_M = np.c_[d[0]*np.ones((1, n_M_1)), d[1]*np.ones((1, n_M_2))] - 0.6*random_factor**2

# D -> RS and IB neuron
a_D = np.c_[a[0]*np.ones((1, n_D_1)), a[1]*np.ones((1, n_D_2))]
b_D = np.c_[b[0]*np.ones((1, n_D_1)), b[1]*np.ones((1, n_D_2))]
c_D = np.c_[c[0]*np.ones((1, n_D_1)), c[1]*np.ones((1, n_D_2))] + 15*random_factor**2
d_D = np.c_[d[0]*np.ones((1, n_D_1)), d[1]*np.ones((1, n_D_2))] - 0.6*random_factor**2

# CI -> FS and LTS neuron
a_CI = np.c_[a[2]*np.ones((1, n_CI_1)), a[3]*np.ones((1, n_CI_2))] + 0.008*random_factor
b_CI = np.c_[b[2]*np.ones((1, n_CI_1)), b[3]*np.ones((1, n_CI_2))] - 0.005*random_factor
c_CI = np.c_[c[2]*np.ones((1, n_CI_1)), c[3]*np.ones((1, n_CI_2))]
d_CI = np.c_[d[2]*np.ones((1, n_CI_1)), d[3]*np.ones((1, n_CI_2))]

v_S = np.zeros((n_S, num_steps))
u_S = np.zeros((n_S, num_steps))

v_M = np.zeros((n_M, num_steps))
u_M = np.zeros((n_M, num_steps))

v_D = np.zeros((n_D, num_steps))
u_D = np.zeros((n_D, num_steps))

v_CI = np.zeros((n_CI, num_steps))
u_CI = np.zeros((n_CI, num_steps))

v_CI = np.zeros((n_CI, num_steps))
u_CI = np.zeros((n_CI, num_steps))

v_S[:, 0] = v_resting
u_S[:, 0] = b_S*v_resting

v_M[:, 0] = v_resting
u_M[:, 0] = b_M*v_resting

v_D[:, 0] = v_resting
u_D[:, 0] = b_D*v_resting

v_CI[:, 0] = v_resting
u_CI[:, 0] = b_CI*v_resting

# =============================================================================
# Idc
# =============================================================================
Idc = [3.6, 3.7, 3.9, 0.5, 0.7]
Idc = [value * 4 for value in Idc]

I_S = np.c_[Idc[0]*np.ones((1, n_S_1)), Idc[1]*np.ones((1, n_S_2))]
I_M = np.c_[Idc[0]*np.ones((1, n_M_1)), Idc[0]*np.ones((1, n_M_2))]
I_D = np.c_[Idc[0]*np.ones((1, n_D_1)), Idc[1]*np.ones((1, n_D_2))]
I_CI = np.c_[Idc[2]*np.ones((1, n_CI_1)), Idc[3]*np.ones((1, n_CI_2))]

# =============================================================================
# Post Synaptic Currents
# =============================================================================
PSC_S = np.zeros((1, num_steps))
PSC_M = np.zeros((1, num_steps))
PSC_D = np.zeros((1, num_steps))
PSC_CI = np.zeros((1, num_steps))
PSC_TC = np.zeros((1, num_steps))
PSC_TR = np.zeros((1, num_steps))

# =============================================================================
# DBS
# =============================================================================
connectivity_factor_normal = 2.5 
connectivity_factor_PD = 5
connectivity_factor = connectivity_factor_normal

# =============================================================================
# CONNECTION MATRIX
# =============================================================================
r_S = 0 + 1*np.random.rand(n_S, 1)
r_M = 0 + 1*np.random.rand(n_M, 1)
r_D = 0 + 1*np.random.rand(n_D, 1)
r_CI = 0 + 1*np.random.rand(n_CI, 1)
r_TC = 0 + 1*np.random.rand(n_TC, 1)
r_TR = 0 + 1*np.random.rand(n_TR, 1)

# S COUPLINGS
## S to S
aee_S = -1e1/connectivity_factor;        W_S = aee_S*r_S;
## S to M
aee_S_M = 1e1/connectivity_factor;       W_S_M = aee_S_M*r_S;
## S to D
aee_S_D = 5e2/connectivity_factor;       W_S_D = aee_S_D*r_S;
## S to CI
aei_S_CI = -5e2/connectivity_factor;     W_S_CI = aei_S_CI*r_S;
## S to TC
aee_S_TC = 0/connectivity_factor;        W_S_TC = aee_S_TC*r_S;    
## S to TR 
aei_S_TR = 0/connectivity_factor;        W_S_TR = aei_S_TR*r_S;

# M COUPLINGS
## M to M
aee_M = -1e1/connectivity_factor;        W_M = aee_M*r_M;
## M to S
aee_M_S = 3e2/connectivity_factor;       W_M_S = aee_M_S*r_M; 
## M to D
aee_M_D = 0/connectivity_factor;         W_M_D = aee_M_D*r_M;     
## M to CI
aei_M_CI = -3e2/connectivity_factor;     W_M_CI = aei_M_CI*r_M;
## M to TC
aee_M_TC = 0/connectivity_factor;        W_M_TC = aee_M_TC*r_M;
## M to TR
aei_M_TR = 0/connectivity_factor;        W_M_TR = aei_M_TR*r_M;

# D COUPLINGS
## D to D 
aee_D = -1e1/connectivity_factor;        W_D = aee_D*r_D;
## D to S
aee_D_S = 3e2/connectivity_factor;       W_D_S = aee_D_S*r_D;
## D to M
aee_D_M = 0/connectivity_factor;         W_D_M = aee_D_M*r_D;
# D to CI
aei_D_CI = -7.5e3/connectivity_factor;   W_D_CI = aei_D_CI*r_D;
# D to TC 
aee_D_TC = 1e1/connectivity_factor;      W_D_TC = aee_D_TC*r_D;
# D to TR
aei_D_TR = 0/connectivity_factor;        W_D_TR = aei_D_TR*r_D;

# CI COUPLINGS
## CI to CI
aii_CI = -5e2/connectivity_factor;       W_CI = aii_CI*r_CI;
## CI to S
aie_CI_S = 2e2/connectivity_factor;      W_CI_S = aie_CI_S*r_CI;
## CI to M
aie_CI_M = 2e2/connectivity_factor;      W_CI_M = aie_CI_M*r_CI;
## CI to D
aie_CI_D = 2e2/connectivity_factor;      W_CI_D = aie_CI_D*r_CI;
## CI to TC
aie_CI_TC = 1e1/connectivity_factor;     W_CI_TC = aie_CI_TC*r_CI;
## CI to TR
aii_CI_TR = 0/connectivity_factor;       W_CI_TR = aii_CI_TR*r_CI;

# TC COUPLINGS
## TC to TC
aee_tc = 0/connectivity_factor;          W_TC_TC = aee_tc*r_TC;
## TC to S
aee_tcs = 0/connectivity_factor;         W_TC_S = aee_tcs*r_TC;   
## TC to M
aee_tcm = 0/connectivity_factor;         W_TC_M = aee_tcm*r_TC;
## TC to D
aee_tcd = 7e2/connectivity_factor;       W_TC_D = aee_tcd*r_TC;
## TC to CI
aei_tcci = 0/connectivity_factor;        W_TC_CI = aei_tcci*r_TC;
## TC to TR
aei_tctr = -5e2/connectivity_factor;     W_TC_TR = aei_tctr*r_TC;

# TR COUPLINGS
## TR to TR
aii_tr = -5e1/connectivity_factor;       W_TR_TR = aii_tr*r_TR;
## TR to S
aie_trs = 0/connectivity_factor;         W_TR_S = aie_trs*r_TR;
## TR to M
aie_trm = 0/connectivity_factor;         W_TR_M = aie_trm*r_TR;
## TR to D
aie_trd = 7e2/connectivity_factor;       W_TR_D = aie_trd*r_TR;
## TR to CI
aii_trci = 0/connectivity_factor;        W_TR_CI = aii_trci*r_TR;
## TR to TC
aie_trtc = 1e3/connectivity_factor;      W_TR_TC = aie_trtc*r_TR;

# Initialize matrix (6 structures -> 6x6 matrix)
matrix = np.zeros((6,6))
# Main Diagonal
matrix[0][0] = np.mean(W_S)
matrix[1][1] = np.mean(W_M)
matrix[2][2] = np.mean(W_D)
matrix[3][3] = np.mean(W_CI)
matrix[4][4] = np.mean(W_TC_TC)
matrix[5][5] = np.mean(W_TR_TR)
# First column - Layer S
matrix[1][0] = np.mean(W_S_M)
matrix[2][0] = np.mean(W_S_D)
matrix[3][0] = np.mean(W_S_CI)
matrix[4][0] = np.mean(W_S_TC)
matrix[5][0] = np.mean(W_S_TR)
# Second column - Layer M
matrix[0][1] = np.mean(W_M_S)
matrix[2][1] = np.mean(W_M_D)
matrix[3][1] = np.mean(W_M_CI)
matrix[4][1] = np.mean(W_M_TC)
matrix[5][1] = np.mean(W_M_TR)
# Thid column - Layer D
matrix[0][2] = np.mean(W_D_S)
matrix[1][2] = np.mean(W_D_M)
matrix[3][2] = np.mean(W_D_CI)
matrix[4][2] = np.mean(W_D_TC)
matrix[5][2] = np.mean(W_D_TR)
# Fourth column - Structure CI
matrix[0][3] = np.mean(W_CI_S)
matrix[1][3] = np.mean(W_CI_M)
matrix[2][3] = np.mean(W_CI_D)
matrix[4][3] = np.mean(W_CI_TC)
matrix[5][3] = np.mean(W_CI_TR)
# Fifth column - Structure TC
matrix[0][4] = np.mean(W_TC_S)
matrix[1][4] = np.mean(W_TC_M)
matrix[2][4] = np.mean(W_TC_D)
matrix[3][4] = np.mean(W_TC_CI)
matrix[5][4] = np.mean(W_TC_TR)
# Sixth column - Structure TR
matrix[0][5] = np.mean(W_TR_S)
matrix[1][5] = np.mean(W_TR_M)
matrix[2][5] = np.mean(W_TR_D)
matrix[3][5] = np.mean(W_TR_CI)
matrix[4][5] = np.mean(W_TR_TC)

# normalizing Normal coupling matrix
matrix_norm = matrix/np.linalg.norm(matrix)

print("-- Printing the coupling matrixes")

CM_Normal = pd.DataFrame(matrix_norm, columns=['S', 'M', 'D', 'CI', 'TC', 'TR'])
plot_heat_map(matrix_normal = CM_Normal, labels=['S', 'M', 'D', 'CI', 'TC', 'TR'])

# =============================================================================
#     Noise terms
# =============================================================================
white_gaussian_add = 1.5; cn = 1 # additive white Gaussian noise strength
white_gaussian_thr = 0.5 # threshold white Gaussian noise strength

random_S = np.random.randn(n_S, fs)
random_M = np.random.randn(n_M, fs)
random_D = np.random.randn(n_D, fs)
random_CI = np.random.randn(n_CI, fs)
random_TR = np.random.randn(n_TR, fs)
random_TC = np.random.randn(n_TC, fs)

random_S_diff = np.random.randn(n_S, num_steps - fs)
random_M_diff = np.random.randn(n_M, num_steps - fs)
random_D_diff = np.random.randn(n_D, num_steps - fs)
random_CI_diff = np.random.randn(n_CI, num_steps - fs)
random_TR_diff = np.random.randn(n_TR, num_steps - fs)
random_TC_diff = np.random.randn(n_TC, num_steps - fs)

zeta_S = white_gaussian_thr*np.c_[ random_S, cn*random_S_diff ]
zeta_M = white_gaussian_thr*np.c_[ random_M, cn*random_M_diff ]    
zeta_D = white_gaussian_thr*np.c_[ random_D, cn*random_D_diff ]
zeta_CI = white_gaussian_thr*np.c_[ random_CI, cn*random_CI_diff ]
zeta_TR = white_gaussian_thr*np.c_[ random_TR, cn*random_TR_diff ]
zeta_TC = white_gaussian_thr*np.c_[ random_TC, cn*random_TC_diff ]

kisi_S = white_gaussian_add*np.c_[ random_S, cn*random_S_diff ]
kisi_M = white_gaussian_add*np.c_[ random_M, cn*random_M_diff ]    
kisi_D = white_gaussian_add*np.c_[ random_D, cn*random_D_diff ]
kisi_CI = white_gaussian_add*np.c_[ random_CI, cn*random_CI_diff ]
kisi_TC = white_gaussian_add*np.c_[ random_TC, cn*random_TC_diff ]
kisi_TR = white_gaussian_add*np.c_[ random_TR, cn*random_TR_diff ]

# =============================================================================
# TM synapse
# =============================================================================
r_S = np.zeros((1, 3)); x_S = np.zeros((1, 3)); Is_S = np.zeros((1, 3));

r_M = np.zeros((1, 3)); x_M = np.zeros((1, 3)); Is_M = np.zeros((1, 3));

r_D = np.zeros((1, 3)); x_D = np.zeros((1, 3)); Is_D = np.zeros((1, 3));

r_CI = np.zeros((1, 3)); x_CI = np.zeros((1, 3)); Is_CI = np.zeros((1, 3));

r_TC = np.zeros((1, 3)); x_TC = np.zeros((1, 3)); Is_TC = np.zeros((1, 3));

r_TR = np.zeros((1, 3)); x_TR = np.zeros((1, 3)); Is_TR = np.zeros((1, 3));

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

Ipost_S = np.zeros((1, n_S))
Ipost_M = np.zeros((1, n_M))
Ipost_D = np.zeros((1, n_D))
Ipost_CI = np.zeros((1,n_CI))
Ipost_TC = np.zeros((1,n_TC))
Ipost_TR = np.zeros((1,n_TR))

# =============================================================================
# MAKING THALAMIC INPUT
# =============================================================================
TC_spikes, PSC_TC = poisson_spike_generator(
    num_steps = num_steps, 
    dt = dt, 
    num_neurons = n_TC, 
    thalamic_firing_rate = f_thalamus,
    )

get_frequency(PSC_TC[0], sim_time)

plot_raster(title="TC Raster Plot", num_neurons=n_TC, spike_times=TC_spikes, sim_time=sim_time, dt=dt)
plot_voltage(title="TC spikes", y=PSC_TC[0], dt=dt, sim_time=sim_time)

TR_spikes, PSC_TR = poisson_spike_generator(
    num_steps = num_steps, 
    dt = dt, 
    num_neurons = n_TR, 
    thalamic_firing_rate = f_thalamus,
    )

get_frequency(PSC_TR[0], sim_time)

plot_raster(title="TR Raster Plot", num_neurons=n_TR, spike_times=TR_spikes, sim_time=sim_time, dt=dt)
plot_voltage(title="TR spikes", y=PSC_TR[0], dt=dt, sim_time=sim_time)

# =============================================================================
# LAYER D & LAYER CI & LAYER S
# =============================================================================
for t in range(1, num_steps):    

# # =============================================================================
# #     # S
# # =============================================================================
#     for s in range(n_S):
#         v_S_aux = v_S[s][t - 1]
#         u_S_aux = u_S[s][t - 1]
#         AP_S = 0
        
#         if(v_S_aux >= v_threshold + zeta_S[s][t - 1]):
#             v_S_aux = 1*v_S[s][t]
#             v_S[s][t] = 1*c_S[0][s]
#             u_S[s][t] = u_S_aux + d_S[0][s]
#             AP_S = 1
#             spikes_S[s][t] = t
#         else:
#             dvdt_S = izhikevich_dvdt(v_S_aux, u_S_aux, I_S[0][s])
#             dudt_S = izhikevich_dudt(v_S_aux, u_S_aux, a_S[0][s], b_S[0][s])
            
#             # Self feedback - Inhibitory
#             coupling_S_S = W_S[s][0]*PSC_S[0][t]
#             # Coupling S to M - Excitatory
#             coupling_S_M = W_S_M[s][0]*PSC_M[0][t]
#             # Coupling S to D - Excitatory
#             coupling_S_D = W_S_D[s][0]*PSC_D[0][t]
#             # Coupling S to CI - Excitatory 
#             coupling_S_CI = W_S_CI[s][0]*PSC_CI[0][t]
#             # Coupling S to TC - Excitatory
#             coupling_S_TC = W_S_TC[s][0]*PSC_TC[0][t]
#             # Coupling S to TR - Inhibitory
#             coupling_S_TR = W_S_TR[s][0]*PSC_TR[0][t]
            
#             v_S[s][t] = v_S_aux + dt*(dvdt_S + coupling_S_S + coupling_S_M + coupling_S_D + coupling_S_CI + coupling_S_TC + coupling_S_TR)
#             u_S[s][t] = u_S_aux + dudt_S*dt
            
#         # Synaptic connection - within cortex
#         syn_S = tm_synapse_eq(r=r_S, x=x_S, Is=Is_S, AP=AP_S, tau_f=t_f_E, tau_d=t_f_E, tau_s=t_s_E, U=U_E, A=A_E, dt=dt)
#         r_S = syn_S['r']
#         x_S = syn_S['x']
#         Is_S = syn_S['Is']
#         Ipost_S = syn_S['Ipost']
            
#     PSC_S[0][t] = np.sum(Ipost_S)
# =============================================================================
#     # D
# =============================================================================
    for d in range(n_D):
        v_D_aux = 1*v_D[d][t - 1]
        u_D_aux = 1*u_D[d][t - 1]
        AP_D = 0
        
        # Izhikevich neuron model
        dvdt_D = izhikevich_dvdt(v_D_aux, u_D_aux, I_D[0][d])
        dudt_D = izhikevich_dudt(v_D_aux, u_D_aux, a_D[0][d], b_D[0][d])
        # Self feedback - Inhibitory
        coupling_D_D = W_D[d][0]*PSC_D[0][t]
        # Coupling D to S - Excitatory
        coupling_D_S = W_D_S[d][0]*PSC_S[0][t]
        # Coupling D to M - Excitatory
        coupling_D_M = W_D_M[d][0]*PSC_M[0][t]
        # Coupling D to CI - Excitatory 
        coupling_D_CI = W_D_CI[d][0]*PSC_CI[0][t]
        # Coupling D to TC - Excitatory
        coupling_D_TC = W_D_TC[d][0]*PSC_TC[0][t]
        # Coupling D to TR - Inhibitory
        coupling_D_TR = W_D_TR[d][0]*PSC_TR[0][t]
        
        if(v_D_aux >= v_threshold + zeta_D[d][t - 1]):
            v_D_aux = 1*v_D[d][t]
            v_D[d][t] = 1*c_D[0][d]
            u_D[d][t] = 1*(u_D_aux + d_D[0][d])
            AP_D = 1
            spikes_D[d][t] = t
            
        v_D[d][t] = v_D_aux + dt*(dvdt_D + coupling_D_S + coupling_D_M + coupling_D_D + coupling_D_CI + coupling_D_TC + coupling_D_TR)
        u_D[d][t] = u_D_aux + dudt_D*dt
            
        # Synaptic connection - Within Cortex
        syn_D = tm_synapse_eq(r=r_D, x=x_D, Is=Is_D, AP=AP_D, tau_f=t_f_E, tau_d=t_f_E, tau_s=t_s_E, U=U_E, A=A_E, dt=dt)
        r_D = syn_D['r']
        x_D = syn_D['x']
        Is_D = syn_D['Is']
        Ipost_D = syn_D['Ipost']
        
        syn_D_Thalamus = tm_synapse_eq(r=r_D, x=x_D, Is=Is_D, AP=AP_D, tau_f=t_f_E, tau_d=t_f_E, tau_s=t_s_E, U=U_E, A=A_E, dt=dt)
        r_D = syn_D['r']
        x_D = syn_D['x']
        Is_D = syn_D['Is']
        Ipost_D = syn_D['Ipost']
        
        syn_D_ = tm_synapse_eq(r=r_D, x=x_D, Is=Is_D, AP=AP_D, tau_f=t_f_E, tau_d=t_f_E, tau_s=t_s_E, U=U_E, A=A_E, dt=dt)
        r_D = syn_D['r']
        x_D = syn_D['x']
        Is_D = syn_D['Is']
        Ipost_D = syn_D['Ipost']
            
    PSC_D[0][t] = np.sum(Ipost_D)
    
# =============================================================================
#     # CI
# =============================================================================
    for ci in range(n_CI):
        v_CI_aux = 1*v_CI[ci][t - 1]
        u_CI_aux = 1*u_CI[ci][t - 1]
        AP_CI = 0
        
        # Izhikevich neuron model
        dvdt_CI = izhikevich_dvdt(v_CI_aux, u_CI_aux, I_CI[0][ci])
        dudt_CI = izhikevich_dudt(v_CI_aux, u_CI_aux, a_CI[0][ci], b_CI[0][ci])        
        # Self feeback - Inhibitory
        coupling_CI_CI = W_CI[ci][0]*PSC_CI[0][t]
        # Coupling CI to S - Inhibitory
        coupling_CI_S = W_CI_S[ci][0]*PSC_S[0][t]
        # coupling CI to M - Excitatory
        coupling_CI_M = W_CI_M[ci][0]*PSC_M[0][t]
        # Coupling CI to D - Inhibitory
        coupling_CI_D = W_CI_D[ci][0]*PSC_D[0][t]
        # Coupling CI to TC - Inhibitory
        coupling_CI_TC = W_CI_TC[ci][0]*PSC_TC[0][t]
        # Coupling CI to TR - Inhibitory
        coupling_CI_TR = W_CI_TR[ci][0]*PSC_TR[0][t]
        
        if(v_CI_aux >= v_threshold + zeta_CI[ci][t - 1]):
            v_CI_aux = 1*v_CI[ci][t]
            v_CI[ci][t] = 1*c_CI[0][ci]
            u_CI[ci][t] = 1*(u_CI_aux + d_CI[0][ci])
            AP_CI = 1
            spikes_CI[ci][t] = t
            
        v_CI[ci][t] = v_CI_aux + dt*(dvdt_CI + coupling_CI_S + coupling_CI_M + coupling_CI_D + coupling_CI_CI + coupling_CI_TC + coupling_CI_TR)
        u_CI[ci][t] = u_CI_aux + dudt_CI*dt
            
        # Synaptic connection - Within cortex
        syn_CI = tm_synapse_eq(r=r_CI, x=x_CI, Is=Is_CI, AP=AP_CI, tau_f=t_f_I, tau_d=t_d_I, tau_s=t_s_I, U=U_I, A=A_I, dt=dt)
        r_CI = syn_CI['r']
        x_CI = syn_CI['x']
        Is_CI = syn_CI['Is']
        Ipost_CI = syn_CI['Ipost']
            
    PSC_CI[0][t] = np.sum(Ipost_CI)
    
        
# # =============================================================================
# # TC
# # =============================================================================
#     for tc in range(n_TC):
#         v_TC_aux = v_TC[tc][t - 1]
#         u_TC_aux = u_TC[tc][t - 1]
#         AP_TC = 0
        
#         if(v_TC_aux >= v_threshold + zeta_TC[tc][t - 1]):
#             v_TC_aux = 1*v_TC[tc][t]
#             v_TC[tc][t] = 1*c_TC[0][tc]
#             u_TC[tc][t] = 1*u_TC_aux + d_TC[0][tc]
#             AP_TC = 1
#             spikes_TC[tc][t] = t
#         else:
#             dvdt_TC = izhikevich_dvdt(v_TC_aux, u_TC_aux, I_TC[0][tc])
#             dudt_TC = izhikevich_dudt(v_TC_aux, u_TC_aux, a_TC[0][tc], b_TC[0][tc])
            
#             # Self feeback - Inhibitory
#             coupling_TC_TC = W_TC_TC[tc][0]*PSC_TC[0][t]
#             # Coupling TC to S - Inhibitory
#             coupling_TC_S = W_TC_S[tc][0]*PSC_S[0][t]
#             # coupling TC to M - Excitatory
#             coupling_TC_M = W_TC_M[tc][0]*PSC_M[0][t]
#             # Coupling TC to D - Inhibitory
#             coupling_TC_D = W_TC_D[tc][0]*PSC_D[0][t]
#             # Coupling TC to CI - Inhibitory
#             coupling_TC_CI = W_TC_CI[tc][0]*PSC_CI[0][t]
#             # Coupling TC to TR - Inhibitory
#             coupling_TC_TR = W_TC_TR[tc][0]*PSC_TR[0][t]
            
#             v_TC[tc][t] = v_TC_aux + dt*(dvdt_TC + coupling_TC_S + coupling_TC_M + coupling_TC_D + coupling_TC_CI + coupling_TC_TC + coupling_TC_TR)
#             u_TC[tc][t] = u_TC_aux + dudt_TC*dt
            
#         # Synaptic connection - Within Thalamus
#         syn_TC = tm_synapse_eq(r=r_TC, x=x_TC, Is=Is_TC, AP=AP_TC, tau_f=t_f_E, tau_d=t_d_E, tau_s=t_s_E, U=U_E, A=A_E, dt=dt)
#         r_TC = syn_TC['r']
#         x_TC = syn_TC['x']
#         Is_TC = syn_TC['Is']
#         Ipost_TC = syn_TC['Ipost']
        
#         # Synaptic connection - Thalamus to Cortex
#         syn_TC_Cortex = tm_synapse_eq(r=r_TC, x=x_TC, Is=Is_TC, AP=AP_TC, tau_f=t_f_E, tau_d=t_d_E, tau_s=t_s_E, U=U_E, A=A_E, dt=dt)
#         r_TC_Cortex = syn_TC_Cortex['r']
#         x_TC_Cortex = syn_TC_Cortex['x']
#         Is_TC_Cortex = syn_TC_Cortex['Is']
#         Ipost_TC_Cortex = syn_TC_Cortex['Ipost']
        
#     PSC_TC[0][t] = np.sum(Ipost_TC)
#     PSC_TC_Cortex[0][t] = np.sum(Ipost_TC_Cortex)
# # =============================================================================
# # TR
# # =============================================================================
#     for tr in range(n_TR):
#         v_TR_aux = v_TR[tr][t - 1]
#         u_TR_aux = u_TR[tr][t - 1]
#         AP_TR = 0
        
#         if (v_TR_aux >= v_threshold + zeta_TR[tr][t - 1]):
#             v_TR_aux = 1*v_TR[tr][t]
#             v_TR[tr][t] = 1*c_TR[0][tr]
#             u_TR[tr][t] = 1*u_TR_aux + d_TR[0][tr]
#             AP_TR = 1
#             spikes_TR[tr][t] = t
#         else:
#             dvdt_TR = izhikevich_dvdt(v_TR_aux, u_TR_aux, I_TR[0][tr])
#             dudt_TR = izhikevich_dudt(v_TR_aux, u_TR_aux, a_TR[0][tr], b_TR[0][tr])
            
#             # Self feeback - Inhibitory
#             coupling_TR_TR = W_TR_TR[tr][0]*PSC_TR[0][t]
#             # Coupling TR to S - Inhibitory
#             coupling_TR_S = W_TR_S[tr][0]*PSC_S[0][t]
#             # coupling TR to M - Excitatory
#             coupling_TR_M = W_TR_M[tr][0]*PSC_M[0][t]
#             # Coupling TR to D - Inhibitory
#             coupling_TR_D = W_TR_D[tr][0]*PSC_D[0][t]
#             # Coupling TR to CI - Inhibitory
#             coupling_TR_CI = W_TR_CI[tr][0]*PSC_CI[0][t]
#             # Coupling TR to TC - Inhibitory
#             coupling_TR_TC = W_TR_TC[tr][0]*PSC_TC[0][t]
            
#             v_TC[tr][t] = v_TR_aux + dt*(dvdt_TR + coupling_TR_S + coupling_TR_M + coupling_TR_D + coupling_TR_CI + coupling_TR_TC + coupling_TR_TR)
#             u_TC[tr][t] = u_TR_aux + dudt_TR*dt
            
#         # Synaptic connection - Within Thalamus
#         syn_TR = tm_synapse_eq(r=r_TR, x=x_TR, Is=Is_TR, AP=AP_TR, tau_f=t_f_I, tau_d=t_d_I, tau_s=t_s_I, U=U_I, A=A_I, dt=dt)
#         r_TR = syn_TR['r']
#         x_TR = syn_TR['x']
#         Is_TR = syn_TR['Is']
#         Ipost_TR = syn_TR['Ipost']
            
#     PSC_TR[0][t] = np.sum(Ipost_TR)

plot_voltage(title="Layer S spikes", y=v_S[0], dt=dt, sim_time=sim_time)
plot_voltage(title="Layer M spikes", y=v_M[0], dt=dt, sim_time=sim_time)
plot_voltage(title="Layer D spikes", y=v_D[0], dt=dt, sim_time=sim_time)
plot_voltage(title="Layer CI spikes", y=v_CI[0], dt=dt, sim_time=sim_time)
# plot_voltage(title="Layer TC spikes", y=v_TC[0], dt=dt, sim_time=sim_time)
# plot_voltage(title="Layer TR spikes", y=v_TR[0], dt=dt, sim_time=sim_time)

plot_voltage(title="LFP Layer S", y=PSC_S[0], dt=dt, sim_time=sim_time)
plot_voltage(title="LFP Layer M", y=PSC_M[0], dt=dt, sim_time=sim_time)
plot_voltage(title="LFP Layer D", y=PSC_D[0], dt=dt, sim_time=sim_time)
plot_voltage(title="LFP Layer CI", y=PSC_CI[0], dt=dt, sim_time=sim_time)
# plot_voltage(title="LFP Layer TC", y=PSC_TC[0], dt=dt, sim_time=sim_time)
# plot_voltage(title="LFP Layer TR", y=PSC_TR[0], dt=dt, sim_time=sim_time)

plot_raster(title="S Raster Plot", num_neurons=n_S, spike_times=spikes_S, sim_time=sim_time, dt=dt)
plot_raster(title="M Raster Plot", num_neurons=n_M, spike_times=spikes_M, sim_time=sim_time, dt=dt)
plot_raster(title="D Raster Plot", num_neurons=n_D, spike_times=spikes_D, sim_time=sim_time, dt=dt)
plot_raster(title="CI Raster Plot", num_neurons=n_CI, spike_times=spikes_CI, sim_time=sim_time, dt=dt)

spikes_TR = TR_spikes
spikes_TC = TC_spikes

plot_raster_cortex(
    n_TR = n_TR, 
    n_TC = n_TC, 
    n_CI = n_CI, 
    n_D = n_D, 
    n_M = n_M, 
    n_S = n_S, 
    spike_times_TR = spikes_TR, 
    spike_times_TC = spikes_TC, 
    spike_times_CI = spikes_CI, 
    spike_times_D = spikes_D, 
    spike_times_M = spikes_M,
    spike_times_S = spikes_S,
    dbs = 0,
    sim_steps = num_steps,
    dt = dt,
    chop_till = chop_till, 
    n_total = total_neurons,
    n_CI_LTS = n_CI_2,
    n_D_IB = n_D_2,
    n_S_IB = n_S_2,
    )