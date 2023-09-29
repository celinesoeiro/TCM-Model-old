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

n_S = 1
n_M = 1
n_D = 1
n_CI = 1
n_TC = 1
n_TR = 1
total_neurons = n_S + n_M + n_D + n_CI + n_TC + n_TR

spikes_S = np.zeros((n_S, num_steps))
spikes_M = np.zeros((n_M, num_steps))
spikes_D = np.zeros((n_D, num_steps))
spikes_CI = np.zeros((n_CI, num_steps))

f_thalamus = 8                                  # Hz (Thalamus frequency)

# =============================================================================
# Izhikevich neuron parameters
# =============================================================================
#    0-RS  1-IB  2-FS 3-LTS 4-TC  5-TR 
a = [0.02, 0.02, 0.1, 0.02, 0.02, 0.02]
b = [0.2,  0.2,  0.2, 0.25, 0.25, 0.25]
c = [-65,  -55,  -65, -65,   -65,  -65]
d = [8,    4,      2,   2,  0.05, 2.05]

# S -> 1 RS neuron
a_S = np.c_[a[0]*np.ones((1, 1)), a[1]*np.ones((1, 0))]
b_S = np.c_[b[0]*np.ones((1, 1)), b[1]*np.ones((1, 0))]
c_S = np.c_[c[0]*np.ones((1, 1)), c[1]*np.ones((1, 0))] + 15*random_factor**2
d_S = np.c_[d[0]*np.ones((1, 1)), d[1]*np.ones((1, 0))] - 0.6*random_factor**2

# M -> 1 RS neuron
a_M = np.c_[a[0]*np.ones((1, 1)), a[1]*np.ones((1, 0))]
b_M = np.c_[b[0]*np.ones((1, 1)), b[1]*np.ones((1, 0))]
c_M = np.c_[c[0]*np.ones((1, 1)), c[1]*np.ones((1, 0))] + 15*random_factor**2
d_M = np.c_[d[0]*np.ones((1, 1)), d[1]*np.ones((1, 0))] - 0.6*random_factor**2

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

v_S = np.zeros((n_S, num_steps))
u_S = np.zeros((n_S, num_steps))

v_M = np.zeros((n_M, num_steps))
u_M = np.zeros((n_M, num_steps))

v_D = np.zeros((n_D, num_steps))
u_D = np.zeros((n_D, num_steps))

v_CI = np.zeros((n_CI, num_steps))
u_CI = np.zeros((n_CI, num_steps))

v_S[0][0] = v_resting
u_S[0][0] = b_S*v_resting

v_M[0][0] = v_resting
u_M[0][0] = b_M*v_resting

v_D[0][0] = v_resting
u_D[0][0] = b_D*v_resting

v_CI[0][0] = v_resting
u_CI[0][0] = b_CI*v_resting

# =============================================================================
# Idc
# =============================================================================
Idc = [3.6, 3.7, 3.9, 0.5, 0.7]

I_S = np.c_[Idc[0]*np.ones((1, 1)), Idc[1]*np.ones((1, 0))]
I_M = np.c_[Idc[0]*np.ones((1, 1)), Idc[0]*np.ones((1, 0))]
I_D = np.c_[Idc[0]*np.ones((1, 1)), Idc[1]*np.ones((1, 0))]
I_CI = np.c_[Idc[2]*np.ones((1, 1)), Idc[3]*np.ones((1, 0))]
I_TC = np.c_[Idc[4]*np.ones((1, 1)), Idc[4]*np.ones((1, 0))]

# =============================================================================
# Post Synaptic Currents
# =============================================================================
PSC_S = np.zeros((n_S, num_steps))
PSC_M = np.zeros((n_M, num_steps))
PSC_D = np.zeros((n_D, num_steps))
PSC_CI = np.zeros((n_CI, num_steps))
PSC_TC = np.zeros((n_TC, num_steps))

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

# Initialize matrix (4 structures -> 4x4 matrix)
matrix = np.zeros((4,4))
# Main Diagonal
matrix[0][0] = np.mean(W_S)
matrix[1][1] = np.mean(W_M)
matrix[2][2] = np.mean(W_D)
matrix[3][3] = np.mean(W_CI)
# First column - Layer S
matrix[1][0] = np.mean(W_S_M)
matrix[2][0] = np.mean(W_S_D)
matrix[3][0] = np.mean(W_S_CI)
# Second column - Layer M
matrix[0][1] = np.mean(W_M_S)
matrix[2][1] = np.mean(W_M_D)
matrix[3][1] = np.mean(W_M_CI)
# Thid column - Layer D
matrix[0][2] = np.mean(W_D_S)
matrix[1][2] = np.mean(W_D_M)
matrix[3][2] = np.mean(W_D_CI)
# Fourth column - Structure CI
matrix[0][3] = np.mean(W_CI_S)
matrix[1][3] = np.mean(W_CI_M)
matrix[2][3] = np.mean(W_CI_D)

# normalizing Normal coupling matrix
matrix_norm = matrix/np.linalg.norm(matrix)

print("-- Printing the coupling matrixes")

CM_Normal = pd.DataFrame(matrix_norm, columns=['S', 'M', 'D', 'CI'])
plot_heat_map(matrix_normal = CM_Normal)

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
r_S = np.zeros((1, 3))
x_S = np.zeros((1, 3))
Is_S = np.zeros((1, 3))

r_M = np.zeros((1, 3))
x_M = np.zeros((1, 3))
Is_M = np.zeros((1, 3))

r_D = np.zeros((1, 3))
x_D = np.zeros((1, 3))
Is_D = np.zeros((1, 3))

r_CI = np.zeros((1, 3))
x_CI = np.zeros((1, 3))
Is_CI = np.zeros((1, 3))

r_TC = np.zeros((1, 3))
x_TC = np.zeros((1, 3))
I_syn_TC = np.zeros((1, 3))

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

# =============================================================================
# MAKING THALAMIC INPUT
# =============================================================================
TC_spikes, I_TC = poisson_spike_generator(
    num_steps = num_steps, 
    dt = dt, 
    num_neurons = n_TC, 
    thalamic_firing_rate = f_thalamus,
    current_value = Idc[4])

get_frequency(I_TC, sim_time)

plot_raster(title="TC Raster Plot", num_neurons=1, spike_times=TC_spikes, sim_time=sim_time, dt=dt)
plot_voltage(title="TC spikes", y=I_TC[0], dt=dt, sim_time=sim_time)

TR_spikes, I_TR = poisson_spike_generator(
    num_steps = num_steps, 
    dt = dt, 
    num_neurons = n_TR, 
    thalamic_firing_rate = f_thalamus,
    current_value = Idc[4])

get_frequency(I_TR, sim_time)

plot_raster(title="TR Raster Plot", num_neurons=1, spike_times=TR_spikes, sim_time=sim_time, dt=dt)
plot_voltage(title="TR spikes", y=I_TR[0], dt=dt, sim_time=sim_time)

# =============================================================================
# LAYER D & LAYER CI & LAYER S
# =============================================================================
for t in range(1, num_steps):
# =============================================================================
#     # S
# =============================================================================
    v_S_aux = v_S[0][t - 1]
    u_S_aux = u_S[0][t - 1]
    AP_S = 0
    
    if(v_S_aux >= v_threshold + zeta_S[0][t - 1]):
        v_S_aux = 1*v_S[0][t]
        v_S[0][t] = 1*c_S[0][0]
        u_S[0][t] = u_S_aux + d_S[0][0]
        AP_S = 1
        spikes_S[0][t] = t
    else:
        dvdt_S = izhikevich_dvdt(v_S_aux, u_S_aux, I_S[0][0])
        dudt_S = izhikevich_dudt(v_S_aux, u_S_aux, a_S[0][0], b_S[0][0])
        
        # Self feedback - Inhibitory
        coupling_S_S = W_S*PSC_S[0][t]
        # Coupling S to M - Excitatory
        coupling_S_M = W_S_M*PSC_M[0][t]
        # Coupling S to D - Excitatory
        coupling_S_D = W_S_D*PSC_D[0][t]
        # Coupling S to CI - Excitatory 
        coupling_S_CI = W_S_CI*PSC_CI[0][t]
        # Coupling S to TC - Excitatory
        coupling_S_TC = W_S_TC*I_TC[0][t]
        # Coupling S to TR - Inhibitory
        coupling_S_TR = W_S_TR*I_TR[0][t]
        
        v_S[0][t] = v_S_aux + dt*(dvdt_S + coupling_S_S + coupling_S_M + coupling_S_D + coupling_S_CI + coupling_S_TC + coupling_S_TR)
        u_S[0][t] = u_S_aux + dudt_S*dt
        
    # Synaptic connection - within cortex
    syn_S = tm_synapse_eq(r=r_S, x=x_S, Is=Is_S, AP=AP_S, tau_f=t_f_E, tau_d=t_f_E, tau_s=t_s_E, U=U_E, A=A_E, dt=dt)
    r_S = syn_S['r']
    x_S = syn_S['x']
    Is_S = syn_S['Is']
    Ipost_S = syn_S['Ipost']
        
    PSC_S[0][t] = np.sum(Ipost_S)
    
# =============================================================================
#     # M
# =============================================================================
    v_M_aux = v_M[0][t - 1]
    u_M_aux = u_M[0][t - 1]
    AP_M = 0
    
    if(v_M_aux >= v_threshold + zeta_M[0][t - 1]):
        v_M_aux = 1*v_M[0][t]
        v_M[0][t] = 1*c_M[0][0]
        u_M[0][t] = u_M_aux + d_M[0][0]
        AP_M = 1
        spikes_M[0][t] = t
    else:
        dvdt_M = izhikevich_dvdt(v_M_aux, u_M_aux, I_M[0][0])
        dudt_M = izhikevich_dudt(v_M_aux, u_M_aux, a_M[0][0], b_M[0][0])
        
        # Self feedback - Inhibitory
        coupling_M_M = W_M*PSC_M[0][t]
        # Coupling M to S - Excitarory
        coupling_M_S = W_M_S*PSC_S[0][t]
        # Coupling M to D - Excitatory
        coupling_M_D = W_S_D*PSC_D[0][t]
        # Coupling M to CI - Excitatory 
        coupling_M_CI = W_S_CI*PSC_CI[0][t]
        # Coupling M to TC - Excitatory
        coupling_M_TC = W_S_TC*I_TC[0][t]
        # Coupling M to TR - Inhibitory
        coupling_M_TR = W_S_TR*I_TR[0][t]
        
        v_M[0][t] = v_M_aux + dt*(dvdt_M + coupling_M_S + coupling_M_M + coupling_M_D + coupling_S_CI + coupling_S_TC + coupling_S_TR)
        u_M[0][t] = u_M_aux + dudt_M*dt
        
    # Synaptic connection - within cortex
    syn_M = tm_synapse_eq(r=r_M, x=x_M, Is=Is_M, AP=AP_M, tau_f=t_f_E, tau_d=t_f_E, tau_s=t_s_E, U=U_E, A=A_E, dt=dt)
    r_M = syn_S['r']
    x_M = syn_S['x']
    Is_M = syn_S['Is']
    Ipost_M = syn_S['Ipost']
        
    PSC_M[0][t] = np.sum(Ipost_M)
    
# =============================================================================
#     # D
# =============================================================================
    v_D_aux = v_D[0][t - 1]
    u_D_aux = u_D[0][t - 1]
    AP_D = 0
    
    if(v_D_aux >= v_threshold + zeta_D[0][t - 1]):
        v_D_aux = 1*v_D[0][t]
        v_D[0][t] = 1*c_D[0][0]
        u_D[0][t] = u_D_aux + d_D[0][0]
        AP_D = 1
        spikes_D[0][t] = t
    else:
        dvdt_D = izhikevich_dvdt(v_D_aux, u_D_aux, I_D[0][0])
        dudt_D = izhikevich_dudt(v_D_aux, u_D_aux, a_D[0][0], b_D[0][0])
        
        # Self feedback - Inhibitory
        coupling_D_D = W_D*PSC_D[0][t]
        # Coupling D to S - Excitatory
        coupling_D_S = W_D_S*PSC_S[0][t]
        # Coupling D to M - Excitatory
        coupling_D_M = W_D_M*PSC_M[0][t]
        # Coupling D to CI - Excitatory 
        coupling_D_CI = W_D_CI*PSC_CI[0][t]
        # Coupling D to TC - Excitatory
        coupling_D_TC = W_D_TC*I_TC[0][t]
        # Coupling D to TR - Inhibitory
        coupling_D_TR = W_D_TR*I_TR[0][t]
        
        v_D[0][t] = v_D_aux + dt*(dvdt_D + coupling_D_S + coupling_D_M + coupling_D_D + coupling_D_CI + coupling_D_TC + coupling_D_TR)
        u_D[0][t] = u_D_aux + dudt_D*dt
        
    # Synaptic connection - Within Cortex
    syn_D = tm_synapse_eq(r=r_D, x=x_D, Is=Is_D, AP=AP_D, tau_f=t_f_E, tau_d=t_f_E, tau_s=t_s_E, U=U_E, A=A_E, dt=dt)
    r_D = syn_D['r']
    x_D = syn_D['x']
    Is_D = syn_D['Is']
    Ipost_D = syn_D['Ipost']
        
    PSC_D[0][t] = np.sum(Ipost_D)
    
# =============================================================================
#     # CI
# =============================================================================
    v_CI_aux = v_CI[0][t - 1]
    u_CI_aux = u_CI[0][t - 1]
    AP_CI = 0
    
    if(v_CI_aux >= v_threshold + zeta_CI[0][t - 1]):
        v_CI_aux = 1*v_CI[0][t]
        v_CI[0][t] = 1*c_CI[0][0]
        u_CI[0][t] = u_CI_aux + d_CI[0][0]
        AP_CI = 1
        spikes_CI[0][t] = t
    else:
        dvdt_CI = izhikevich_dvdt(v_CI_aux, u_CI_aux, I_CI[0][0])
        dudt_CI = izhikevich_dudt(v_CI_aux, u_CI_aux, a_CI[0][0], b_CI[0][0])
        
        # Self feeback - Inhibitory
        coupling_CI_CI = W_CI*PSC_CI[0][t]
        # Coupling CI to S - Inhibitory
        coupling_CI_S = W_CI_S*PSC_S[0][t]
        # coupling CI to M - Excitatory
        coupling_CI_M = W_CI_M*PSC_M[0][t]
        # Coupling CI to D - Inhibitory
        coupling_CI_D = W_CI_D*PSC_D[0][t]
        # Coupling CI to TC - Inhibitory
        coupling_CI_TC = W_CI_TC*I_TC[0][t]
        # Coupling CI to TR - Inhibitory
        coupling_CI_TR = W_CI_TR*I_TR[0][t]
        
        v_CI[0][t] = v_CI_aux + dt*(dvdt_CI + coupling_CI_S + coupling_CI_M + coupling_CI_D + coupling_CI_CI + coupling_CI_TC + coupling_CI_TR)
        u_CI[0][t] = u_CI_aux + dudt_CI*dt
        
    # Synaptic connection - Within cortex
    syn_CI = tm_synapse_eq(r=r_CI, x=x_CI, Is=Is_CI, AP=AP_CI, tau_f=t_f_I, tau_d=t_d_I, tau_s=t_s_I, U=U_I, A=A_I, dt=dt)
    r_CI = syn_CI['r']
    x_CI = syn_CI['x']
    Is_CI = syn_CI['Is']
    Ipost_CI = syn_CI['Ipost']
        
    PSC_CI[0][t] = np.sum(Ipost_CI)
    
plot_voltage(title="Layer S spikes", y=v_S[0], dt=dt, sim_time=sim_time)
plot_voltage(title="Layer M spikes", y=v_M[0], dt=dt, sim_time=sim_time)
plot_voltage(title="Layer D spikes", y=v_D[0], dt=dt, sim_time=sim_time)
plot_voltage(title="Layer CI spikes", y=v_CI[0], dt=dt, sim_time=sim_time)

plot_voltage(title="LFP Layer S", y=PSC_S[0], dt=dt, sim_time=sim_time)
plot_voltage(title="LFP Layer M", y=PSC_M[0], dt=dt, sim_time=sim_time)
plot_voltage(title="LFP Layer D", y=PSC_D[0], dt=dt, sim_time=sim_time)
plot_voltage(title="LFP Layer CI", y=PSC_CI[0], dt=dt, sim_time=sim_time)


