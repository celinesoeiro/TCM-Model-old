"""
@description: Layer D + Cortical Interneurons - Motor Cortex
@author: Celine Soeiro

Neuron firing rate = 8Hz => Means that the neuron fires 8 action potentials per second on avarage
(ISI) Inter Spike Interval refers to the time interval between two spikes => 1000/8 = 125ms => 1 spike a cada 125ms

Normal condition:
    - Firing rate: 8Hz
    - ISI: 1000/8 = 125ms (1 spike in each 125ms)
Parkinsonian condition:
    - Firing rate: 22.5Hz (13 ~ 30Hz (Beta band))
    - ISI: 44,45ms (1 spike in each 44,45ms) (1000/13 = 76,9ms (1 spike in each 77ms) to 1000/30=33,34ms (1 spike in each 33,34ms))
DBS:
    - Firing rate: 80Hz +
    - ISI: 1000/80 = 12,5ms (1 spike in each 12,5ms)
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
# DBS
# =============================================================================
connectivity_factor_normal = 2.5 
connectivity_factor_PD = 5
connectivity_factor = connectivity_factor_normal

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
# Noise
# =============================================================================
mean = 0
std = 1
zeta_noise = np.random.normal(mean, std, size=num_steps)
kisi_noise = zeta_noise/2

# =============================================================================
# Post Synaptic Currents
# =============================================================================
PSC_D = np.zeros((1, num_steps))
PSC_CI = np.zeros((1, num_steps))
PSC_TC = np.zeros((1, num_steps))

# =============================================================================
# Idc
# =============================================================================
Idc = [3.6, 3.7, 3.9, 0.5, 0.7]

I_D = np.c_[Idc[0]*np.ones((1, 1)), Idc[1]*np.ones((1, 0))]
I_CI = np.c_[Idc[2]*np.ones((1, 1)), Idc[3]*np.ones((1, 0))]
I_TC = np.c_[Idc[4]*np.ones((1, 1)), Idc[4]*np.ones((1, 0))]

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
# TM synapse
# =============================================================================
r_D = np.zeros((1, 3))
x_D = np.zeros((1, 3))
I_syn_D = np.zeros((1, 3))

r_CI = np.zeros((1, 3))
x_CI = np.zeros((1, 3))
I_syn_CI = np.zeros((1, 3))

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

Is_D = np.zeros((1, n_D))
Is_CI = np.zeros((1, n_CI))

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

# =============================================================================
# LOOPING THROUGH
# =============================================================================
for t in range(num_steps):
    # D Layer
    for k in range(n_D):
        print('k - D')
        v_D_aux = 1*v_D[k][t - 1]
        u_D_aux = 1*u_D[k][t - 1]
        AP_D = 0
        
        coupling_D_D = W_D[0][0]*PSC_D[k][t - 1]/n_D
        coupling_D_CI = W_D_CI[0][0]*PSC_CI[k][t - 1]/n_CI
        coupling_D_T = W_D_TC[0][0]*I_Thalamus[k][t - 1]/n_TC
        
        izhikevich_v_D = izhikevich_dvdt(v_D_aux, u_D_aux, I_D[0][0])
        izhikevich_u_D = izhikevich_dudt(v_D_aux, u_D_aux, a_D[0][0], b_D[0][0])
        
        v_D[k][t] = v_D_aux + dt*(izhikevich_v_D + coupling_D_D + coupling_D_CI + coupling_D_T + zeta_noise[t - 1]) 
        u_D[k][t] = u_D_aux + dt*izhikevich_u_D
        
        if (v_D[0][t] >= v_threshold):
            AP_D = 1
            v_D[0][t] = 1*c_D[0][0]
            u_D[0][t] = 1*(u_D_aux + d_D[0][0])
            spikes_D[0][t] = int(t)
            
        # Pseudo-linear
        for p in range(3):
            r_D_aux = r_D[0][p]
            x_D_aux = x_D[0][p]
            I_s_D_aux = I_syn_D[0][p]
            # Solve EDOs using Euler method
            r_D[0][p] = r_D_aux + dt*(-r_D_aux/t_f_E[p] + U_E[p]*(1 - r_D_aux)*AP_D)
            x_D[0][p] = x_D_aux + dt*((1 - x_D_aux)/t_d_E[p] - (r_D[0][p] + U_E[p]*(1 - r_D[0][p]))*x_D_aux*AP_D)
            I_syn_D[0][p] = I_s_D_aux + dt*(-I_s_D_aux/t_s_E + A_E[p]*x_D[0][p]*(r_D[0][p] + U_E[p]*(1 - r_D[0][p]))*AP_D)
                
        Is_D[0][k] = np.sum(I_syn_D)
        
    PSC_D[0][t] = np.sum(Is_D)
        
    # CI layer
    for k in range(n_CI):
        print('k - CI')
        v_CI_aux = 1*v_CI[k][t - 1]
        u_CI_aux = 1*u_CI[k][t - 1]
        AP_CI = 0
        
        coupling_CI_CI = W_D_CI[0][0]*PSC_CI[k][t - 1]/n_CI
        coupling_CI_D = W_CI_D[0][0]*PSC_D[k][t - 1]/n_D
        coupling_CI_T = W_CI_TC[0][0]*I_Thalamus[k][t - 1]/n_TC
        
        izhikevich_v_CI = izhikevich_dvdt(v_CI_aux, u_CI_aux, I_CI[0][0])
        izhikevich_u_CI = izhikevich_dudt(v_CI_aux, u_CI_aux, a_CI[0][0], b_CI[0][0]) 
        
        v_CI[k][t] = v_CI_aux +  dt*(izhikevich_v_CI + coupling_CI_CI + coupling_CI_D + coupling_CI_T + zeta_noise[t - 1])
        u_CI[k][t] = u_CI_aux + dt*izhikevich_u_CI
        
        if(v_CI_aux >= v_threshold):
            AP_CI = 1
            v_CI[0][t] = 1*c_CI[0][0]
            u_CI[0][t] = 1*(u_CI_aux + d_CI[0][0])
            spikes_CI[0][t] = int(t)

        # Pseudo-linear
        for p in range(3):
            r_CI_aux = r_CI[0][p]
            x_CI_aux = x_CI[0][p]
            I_s_CI_aux = I_syn_CI[0][p]
            # Solve EDOs using Euler method
            r_CI[0][p] = r_CI_aux + dt*(-r_CI_aux/t_f_I[p] + U_I[p]*(1 - r_CI_aux)*AP_CI)
            x_CI[0][p] = x_CI_aux + dt*((1 - x_CI_aux)/t_d_I[p] - (r_D[0][p] + U_I[p]*(1 - r_CI[0][p]))*x_CI_aux*AP_CI)
            I_syn_CI[0][p] = I_s_CI_aux + dt*(-I_s_CI_aux/t_s_I + A_I[p]*x_CI[0][p]*(r_CI[0][p] + U_I[p]*(1 - r_CI[0][p]))*AP_CI)
                
        Is_CI[0][k] = np.sum(I_syn_CI)
    
    PSC_CI[0][t] = np.sum(Is_CI)
    
# =============================================================================
# PLOTS - VOLTAGE
# =============================================================================
plot_voltage('membrane potential - Layer D', v_D[0], dt, sim_time)
plot_voltage('membrane potential - Layer CI', v_CI[0], dt, sim_time)
plot_voltage('LFP - Layer D', PSC_D[0], dt, sim_time)
plot_voltage('LFP - CI', PSC_CI[0], dt, sim_time)

# # =============================================================================
# # PLOTS - RASTER PLOTS
# # =============================================================================
# f_D = get_frequency(spikes_D, sim_time_total)    # Hz
# f_CI = get_frequency(spikes_CI, sim_time_total)  # Hz

# plot_raster('Layer D', spikes_D, sim_time_total, f_D, n_D, dt)
# plot_raster('Layer CI', spikes_CI, sim_time_total, f_CI, n_CI, dt)

# # =============================================================================
# # PLOTS - PSC = LFP 
# # =============================================================================
# plot_lfp('LFP - Layer D', sim_steps[0], PSC_D[0])
# plot_lfp('LFP - Layer CI', sim_steps[0], PSC_CI[0])

# # =============================================================================
# # PLOTS - POWER SPECTRAL DENSITY
# # =============================================================================
# plot_psd_welch('PSD - Layer D', PSC_D, fs)
# plot_psd_welch('PSD - Layer CI', PSC_CI, fs)
