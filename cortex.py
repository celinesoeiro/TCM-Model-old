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
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

random.seed(0)
random_factor = np.round(random.random(),2)

def poissonSpikeGen(n_trials, time, fr, dt):
    spikes_matriz = np.random.rand(n_trials,len(time)) < fr*dt
    
    return {"spikes": spikes_matriz, "time": time}

def plot_raster(spikes, spikes_time, fr, n_total, dt):
    # plt.figure(figsize=(15, 10))
    plt.ylim(-0.1, spikes.shape[0] + 0.1)
    plt.yticks(np.arange(0, spikes.shape[0]))
    # plt.xticks(np.arange(0, len(spikes_time) + dt, ms*dt))
    
    for trial in range(spikes.shape[0]):
        spike_position = spikes_time[spikes[trial, :]]
        for spike_count in range(len(spike_position)):
            plt.plot([spike_position[spike_count], spike_position[spike_count]], 
                          [trial-0.05, trial+0.05], 'k')
            
    plt.title(f'spikes at {fr}Hz')
    plt.show()

def plot_voltage(title, x, y):
    # plt.figure(figsize=(15, 15))

    plt.title(title)

    plt.plot(x, y)

    # Set the x-axis label
    plt.xlabel('Time')
    plt.ylabel('Voltage')

    # Show the plot
    plt.show()

def izhikevich_dvdt(v, u, I):
    return 0.04*v**2 + 5*v + 140 - u + I

def izhikevich_dudt(v, u, a, b):
    return a*(b*v - u)

# =============================================================================
# GLOBAL PARAMS
# =============================================================================
vp = 30
vr = -65

ms = 1000                                       # seconds
fr = 20                                         # Hz
sim_time = 1                                    # s
dt = 10/ms                                      # 10ms
ss_time = 0                                     # s - steady state time to trash later
sim_time_total = int((sim_time + ss_time)*ms)   # ms
sim_steps_aux = np.arange(0, sim_time_total, dt)
sim_steps = sim_steps_aux.reshape(1, len(sim_steps_aux))
chop_till = int(ss_time*ms); 
n_trials = 20

n_D = 1
n_CI = 1
total_neurons = n_D + n_CI

spikes_D = np.zeros((1, len(sim_steps_aux)))
spikes_CI = np.zeros((1, len(sim_steps_aux)))

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

v_D = np.zeros((1,len(sim_steps_aux)))
v_D[0][0] = vr
u_D = np.zeros((1,len(sim_steps_aux)))
u_D[0][0] = b_D*v_D[0][0]

v_CI = np.zeros((1, len(sim_steps_aux)))
v_CI[0][0] = vr
u_CI = np.zeros((1, len(sim_steps_aux)))
u_CI[0][0] = b_CI*v_CI[0][0]
# =============================================================================
# Noise
# =============================================================================
mean = 0
std = 1
zeta_noise = np.random.normal(mean, std, size=len(sim_steps_aux))
kisi_noise = zeta_noise/2

# =============================================================================
# Post Synaptic Currents
# =============================================================================
PSC_D = np.zeros((1, len(sim_steps_aux)))
PSC_CI = np.zeros((1, len(sim_steps_aux)))

# =============================================================================
# MAKING THALAMIC INPUT
# =============================================================================
I_Thalamus = poissonSpikeGen(1, sim_steps[0], fr, dt)
I_Thalamus_spikes = I_Thalamus['spikes']

plot_raster(I_Thalamus_spikes, sim_steps_aux, fr, total_neurons, dt)

# =============================================================================
# Idc
# =============================================================================
Idc = [3.6, 3.7, 3.9, 0.5, 0.7]
I_D = np.c_[Idc[0]*np.ones((1, 1)), Idc[1]*np.ones((1, 0))]
I_CI = np.c_[Idc[2]*np.ones((1, 1)), Idc[3]*np.ones((1, 0))]

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
# LOOPING THROUGH
# =============================================================================
for t in range(1, len(sim_steps[0])):
    # D Layer
    v_D_aux = 1*v_D[0][t - 1]
    u_D_aux = 1*u_D[0][t - 1]
    
    if (v_D_aux >= vp):
        AP_aux_D = 1
        v_aux = 1*v_D[0][t]
        v_D[0][t] = c_D[0][0]
        u_D[0][t] = u_D_aux + d_D[0][0]
        spikes_D[0][t] = t
    else:
        coupling_D_D = W_D*PSC_D[0][t - 1]/n_D
        coupling_D_CI = W_D_CI*PSC_CI[0][t - 1]/n_CI
        
        coupling_D_T = W_D_TC*I_Thalamus_spikes[0][t - 1]
        
        izhikevich_v_D = izhikevich_dvdt(v_D_aux, u_D_aux, I_D[0][0])
        izhikevich_u_D = izhikevich_dudt(v_D_aux, u_D_aux, a_D[0][0], b_D[0][0])
        
        v_D[0][t] = v_D_aux + dt*(izhikevich_v_D + coupling_D_D + coupling_D_CI + coupling_D_T + zeta_noise[t - 1]) 
        u_D[0][t] = u_D_aux + dt*izhikevich_u_D
    
    
    # CI layer
    v_CI_aux = 1*v_CI[0][t - 1]
    u_CI_aux = 1*u_CI[0][t - 1]
    
    if(v_CI_aux >= vp):
        AP_aux_CI = 1
        v_aux = 1*v_CI[0][t]
        v_CI[0][t] = c_CI[0][0]
        u_CI[0][t] = u_CI_aux + d_CI[0][0]
        spikes_CI[0][t] = t
    else:
        coupling_CI_CI = W_D_CI*PSC_CI[0][t - 1]/n_CI
        coupling_CI_D = W_CI_D*PSC_D[0][t - 1]/n_D
        
        coupling_CI_T = W_CI_TC*I_Thalamus_spikes[0][t - 1]
        
        izhikevich_v_CI = izhikevich_dvdt(v_CI_aux, u_CI_aux, I_CI[0][0])
        izhikevich_u_CI = izhikevich_dudt(v_CI_aux, u_CI_aux, a_CI[0][0], b_CI[0][0]) 
        
        v_CI[0][t] = v_CI_aux +  dt*(izhikevich_v_CI + coupling_CI_CI + coupling_CI_D + coupling_CI_T + zeta_noise[t - 1])
        u_CI[0][t] = u_CI_aux + dt*izhikevich_u_CI
    
        
# =============================================================================
# Next: Add TM synapse model so we can have spikes
# =============================================================================
    

# =============================================================================
# PLOTS
# =============================================================================
plot_voltage('membrane potential - Layer D', sim_steps[0], v_D[0])
plot_voltage('membrane potential - Layer CI', sim_steps[0], v_CI[0])


# =============================================================================
# 
# =============================================================================
