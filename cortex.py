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
import matplotlib.pyplot as plt

random.seed(0)
random_factor = np.round(random.random(),2)


def poissonSpikeGen(n_trials, time, fr, dt):
    spikes_matriz = np.random.rand(n_trials,len(time)) < fr*dt
    print(spikes_matriz)
    
    # time = np.arange(0, time)
    
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

# I = poissonSpikeGen(
#     n_trials = total_neurons, 
#     time = sim_steps, 
#     fr = fr, 
#     dt = dt)

# spikes = I['spikes']
# spikes_time = I['time']
# plot_raster(spikes, spikes_time, fr, total_neurons, dt)

# =============================================================================
# PARAMS
# =============================================================================
vp = 30
vr = -65

ms = 1000                                       # seconds
fr = 20                                         # Hz
sim_time = 1                                    # s
dt = 10/ms                                      # 10ms
ss_time = 0                                     # s - steady state time to trash later
sim_time_total = int((sim_time + ss_time)*ms)   # ms
sim_steps = np.arange(0, sim_time_total, dt)
chop_till = int(ss_time*ms); 
n_trials = 20

n_D = 1
n_CI = 1
total_neurons = n_D + n_CI

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

# NOISE
mean = 0
std = 1
zeta_noise = np.random.normal(mean, std, size=len(sim_steps))
kisi_noise = zeta_noise/2

# =============================================================================
# Creating Neuronal Network
# =============================================================================
v_D = vr*np.ones((1,len(sim_steps)))
u_D = np.zeros((1,len(sim_steps)))

v_CI = vr*np.ones((1, len(sim_steps)))
u_CI = np.zeros((1, len(sim_steps)))

I_Thalamus = poissonSpikeGen(1, sim_steps, fr, dt)
I_Thalamus_spikes = I_Thalamus['spikes']

r_D = 0 + 1*np.random.rand(n_D, 1)
r_CI = 0 + 1*np.random.rand(n_CI, 1)

connectivity_factor_normal = 2.5 

I_D = np.zeros((1,len(sim_steps)))
I_CI = np.zeros((1,len(sim_steps)))
spikes_D = np.zeros((1, len(sim_steps)))
spikes_CI = np.zeros((1, len(sim_steps)))

aei_D_CI = -7.5e3/connectivity_factor_normal;   
W_EI_D_CI = aei_D_CI*r_D;

aie_CI_D = 2e2/connectivity_factor_normal;     
W_IE_CI_D = aie_CI_D*r_CI;

for t in range(1, len(sim_steps)):
    v_D_aux = v_D[0][t - 1]
    v_CI_aux = v_CI[0][t - 1]
    
    u_D_aux = u_D[0][t - 1]
    u_CI_aux = u_CI[0][t - 1]
    
    coupling_D_CI = W_EI_D_CI*I_CI[0][t - 1]
    coupling_CI_D = W_IE_CI_D*I_D[0][t - 1]
    
    v_D[0][t] = v_D_aux +  dt*(0.04*v_D_aux** + 5*v_D_aux - u_D_aux + 140 + I_Thalamus_spikes[0][t - 1] + coupling_D_CI) + zeta_noise[t - 1]
    u_D[0][t] = u_D_aux + dt*(a_D[0][0]*(b_D[0][0]*v_D_aux - u_D_aux)) + zeta_noise[t - 1]
    
    if (v_D[0][t] >= vp):
        AP_aux = 1
        v_aux = vp
        v_D[0][t] = c_D[0][0]
        u_D[0][t] = u_D_aux + d_D[0][0]
        spikes_D[0][t] = t
        
    v_CI[0][t] = v_CI_aux +  dt*(0.04*v_CI_aux** + 5*v_CI_aux - u_CI_aux + 140 + I_Thalamus_spikes[0][t - 1] + coupling_CI_D) + zeta_noise[t - 1]
    u_CI[0][t] = u_CI_aux + dt*(a_CI[0][0]*(b_CI[0][0]*v_CI_aux - u_CI_aux)) + zeta_noise[t - 1]
    
    if (v_CI[0][t] >= vp):
        AP_aux = 1
        v_aux = vp
        v_CI[0][t] = c_CI[0][0]
        u_CI[0][t] = u_CI_aux + d_CI[0][0]
        spikes_CI[0][t] = t
        
    
    
    
    
    
