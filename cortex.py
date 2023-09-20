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
from scipy.signal import welch

random.seed(0)
random_factor = np.round(random.random(),2)

def poissonSpikeGen(time, fr, dt, n_neurons):
    # Calculate the number of spikes expected per neuron in the given time step
    mean_spikes = fr * dt
    spike_times = []
    
    for _ in range(n_neurons):
        num_steps = int(time/dt)    
        spike_train = np.random.poisson(mean_spikes, num_steps)*dt
        spike_times.append(spike_train)
        
    #spikes_matriz = np.random.rand(n_trials,len(time)) < fr*dt
            
    return spike_times

def plot_raster(title, spikes, spikes_time, fr, n_total, dt):
    # plt.figure(figsize=(15, 10))
    # plt.figure(figsize=(8, 4))
    plt.eventplot(spikes, color='black', linewidths=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Neuron')
    plt.title(f'{title} - spikes at {fr}Hz')
    plt.xlim(0, sim_time_total)
    plt.ylim(0, 1)
    plt.yticks([])
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
    
def plot_lfp(title, x, y):
    # plt.figure(figsize=(15, 15))

    plt.title(title)

    plt.plot(x, y)

    # Set the x-axis label
    plt.xlabel('Time')
    plt.ylabel('Current')

    # Show the plot
    plt.show()
    
def plot_psd_welch(title, signal, frequency):
    frequencie, psd = welch(signal, fs = frequency,  nperseg=1024)

    # Create a plot
    # plt.figure(figsize=(8, 4))
    plt.semilogy(frequencie.reshape(1, len(frequencie)), psd)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.title(title)
    plt.grid(True)
    plt.show()

def izhikevich_dvdt(v, u, I):
    return 0.04*v**2 + 5*v + 140 - u + I

def izhikevich_dudt(v, u, a, b):
    return a*(b*v - u)

def tm_synapse_eq(r, x, Is, AP, tau_f, tau_d, tau_s, U, A, dt):        
    for p in range(1, 3):
        # Solve EDOs using Euler method
        r[p][0] = r[p][0] + dt*(-r[p][0]/tau_f[p - 1] + U[p - 1]*(1 - r[p][0])*AP)
        x[p][0] = x[p][0] + dt*((1 - x[p][0])/tau_d[p - 1] - (r[p][0] + U[p - 1]*(1 - r[p][0]))*x[p][0]*AP)
        Is[p][0] = Is[p][0] + dt*(-Is[p][0]/tau_s + A[p - 1]*x[p][0]*(r[p][0] + U[p - 1]*(1 - r[p][0]))*AP)
        
    Ipost = np.sum(Is)
    
    tm_syn_inst = dict()
    tm_syn_inst['r'] = r
    tm_syn_inst['x'] = x
    tm_syn_inst['Is'] = Is
    tm_syn_inst['Ipost'] = Ipost
        
    return tm_syn_inst

def get_frequency(signal, time):
    return int(np.count_nonzero(signal)/time)
# =============================================================================
# GLOBAL PARAMS
# =============================================================================
vp = 30
vr = -65

ms = 1000                                       # 1 second = 1000 milliseconds
sim_time = 1                                    # 1s
dt = 10/ms                                      # 10ms
fs = 1/dt                                       # Hz (Sampling Frequency)
ss_time = 0                                     # s - steady state time to trash later
sim_time_total = int((sim_time + ss_time)*ms)   # ms
sim_steps_aux = np.arange(0, sim_time_total, dt)
sim_steps = sim_steps_aux.reshape(1, len(sim_steps_aux))
chop_till = int(ss_time*ms); 
n_trials = 20

n_D = 1
n_CI = 1
n_TC = 1
total_neurons = n_D + n_CI

spikes_D = np.zeros((1, len(sim_steps_aux)))
spikes_CI = np.zeros((1, len(sim_steps_aux)))

f_thalamus = 8                                  # Hz (Thalamus frequency)

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
PSC_TC = np.zeros((1, len(sim_steps_aux)))

# =============================================================================
# MAKING THALAMIC INPUT
# =============================================================================
I_Thalamus = poissonSpikeGen(sim_time, f_thalamus, dt, 1)
fr_thalamus = get_frequency(I_Thalamus,sim_time_total)

plt.figure(figsize=(10, 6))
for i, spike_train in enumerate(I_Thalamus):
    plt.plot(np.arange(0, sim_time, dt), spike_train, '|')

plt.xlabel('Time (s)')
plt.ylabel('Neuron')
plt.title('Poisson Spike Generator (dt=10ms)')
plt.show()

#plot_voltage('Thalamus', sim_steps[0], I_Thalamus[0])
#plot_raster('Thalamus', I_Thalamus[0], sim_time_total, fr_thalamus, 1, dt)

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
# LOOPING THROUGH
# =============================================================================
for t in range(1, len(sim_steps[0])):
    # D Layer
    for k in range(n_D):
        v_D_aux = 1*v_D[0][t - 1]
        u_D_aux = 1*u_D[0][t - 1]
        AP_D = 0
        
        coupling_D_D = W_D*PSC_D[0][t - 1]/n_D
        coupling_D_CI = W_D_CI*PSC_CI[0][t - 1]/n_CI
        coupling_D_T = W_D_TC*I_Thalamus[0][t - 1]/n_TC
        
        izhikevich_v_D = izhikevich_dvdt(v_D_aux, u_D_aux, I_D[0][0])
        izhikevich_u_D = izhikevich_dudt(v_D_aux, u_D_aux, a_D[0][0], b_D[0][0])
        
        v_D[0][t] = v_D_aux + dt*(izhikevich_v_D + coupling_D_D + coupling_D_CI + coupling_D_T + zeta_noise[t - 1]) 
        u_D[0][t] = u_D_aux + dt*izhikevich_u_D
        
        if (v_D[0][t] >= vp):
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
        v_CI_aux = 1*v_CI[0][t - 1]
        u_CI_aux = 1*u_CI[0][t - 1]
        AP_CI = 0
        
        coupling_CI_CI = W_D_CI*PSC_CI[0][t - 1]/n_CI
        coupling_CI_D = W_CI_D*PSC_D[0][t - 1]/n_D
        coupling_CI_T = W_CI_TC*I_Thalamus[0][t - 1]/n_TC
        
        izhikevich_v_CI = izhikevich_dvdt(v_CI_aux, u_CI_aux, I_CI[0][0])
        izhikevich_u_CI = izhikevich_dudt(v_CI_aux, u_CI_aux, a_CI[0][0], b_CI[0][0]) 
        
        v_CI[0][t] = v_CI_aux +  dt*(izhikevich_v_CI + coupling_CI_CI + coupling_CI_D + coupling_CI_T + zeta_noise[t - 1])
        u_CI[0][t] = u_CI_aux + dt*izhikevich_u_CI
        
        if(v_CI_aux >= vp):
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
plot_voltage('membrane potential - Layer D', sim_steps[0], v_D[0])
plot_voltage('membrane potential - Layer CI', sim_steps[0], v_CI[0])

# =============================================================================
# PLOTS - RASTER PLOTS
# =============================================================================
f_D = get_frequency(spikes_D, sim_time_total)    # Hz
f_CI = get_frequency(spikes_CI, sim_time_total)  # Hz

plot_raster('Layer D', spikes_D, sim_time_total, f_D, n_D, dt)
plot_raster('Layer CI', spikes_CI, sim_time_total, f_CI, n_CI, dt)

# =============================================================================
# PLOTS - PSC = LFP 
# =============================================================================
plot_lfp('LFP - Layer D', sim_steps[0], PSC_D[0])
plot_lfp('LFP - Layer CI', sim_steps[0], PSC_CI[0])

# =============================================================================
# PLOTS - POWER SPECTRAL DENSITY
# =============================================================================
plot_psd_welch('PSD - Layer D', PSC_D, fs)
plot_psd_welch('PSD - Layer CI', PSC_CI, fs)
