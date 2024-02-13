"""
@author: Celine Soeiro
@description: TCM functions
"""

import math
import numpy as np

# =============================================================================
# Poisson Spike Generator
# =============================================================================
def homogeneous_poisson(rate, tmax, bin_size): 
    nbins = np.floor(tmax/bin_size).astype(int) 
    prob_of_spike = rate * bin_size 
    spikes = np.random.rand(nbins) < prob_of_spike 
    return spikes * 1

# =============================================================================
# Izhikevich neuron equations
# =============================================================================
def izhikevich_dvdt(v, u, I):
    return 0.04*v**2 + 5*v + 140 - u + I

def izhikevich_dudt(v, u, a, b):
    return a*(b*v - u)

# =============================================================================
# TM synapse
# =============================================================================
def synapse_utilization(u, tau_f, U, AP, dt):
    return dt*(-u/tau_f) + U*(1 - u)*AP

def synapse_recovery(R, tau_d, u_next, AP, dt, R_prev):
    return dt*((1/tau_d)*(1 - R)) - u_next*R_prev*AP

def synapse_current(I, tau_s, A, R_prev, u_next, AP, dt):
    return dt*(-I/tau_s) + A*R_prev*u_next*AP

def tm_synapse_eq(u, R, I, AP, t_f, t_d, t_s, U, A, dt, p):
    # Solve EDOs using Euler method
    for j in range(p):
        # u -> utilization factor -> resources ready for use
        u[0][j] = u[0][j - 1] + -dt*u[0][j - 1]/t_f[j] + U[j]*(1 - u[0][j - 1])*AP
        # x -> availabe resources -> Fraction of resources that remain available after neurotransmitter depletion
        R[0][j] = R[0][j - 1] + dt*(1 - R[0][j - 1])/t_d[j] - u[0][j]*R[0][j - 1]*AP
        # PSC
        I[0][j] = I[0][j - 1] + -dt*I[0][j - 1]/t_s + A[j]*R[0][j - 1]*u[0][j]*AP
        
    Ipost = np.sum(I)
    
    tm_syn_inst = dict()
    tm_syn_inst['u'] = u
    tm_syn_inst['R'] = R
    tm_syn_inst['I'] = I
    tm_syn_inst['Ipost'] = np.around(Ipost, decimals=6)
        
    return tm_syn_inst

def tm_synapse_poisson_eq(spikes, sim_steps, t_delay, dt, t_f, t_d, t_s, U, A, time):
    R = np.zeros((3, sim_steps))
    u = np.zeros((3, sim_steps))
    I = np.zeros((3, sim_steps))
    
    for p in range(3):    
        for i in time:
            ap = 0
            if (spikes[0][i - 1] != 0):
                ap = 1
            # u -> utilization factor -> resources ready for use
            u[p][i] = u[p][i - 1] + -dt*u[p][i - 1]/t_f[p] + U[p]*(1 - u[p][i - 1])*ap
            # x -> availabe resources -> Fraction of resources that remain available after neurotransmitter depletion
            R[p][i] = R[p][i - 1] + dt*(1 - R[p][i - 1])/t_d[p] - u[p][i - 1]*R[p][i - 1]*ap
            # PSC
            I[p][i] = I[p][i - 1] + -dt*I[p][i - 1]/t_s + A[p]*R[p][i - 1]*u[p][i - 1]*ap
            
        
    Ipost = np.sum(I, 0)
        
    return Ipost

def tm_syn_excit_dep(sim_steps, vr, vp, a, b, c, d, spikes, time, I, W, dt, neuron_type):
    t_f = 17
    t_d = 671
    U = 0.5
    A = 1
    t_s = 3         
    
    v = np.zeros((1, sim_steps)) 
    v[0][0] = vr
    u = np.zeros((1, sim_steps))
    u[0][0] = vr*b

    R_syn = np.zeros((1, sim_steps)) # R for Excitatory Depression
    u_syn = np.zeros((1, sim_steps)) # u for Excitatory Depression
    I_syn = np.zeros((1, sim_steps)) # I for Excitatory Depression
    R_syn[0][0] = 1

    PSC = np.zeros((1, sim_steps))
    AP_neuron = np.zeros((1, sim_steps))

    for t in time:
        AP_syn = spikes[t]
        
        # Synapse var - Excitatory - Depression
        syn_u_aux = 1*u_syn[0][t - 1]
        syn_R_aux = 1*R_syn[0][t - 1]
        syn_I_aux = 1*I_syn[0][t - 1]
        syn_R_prev = 1*R_syn[0][t - 1]
            
        # Synapse - Excitatory - Depression
        syn_du = synapse_utilization(u = syn_u_aux, tau_f = t_f,  U = U, AP = AP_syn, dt = dt)
        u_syn[0][t] = syn_u_aux + syn_du
        
        syn_dR = synapse_recovery(R = syn_R_aux, tau_d = t_d, u_next = 1*u_syn[0][t], AP = AP_syn, dt = dt, R_prev=syn_R_prev)
        R_syn[0][t] = syn_R_aux + syn_dR
        
        syn_dI = synapse_current(I = syn_I_aux, tau_s = t_s, A = A, R_prev = syn_R_prev, u_next = 1*u_syn[0][t], AP = AP_syn, dt = dt)
        I_syn[0][t] = syn_I_aux + syn_dI
        PSC[0][t] = 1*I_syn[0][t]
        
        v_aux = 1*v[0][t - 1]
        u_aux = 1*u[0][t - 1]
        
        # Neuron - FS - Inhibitory
        if (v_aux >= vp):
            AP_neuron[0][t] = t - 1
            v_aux = v[0][t]
            v[0][t] = c
            u[0][t] = u_aux + d
        else:
            dv = izhikevich_dvdt(v = v_aux, u = u_aux, I = W*PSC[0][t - 1])
            du = izhikevich_dudt(v = v_aux, u = u_aux, a = a, b = b)

            v[0][t] = v_aux + dt*dv
            u[0][t] = u_aux + dt*du

    v_value = []
    for ap in AP_neuron[0]:
        if (ap > 1):
            v_value.append(v[0][int(ap) - 1])
            
    # if (len(v_value) != 0):
    #     fig, (ax1,ax2,ax3, ax4) = plt.subplots(4,1,figsize=(15, 15), constrained_layout=True)
    #     ax4.stem(np.arange(0,len(v_value)), v_value)
    #     ax4.plot(np.arange(0,len(v_value)), v_value,  'o--', color='grey', alpha=0.4)
    #     ax4.set_title('Valor de pico da tensao')
    # else:
    #     print('--- No Action Potential')
    fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(15, 15), constrained_layout=True, sharex=True)
    
    fig.suptitle(f'{neuron_type}', fontsize=30)
    ax1.plot(spikes)
    ax2.plot(PSC[0])
    ax3.plot(v[0])

    ax1.set_title('Trem de pulsos gerado por Poisson', fontsize=20)
    ax2.set_title('PSC - Excitatoria dominada por depressao', fontsize=20)
    ax3.set_title('Tensao do neuronio', fontsize=20)
        
def tm_syn_excit_fac(sim_steps, vr, vp, a, b, c, d, time, dt, spikes, I, W, neuron_type):
    t_f = 670
    t_d = 138
    t_s = 3
    U = 0.5
    A = 1
    
    v = np.zeros((1, sim_steps)) 
    v[0][0] = vr
    u = np.zeros((1, sim_steps)) 
    u[0][0] = vr*b

    R_syn = np.zeros((1, sim_steps)) # R for Excitatory Facilitation
    u_syn = np.zeros((1, sim_steps)) # u for Excitatory Facilitation
    I_syn = np.zeros((1, sim_steps)) # I for Excitatory Facilitation
    R_syn[0][0] = 1

    PSC = np.zeros((1, sim_steps))
    AP_neuron = np.zeros((1, sim_steps))

    for t in time:
        AP_syn = spikes[t - 1]

        # Synapse var - Excitatory - Facilitation
        syn_u_aux = 1*u_syn[0][t - 1]
        syn_R_aux = 1*R_syn[0][t - 1]
        syn_I_aux = 1*I_syn[0][t - 1]
        syn_R_prev = 1*R_syn[0][t - 1]
            
        syn_du = synapse_utilization(u = syn_u_aux, tau_f = t_f, U = U, AP = AP_syn, dt = dt)
        u_syn[0][t] = syn_u_aux + syn_du
        
        syn_dR = synapse_recovery(R = syn_R_aux, tau_d = t_d, u_next = 1*u_syn[0][t], AP = AP_syn, dt = dt, R_prev=syn_R_prev)
        R_syn[0][t] = syn_R_aux + syn_dR
        
        syn_dI = synapse_current(I = syn_I_aux, tau_s = t_s, A = A, R_prev = syn_R_prev, u_next = 1*u_syn[0][t], AP = AP_syn,dt = dt)
        I_syn[0][t] = syn_I_aux + syn_dI
        PSC[0][t] = 1*I_syn[0][t]
        
        v_aux = 1*v[0][t - 1]
        u_aux = 1*u[0][t - 1]
        
        # Neuron - RS - Excitatory
        if (v_aux >= vp):
            AP_neuron[0][t] = t - 1
            v_aux = v[0][t]
            v[0][t] = c
            u[0][t] = u_aux + d
        else:
            dv = izhikevich_dvdt(v = v_aux, u = u_aux, I = W*PSC[0][t - 1])
            du = izhikevich_dudt(v = v_aux, u = u_aux, a = a, b = b)

            v[0][t] = v_aux + dt*dv
            u[0][t] = u_aux + dt*du
        
        v_aux = 1*v[0][t - 1]
        u_aux = 1*u[0][t - 1]
       
    v_value = []
    for ap in AP_neuron[0]:
        if (ap > 1):
            print(ap)
            v_value.append(v[0][int(ap)])

    # if (len(v_value) != 0):
    #     fig, (ax1,ax2,ax3, ax4) = plt.subplots(4,1,figsize=(15, 15), constrained_layout=True)
    #     ax4.stem(np.arange(0,len(v_value)), v_value)
    #     ax4.plot(np.arange(0,len(v_value)), v_value,  'o--', color='grey', alpha=0.4)
    #     ax4.set_title('Valor de pico da tensao')
    # else:
    #     print('--- No Action Potential')
    fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(15, 15), constrained_layout=True, sharex=True)
    
    fig.suptitle(f'{neuron_type}', fontsize=30)
    ax1.plot(spikes)
    ax2.plot(PSC[0])
    ax3.plot(v[0])

    ax1.set_title('Trem de pulsos gerado por Poisson', fontsize=20)
    ax2.set_title('PSC - Excitatoria dominada por facilitacao', fontsize=20)
    ax3.set_title('Tensao do neuronio', fontsize=20)

def tm_syn_inib_dep(sim_steps, dt, time, a, b, c, d, vp, vr, spikes, I, W, neuron_type):
    t_f = 21
    t_d = 706
    U = 0.5
    A = 1
    t_s = 11
    
    v = np.zeros((1, sim_steps)) 
    v[0][0] = vr
    u = np.zeros((1, sim_steps))
    u[0][0] = vr*b

    R_syn = np.zeros((1, sim_steps)) # R for Inhibitory Depression
    u_syn = np.zeros((1, sim_steps)) # u for Inhibitory Depression
    I_syn = np.zeros((1, sim_steps)) # I for Inhibitory Depression
    R_syn[0][0] = 1

    PSC = np.zeros((1, sim_steps))
    AP_neuron = np.zeros((1, sim_steps))

    for t in time:
        AP_syn = spikes[t - 1]
        
        # Synapse var - Inhibitory - Depression
        syn_u_aux = 1*u_syn[0][t - 1]
        syn_R_aux = 1*R_syn[0][t - 1]
        syn_I_aux = 1*I_syn[0][t - 1]
        syn_R_prev = 1*R_syn[0][t - 1]
            
        # Synapse - Inhibitory - Depression
        syn_du = synapse_utilization(u = syn_u_aux, tau_f = t_f, U = U, AP = AP_syn, dt = dt)
        u_syn[0][t] = syn_u_aux + syn_du
        
        syn_dR = synapse_recovery(R = syn_R_aux, tau_d = t_d, u_next = 1*u_syn[0][t], AP = AP_syn, dt = dt, R_prev=syn_R_prev)
        R_syn[0][t] = syn_R_aux + syn_dR
        
        syn_dI = synapse_current(I = syn_I_aux, tau_s = t_s, A = A, R_prev=syn_R_prev, u_next = 1*u_syn[0][t], AP = AP_syn, dt = dt)
        I_syn[0][t] = syn_I_aux + syn_dI
        PSC[0][t] = 1*I_syn[0][t]
        
        v_aux = 1*v[0][t - 1]
        u_aux = 1*u[0][t - 1]
        
        # Neuron - RS - Excitatory
        if (v_aux >= vp):
            AP_neuron[0][t] = t - 1
            v_aux = v[0][t]
            v[0][t] = c
            u[0][t] = u_aux + d
        else:
            dv = izhikevich_dvdt(v = v_aux, u = u_aux, I = W*PSC[0][t - 1])
            du = izhikevich_dudt(v = v_aux, u = u_aux, a = a, b = b)
     
            v[0][t] = v_aux + dt*dv
            u[0][t] = u_aux + dt*du
        
    v_value = []
    for ap in AP_neuron[0]:
        if (ap > 1):
            print(ap)
            v_value.append(v[0][int(ap)])
            
    # if (len(v_value) != 0):
    #     fig, (ax1,ax2,ax3, ax4) = plt.subplots(4,1,figsize=(15, 15), constrained_layout=True)
    #     ax4.stem(np.arange(0,len(v_value)), v_value)
    #     ax4.plot(np.arange(0,len(v_value)), v_value,  'o--', color='grey', alpha=0.4)
    #     ax4.set_title('Valor de pico da tensao')
    # else:
    # print('--- No Action Potential')
    fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(15, 15), constrained_layout=True, sharex=True)
    
    fig.suptitle(f'{neuron_type}', fontsize=30)
    ax1.plot(spikes)
    ax2.plot(PSC[0])
    ax3.plot(v[0])

    ax1.set_title('Trem de pulsos gerado por Poisson', fontsize=20)
    ax2.set_title('PSC - Inibitoria dominada por depressao', fontsize=20)
    ax3.set_title('Tensao do neuronio', fontsize=20)

def tm_syn_inib_fac(sim_steps, time, dt, a, b, c, d, vp, vr, I, spikes, W, neuron_type):
    t_f = 376
    t_d = 45
    U = 0.5
    A = 1
    t_s = 11
    
    v = np.zeros((1, sim_steps))
    v[0][0] = vr
    u = np.zeros((1, sim_steps))
    u[0][0] = vr*b

    R_syn = np.zeros((1, sim_steps)) # R for Inhibitory Facilitation
    u_syn = np.zeros((1, sim_steps)) # u for Inhibitory Facilitation
    I_syn = np.zeros((1, sim_steps)) # I for Inhibitory Facilitation
    R_syn[0][0] = 1

    PSC = np.zeros((1, sim_steps))
    AP_neuron = np.zeros((1, sim_steps))

    for t in time:
        AP_syn = spikes[t - 1]
        
        # Synapse var - Inhibitory - Faciliation
        syn_u_aux = 1*u_syn[0][t - 1]
        syn_R_aux = 1*R_syn[0][t - 1]
        syn_I_aux = 1*I_syn[0][t - 1]   
        syn_R_prev = 1*R_syn[0][t - 1]
            
        # Synapse - Inhibitory - Facilitation
        syn_du = synapse_utilization(u = syn_u_aux, tau_f = t_f, U = U, AP=AP_syn, dt = dt)
        u_syn[0][t] = syn_u_aux + syn_du
        
        syn_dR = synapse_recovery(R = syn_R_aux, tau_d = t_d, u_next = 1*u_syn[0][t], AP = AP_syn, dt = dt, R_prev=syn_R_prev)
        R_syn[0][t] = syn_R_aux + syn_dR
        
        syn_dI = synapse_current(I = syn_I_aux, tau_s = t_s, A = A, R_prev=syn_R_prev, u_next = 1*u_syn[0][t], AP = AP_syn, dt = dt)
        I_syn[0][t] = syn_I_aux + syn_dI
        PSC[0][t] = 1*I_syn[0][t]
        
        v_aux = 1*v[0][t - 1]
        u_aux = 1*u[0][t - 1]
        
        # Neuron - RS - Excitatory
        if (v_aux >= vp):
            AP_neuron[0][t] = t - 1
            v_aux = 1*v[0][t]
            v[0][t] = c
            u[0][t] = u_aux + d
        else:
            dv = izhikevich_dvdt(v = v_aux, u = u_aux, I = W*PSC[0][t - 1])
            du = izhikevich_dudt(v = v_aux, u = u_aux, a = a, b = b)
        
            v[0][t] = v_aux + dt*dv
            u[0][t] = u_aux + dt*du
        
    v_value = []
    
    for ap in AP_neuron[0]:
        if (ap > 1):
            print(ap)
            v_value.append(v[0][int(ap)])

    # if (len(v_value) != 0):
    #     fig, (ax1,ax2,ax3, ax4) = plt.subplots(4,1,figsize=(15, 15), constrained_layout=True)
    #     ax4.stem(np.arange(0,len(v_value)), v_value)
    #     ax4.plot(np.arange(0,len(v_value)), v_value,  'o--', color='grey', alpha=0.4)
    #     ax4.set_title('Valor de pico da tensao')
    # else:
    #     print('--- No Action Potential')
    fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(15, 15), constrained_layout=True, sharex=True)
    
    fig.suptitle(f'{neuron_type}', fontsize=30)
    ax1.plot(spikes)
    ax2.plot(PSC[0])
    ax3.plot(v[0])

    ax1.set_title('Trem de pulsos gerado por Poisson', fontsize=20)
    ax2.set_title('PSC - Inibitoria dominada por facilitacao', fontsize=20)
    ax3.set_title('Tensao do neuronio', fontsize=20)

# =============================================================================
# DBS
# =============================================================================
def I_DBS(sim_steps, dt, fs, dbs_freq, td_syn, t_f_E, t_d_E, U_E, t_s_E, A_E):    
    step = int(sim_steps/3) # 1 part is zero, 1 part is dbs and another part is back to zero -> pulse
    
    I_dbs = np.zeros((2, sim_steps))
    f_dbs = dbs_freq
    
    dbs_duration = step
    dbs_amplitude = 1   # 1mA
    
    T_dbs = np.round(fs/f_dbs)
    dbs_arr = np.arange(0, dbs_duration, T_dbs)
    I_dbs_full = np.zeros((1, dbs_duration))
    
    for i in dbs_arr:
        I_dbs_full[0][int(i)] = dbs_amplitude
    
    I_dbs_pre = 1*np.concatenate((
        np.zeros((1, step)), 
        I_dbs_full, 
        np.zeros((1, step))
        ),axis=1)
    
    R_dbs = np.zeros((3, sim_steps))
    u_dbs = np.ones((3, sim_steps))
    Is_dbs = np.zeros((3, sim_steps))
    
    for p in range(3):
        for i in range(td_syn, sim_steps - 1):
            # u -> utilization factor -> resources ready for use
            u_dbs[p][i] = u_dbs[p][i - 1] + -dt*u_dbs[p][i - 1]/t_f_E[p] + U_E[p]*(1 - u_dbs[p][i - 1])*I_dbs_pre[0][i- td_syn]
            # x -> availabe resources -> Fraction of resources that remain available after neurotransmitter depletion
            R_dbs[p][i] = R_dbs[p][i - 1] + dt*(1 - R_dbs[p][i - 1])/t_d_E[p] - u_dbs[p][i - 1]*R_dbs[p][i - 1]*I_dbs_pre[0][i- td_syn]
            # PSC
            Is_dbs[p][i] = Is_dbs[p][i - 1] + -dt*Is_dbs[p][i - 1]/t_s_E + A_E[p]*R_dbs[p][i - 1]*u_dbs[p][i - 1]*I_dbs_pre[0][i- td_syn]
            
    I_dbs_post = np.sum(Is_dbs, 0)
    
    I_dbs[0] = I_dbs_pre[0]
    I_dbs[1] = I_dbs_post
    
    return I_dbs

# =============================================================================
# POISSON
# =============================================================================
def poissonSpikeGen(firing_rate, sim_steps, n_trials, dt):
    n_bins = math.floor(sim_steps/dt)
    spike_mat = np.random.rand(n_trials, n_bins) < firing_rate*dt
    
    time_vector = np.arange(0, sim_steps - dt, dt)
    
    return spike_mat, time_vector

def poisson_spike_generator(num_steps, dt, num_neurons, thalamic_firing_rate, current_value=None):
    # Initialize an array to store spike times for each neuron
    spike_times = [[] for _ in range(num_neurons)]

    # Calculate firing probability
    firing_prob = thalamic_firing_rate * dt  # Calculate firing probability

    # Generate spikes for each neuron using the Poisson distribution
    for t in range(num_steps):
        for neuron_id in range(num_neurons):
            # Generate a random number between 0 and 1
            rand_num = np.random.rand()
            
            # If the random number is less than the firing probability, spike
            if rand_num < firing_prob:
                spike_times[neuron_id].append(t)
            else: 
                spike_times[neuron_id].append(0)
    
    # Creating a vector to be used as current input
    input_current = np.zeros((1, num_steps))
    for sub_spike in spike_times:
        for spike in sub_spike:
            spike_indice = np.array(spike)
            value = np.random.normal(loc=0.25, scale=0.05)
            input_current[0][spike_indice.astype(int)] = value
                
    return spike_times, input_current

# =============================================================================
# RASTER
# =============================================================================
def make_dict(sim_steps, chop_till, n_neurons, fired):
    clean_sim_steps = np.arange(0, sim_steps - chop_till)
    
    new_length = len(clean_sim_steps)*n_neurons
    neuron = np.zeros((new_length, 3))

    n_aux = 0
    t_aux = 0
    for i in range(new_length):
        if (n_aux == n_neurons):
            n_aux = 0
        
        if (t_aux == len(clean_sim_steps)):
            t_aux = 0
            
        neuron[i][0] = n_aux
        neuron[i][1] = t_aux
        neuron[i][2] = fired[n_aux][t_aux]
            
        n_aux += 1
        t_aux +=1
    
    v_dict = {
        "neuron": neuron[:, 0],
        "time": neuron[:, 1],
        "fired": neuron[:, 2],
        }
    
    return v_dict

def export_spike_dict(n_neuron, sim_steps, chop_till, spikes):
    # Creating a dictionary
    clean_sim_steps = np.arange(0, sim_steps - chop_till)
    neuron = {}
    spike_time = []
    for n in range(n_neuron):
        neuron_name = f"neuron_{n}"
        neuron[neuron_name] = []
    
    # Filling the dictionary with the firing time
    for n in range(n_neuron):
        for t in clean_sim_steps:
            if (spikes[n][t] != 0):
                spike_time.append(int(spikes[n][t]))

        neuron_name = f"neuron_{n}"
        neuron[neuron_name] = np.array(spike_time)
    
    return neuron

# =============================================================================
# SIGNAL ANALYSIS
# =============================================================================
from scipy.signal  import butter, lfilter, welch
from math import pi
from matplotlib import pyplot as plt

def LFP(E_signal, I_signal):
    rho = 0.27
    r = 100e-6
    #### LFP is the sum of the post-synaptic currents
    LFP = (np.subtract(E_signal, 1*I_signal))/(4*pi*r*rho)

    plt.figure()
    plt.plot(LFP)
    plt.title('LFP')
    plt.show()

    return LFP

def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

#funçao butter_bandpass_filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    
    return y

def PSD(signal, fs):
    (f, S) = welch(signal, fs, nperseg=10*1024)
    
    return f, S