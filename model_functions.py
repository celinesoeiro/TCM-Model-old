"""
@author: Celine Soeiro
@description: TCM functions
"""

import math
import numpy as np

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
# def tm_synapse_eq(r, x, Is, AP, tau_f, tau_d, tau_s, U, A, dt):        
#     for p in range(3):
#         # Solve EDOs using Euler method
#         r[p][0] = r[p][0] + dt*(-r[p][0]/tau_f[p - 1] + U[p - 1]*(1 - r[p][0])*AP)
#         x[p][0] = x[p][0] + dt*((1 - x[p][0])/tau_d[p - 1] - (r[p][0] + U[p - 1]*(1 - r[p][0]))*x[p][0]*AP)
#         Is[p][0] = Is[p][0] + dt*(-Is[p][0]/tau_s + A[p - 1]*x[p][0]*(r[p][0] + U[p - 1]*(1 - r[p][0]))*AP)
        
#     Ipost = np.sum(Is)
    
#     tm_syn_inst = dict()
#     tm_syn_inst['r'] = r
#     tm_syn_inst['x'] = x
#     tm_syn_inst['Is'] = Is
#     tm_syn_inst['Ipost'] = Ipost
        
#     return tm_syn_inst

def tm_synapse_eq(u, R, I, AP, t_f, t_d, t_s, U, A, dt, p):
    # Solve EDOs using Euler method
    for j in range(p):
        # u -> utilization factor -> resources ready for use
        u[0][j] = u[0][j - 1] + -dt*u[0][j - 1]/t_f[j] + U[j]*(1 - u[0][j - 1])*AP
        # x -> availabe resources -> Fraction of resources that remain available after neurotransmitter depletion
        R[0][j] = R[0][j - 1] + dt*(1 - R[0][j - 1])/t_d[j - 1] - u[0][j]*R[0][j - 1]*AP
        # PSC
        I[0][j] = I[0][j - 1] + -dt*I[0][j - 1]/t_s + A[j - 1]*R[0][j - 1]*u[0][j - 1]*AP
        
        # print('tm_synapse_eq')
        # print('u = ', u)
        # print('R = ', R)
        # print('I = ', I)
        
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
            u[p][i] = u[p - 1][i - 1] + -dt*u[p - 1][i - 1]/t_f[p - 1] + U[p - 1]*(1 - u[p - 1][i - 1])*ap
            # x -> availabe resources -> Fraction of resources that remain available after neurotransmitter depletion
            R[p][i] = R[p - 1][i - 1] + dt*(1 - R[p - 1][i - 1])/t_d[p - 1] - u[p - 1][i - 1]*R[p - 1][i - 1]*ap
            # PSC
            I[p][i] = I[p - 1][i - 1] + -dt*I[p - 1][i - 1]/t_s + A[p - 1]*R[p - 1][i - 1]*u[p - 1][i - 1]*ap
        
    Ipost = np.sum(I, 0)
        
    return Ipost

def tm_synapse_dbs_eq(I_dbs, t_delay, dt, sim_steps, tau_f, tau_d, tau_s, U, A):    
    r = np.zeros((3, sim_steps))
    x = np.ones((3, sim_steps))
    Is = np.zeros((3, sim_steps))
    
    for p in range(3):
        for i in range(1 + t_delay, sim_steps - 1):
            r[p][i + 1] = r[p][i] + dt*(-r[p][i]/tau_f[p] + U[p]*(1 - r[p][i])*I_dbs[i- t_delay])
            x[p][i + 1] = x[p][i] + dt*((1-x[p][i])/tau_d[p] - r[p][i]*x[p][i]*I_dbs[i - t_delay])
            Is[p][i + 1] = Is[p][i] + dt*(-Is[p][i]/tau_s + A[p]*r[p][i]*x[p][i]*I_dbs[i - t_delay])
            
    dbs_I = np.sum(Is, axis = 0)
    
    return dbs_I.reshape(1,-1)

# =============================================================================
# DBS
# =============================================================================
def I_DBS(sim_steps, chop_till, dt, td_syn, tau_f, tau_d, tau_s, U, A, dbs, samp_freq):    
    I_dbs = np.zeros((2, sim_steps))
    dev = 1 # divide the total simulation time in dev 
    f_dbs = 130

    # Simulate 1/dev of DBS
    if (dbs != 0):
        dev = 3
        
    if (dev == 1):
        print('dbs off')
        dbs_duration = sim_steps
        dbs_amplitude = 0.02
    else:
        print('dbs on')
        dbs_duration = int(np.round((sim_steps - chop_till)/dev))
        dbs_amplitude = 1
    
    T_dbs = np.round(samp_freq/f_dbs)
    dbs_arr = np.arange(0, dbs_duration, T_dbs)
    I_dbs_full = np.zeros((1, dbs_duration))
    
    for i in dbs_arr:
        I_dbs_full[0][int(i)] = dbs_amplitude 
        
    if (dev == 1):
        I_dbs_pre = I_dbs_full
    else:
        I_dbs_pre = np.concatenate((
            np.zeros((1, chop_till)), 
            np.zeros((1, dbs_duration)), 
            I_dbs_full, 
            np.zeros((1, dbs_duration))
            ),axis=1)

    I_dbs_post = tm_synapse_dbs_eq(I_dbs = I_dbs_pre[0], 
                                   t_delay = td_syn, 
                                   dt = dt,
                                   tau_f = tau_f,
                                   tau_d = tau_d,
                                   U = U,
                                   A = A,
                                   tau_s = tau_s,
                                   sim_steps = sim_steps)
    I_dbs[0][:] = I_dbs_pre
    I_dbs[1][:] = I_dbs_post
    
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
    # plt.xlim([0, 2000])
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
    
    plt.figure()
    plt.plot(y)
    plt.title(f'Bandpass filter - ${lowcut} - ${highcut}')
    plt.show()
    
    return y

def PSD(signal, fs):
    (f, S) = welch(signal, fs, nperseg=5*1024)
    
    plt.figure()
    plt.semilogy(f, S)
    plt.ylim([1e-5, 1e5])
    plt.xlim([0, 200])
    # plt.xticks([0,5,10,15,20,25,30,35,40,45,50])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.title('PSD')
    plt.show()
    
    return f, S