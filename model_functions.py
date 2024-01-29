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
        
    Ipost = np.sum(I)
    
    tm_syn_inst = dict()
    tm_syn_inst['u'] = u
    tm_syn_inst['R'] = R
    tm_syn_inst['I'] = I
    tm_syn_inst['Ipost'] = np.around(Ipost, decimals=6)
        
    return tm_syn_inst

def tm_synapse_poisson_eq(AP_position, sim_steps, t_delay, dt, tau_f, tau_d, tau_s, U, A):
    r = np.zeros((3, sim_steps))
    x = np.zeros((3, sim_steps))
    Is = np.zeros((3, sim_steps))
    spd = np.zeros((1, sim_steps))
    
    spd[0][AP_position] = 1/dt
    
    for p in range(3):    
        for i in range(1 + t_delay, sim_steps - 1):
            r[p][i + 1] = r[p][i] + dt*(-r[p][i]/tau_f[p] + U[p]*(1 - r[p][i])*spd[0][i - t_delay])
            x[p][i + 1] = x[p][i] + dt*((1 - x[p][i])/tau_d[p] - r[p][i]*x[p][i]*spd[0][i - t_delay])
            Is[p][i + 1] = Is[p][i] + dt*(-Is[p][i]/tau_s + A[p]*r[p][i]*x[p][i]*spd[0][i - t_delay])
        
    Ipost = np.sum(Is, axis=0)
        
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