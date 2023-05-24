# -*- coding: utf-8 -*-
"""
Created on Fri May 12 16:43:26 2023

@author: Avell
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
def tm_r_eq(r, t_f, U, fired):
    # fraction of available neurotransmitter resources ready to be used
    return -(r/t_f) + U*(1 - r)*fired

def tm_x_eq(x, t_d, r, U, fired):
    # fraction of the neurotransmitter resources that remain available after synaptic transmission
    return (1 - x)/t_d - (r + U*(1 - r))*x*fired
    
def tm_I_eq(I, t_s, A, U, x, r, fired):
    # post-synaptic current
    return -(I/t_s) + A*(r + U*(1 - r))*x*fired

def tm_synapse_eq(r, x, Is, AP, tau_f, tau_d, tau_s, U, A, dt):
    for p in range(1, 3):
        r_aux = r[p - 1]
        x_aux = x[p - 1]
        Is_aux = Is[p - 1]
        # Solve EDOs using Euler method
        r[p] = r_aux + dt*tm_r_eq(r_aux, tau_f[p - 1], U[p - 1], AP)
        x[p] = x_aux + dt*tm_x_eq(x_aux, tau_d[p - 1], r_aux, U[p - 1], AP)
        Is[p] = Is_aux + dt*tm_I_eq(Is_aux, tau_s, A[p - 1], U[p - 1], x_aux, r_aux, AP)
        
    Ipost = np.sum(Is)
        
    return r, x, Is, Ipost

def tm_synapse_dbs_eq(dbs, t_delay, dt, sim_steps, tau_f, tau_d, tau_s, U, A):
    t_vec = t_delay*np.ones((1, sim_steps))
    
    r = np.zeros((3, sim_steps))
    x = np.ones((3, sim_steps))
    Is = np.zeros((3, sim_steps))
    
    for p in range(1,3):
        for q in range(1, len(t_vec)):
            r_aux = r[q - 1]
            x_aux = x[q - 1]
            Is_aux = Is[q - 1]
            r[p][q] = r_aux + dt*tm_r_eq(r_aux, tau_f[p - 1], U[p - 1], dbs[q - t_delay])
            x[p][q] = x_aux + dt*tm_x_eq(x_aux, tau_d[p - 1], r_aux, U[p - 1], dbs[q - t_delay])
            Is[p][q] = Is_aux + dt*tm_I_eq(Is_aux, tau_s, A[p - 1], U[p - 1], x_aux, r_aux, dbs[q - t_delay])
    
    dbs_I = np.sum(Is, axis = 0)
    
    return dbs_I, t_vec

def tm_synapse_poisson_eq(AP_position, sim_steps, t_delay, dt, tau_f, tau_d, tau_s, U, A):
    r = np.zeros((3, sim_steps))
    x = np.zeros((3, sim_steps))
    Is = np.zeros((3, sim_steps))
    spd = np.zeros((3, sim_steps))
    
    for p in range(1, 3):
        spd[p][AP_position] = 1/dt;
        for i in range(1, sim_steps - 1):
            j = t_delay + i
            r_aux = r[p - 1][j - 1]
            x_aux = x[p - 1][j - 1]
            Is_aux = Is[p - 1][j - 1]
            # Solve EDOs using Euler method
            r[p][j] = r_aux + dt*tm_r_eq(r_aux, tau_f[p - 1], U[p - 1], spd[p][i])
            x[p][j] = x_aux + dt*tm_x_eq(x_aux, tau_d[p - 1], r_aux, U[p - 1], spd[p][i])
            Is[p][j] = Is_aux + dt*tm_I_eq(Is_aux, tau_s, A[p - 1], U[p - 1], x_aux, r_aux, spd[p][i])
        
    Ipost = np.sum(Is, axis=0)
        
    return r, x, Is, Ipost

# =============================================================================
# DBS
# =============================================================================
def DBS_delta(f_dbs, dbs_duration, dev, sim_steps, Fs, dbs_amplitude, cut):
    # This is to define Dirac delta pulses, no membrane current but straight dirac delta pulses that reach PNs:
    T_dbs = Fs/f_dbs
    dbs = np.arange(0, dbs_duration, np.round(T_dbs))
    I_dbs_full = np.zeros((1, dbs_duration))
    
    for i in range(len(dbs)):
        I_dbs_full[0][i] = dbs_amplitude 
    
    if (dev == 1):
        dbs_I = I_dbs_full
    else:
        dbs_I = [np.zeros((1, cut)), np.zeros((1, (sim_steps - cut)/dev)), I_dbs_full, np.zeros((1, (sim_steps - cut)/dev))]
        
    return dbs_I

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
        for t in range(len(clean_sim_steps)):
            if (spikes[n][t] != 0):
                spike_time.append(int(spikes[n][t]))

        neuron_name = f"neuron_{n}"
        neuron[neuron_name] = np.array(spike_time)
    
    return neuron