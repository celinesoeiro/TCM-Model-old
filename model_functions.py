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
def tm_synapse_eq(r, x, Is, AP, tau_f, tau_d, tau_s, U, A, dt):
    for p in range(1, 3):
        r_aux = r[p - 1]
        x_aux = x[p - 1]
        Is_aux = Is[p - 1]
        # Solve EDOs using Euler method
        # r[p] = r_aux + dt*tm_r_eq(r_aux, tau_f[p - 1], U[p - 1], AP)
        r[p] = r_aux + dt*(-r_aux/tau_f[p - 1] + U[p - 1]*(1 - r_aux)*AP)
        x[p] = x_aux + dt*((1 - x_aux)/tau_d[p - 1] - (r_aux+U[p-1]*(1-r_aux))*x_aux*AP)
        # x[p] = x_aux + dt*tm_x_eq(x_aux, tau_d[p - 1], r_aux, U[p - 1], AP)
        Is[p] = Is_aux + dt*(-Is_aux/tau_s + A[p-1]*x_aux*(r_aux+U[p-1]*(1-r_aux))*AP)
        # Is[p] = Is_aux + dt*tm_I_eq(Is_aux, tau_s, A[p - 1], U[p - 1], x_aux, r_aux, AP)
        
    Ipost = np.sum(Is)
        
    return r, x, Is, Ipost

def tm_synapse_dbs_eq(I_dbs, t_delay, dt, sim_steps, tau_f, tau_d, tau_s, U, A):
    t_vec = t_delay*np.ones((1, sim_steps - 1))
    
    r = np.zeros((3, sim_steps))
    x = np.ones((3, sim_steps))
    Is = np.zeros((3, sim_steps))
    
    for p in range(0,2):
        for i in range(1 + t_delay, sim_steps - 1):
            r_aux = r[p][i - 1]
            x_aux = x[p][i - 1]
            Is_aux = Is[p][i - 1]
            r[p][i] = r_aux + dt*(-r_aux/tau_f[p] + U[p]*(1 - r_aux)*I_dbs[0][i- t_delay])
            x[p][i] = x_aux + dt*((1-x_aux)/tau_d[p] - r[p][i]*x_aux*I_dbs[0][i - t_delay])
            Is[p][i] = Is_aux + dt*(-Is_aux/tau_s + A[p]*r[p][i]*x_aux*I_dbs[0][i - t_delay])
            
    dbs_I = np.sum(Is, axis = 0)
    
    return dbs_I, t_vec

def tm_synapse_poisson_eq(AP_position, sim_steps, t_delay, dt, tau_f, tau_d, tau_s, U, A):
    r = np.zeros((3, sim_steps))
    x = np.zeros((3, sim_steps))
    Is = np.zeros((3, sim_steps))
    spd = np.zeros((1, sim_steps))
    
    spd[0][AP_position] = 1/dt
    
    for p in range(0, 2):    
        for i in range(1 + t_delay, sim_steps - 1):
            r_aux = r[p][i - 1]
            x_aux = x[p][i - 1]
            Is_aux = Is[p][i - 1]
            # Solve EDOs using Euler method
            r[p][i] = r_aux + dt*(-r_aux/tau_f[p] + U[p]*(1 - r_aux)*spd[0][i - t_delay])
            x[p][i] = x_aux + dt*((1 - x_aux)/tau_d[p] - r[p][i]*x_aux*spd[0][i - t_delay])
            Is[p][i] = Is_aux + dt*(-Is_aux/tau_s + A[p]*r_aux*x[p][i]*spd[0][i - t_delay])
        
    Ipost = np.sum(Is, axis=0)
        
    return Ipost

# =============================================================================
# DBS
# =============================================================================
def DBS_delta(f_dbs, dbs_duration, dev, sim_steps, samp_freq, dbs_amplitude, chop_till):
    # This is to define Dirac delta pulses, no membrane current but straight dirac delta pulses that reach PNs:
    T_dbs = np.round(samp_freq/f_dbs)
    dbs = np.arange(0, dbs_duration, T_dbs)
    I_dbs_full = np.zeros((1, dbs_duration))

    for i in dbs:
        I_dbs_full[0][int(i)] = dbs_amplitude 

    if (dev == 1):
        dbs_I = I_dbs_full
    else:
        dbs_I = np.concatenate((
            np.zeros((1, chop_till)), 
            np.zeros((1, int(np.ceil((sim_steps - chop_till)/dev)))), 
            I_dbs_full, 
            np.zeros((1, int(np.ceil((sim_steps - chop_till)/dev))))
            ),axis=1)
        
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
        for t in clean_sim_steps:
            if (spikes[n][t] != 0):
                spike_time.append(int(spikes[n][t]))

        neuron_name = f"neuron_{n}"
        neuron[neuron_name] = np.array(spike_time)
    
    return neuron