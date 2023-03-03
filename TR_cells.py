"""
Created on Mon Feb  6 20:31:49 2023

@author: Celine Soeiro

@description: Thalamic Reticular Nucleus (TR) cells

# Abreviations:
    PSC: Post Synaptic Current
    IC: Intercortical Neurons
    SW: Synaptic Weight
    S: Surface (Supragranular) layer
    M: Middle (Granular) layer
    D: Deep (Infragranular) layer
    CI: Cortical Interneurons
    TR: Thalamic Reticular Nucleus
    TC: Thalamo-Cortical Relay nucleus
    PD: Parkinsonian Desease

Inputs:
    time step: dt
    peak voltage: vp
    rest voltage: vr
    simulation steps: sim_steps
    number of neurons: n
    neuron_params: a,b,c,d                                                          -> Izhikevich
    membrane recovery variable: v                                                   -> Izhikevich
    membrane potential of the neuron: u                                             -> Izhikevich
    available neurotransmitter resources ready to be used: r                        -> TM model (u in original article)
    neurotransmitter resources that remain available after synaptic transmission: x -> TM model
    post-synaptic current: I                                                        -> TM model
    PSC self contribution: PSC_self
    PSC layer S: PSC_S
    PSC layer M: PSC_M
    PSC layer D: PSC_D
    PSC layer TC: PSC_TC
    PSC CI: PSC_CI
    SW from self: SW_self
    SW from S: SW_S
    SW from M: SW_M
    SW from D: SW_D
    SW from CI: SW_CI
    SW from TC: SW_TC 
    bias current: Ib
    time vector: time
    
------------ OVERVIEW

Receive inhibitory stimulus from:
    - Self 

Receive excitatory stimulus from:
    - Thalamo-cortical relay nucleus (TRN)

Send inhibitory stimulus to:
    - Thalamo-cortical relay nucleus (TRN)
    
Send excitatory stimulus to:
    - None
    
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

sns.set()

from model_parameters import TCM_model_parameters, coupling_matrix_normal

# =============================================================================
# INITIAL VALUES
# =============================================================================
global_parameters = TCM_model_parameters()['model_global_parameters']
neuron_quantities = TCM_model_parameters()['neuron_quantities']
neuron_params = TCM_model_parameters()['neuron_paramaters']
currents = TCM_model_parameters()['currents_per_structure']

facilitating_factor_N = global_parameters['connectivity_factor_normal_condition']
facilitating_factor_PD = global_parameters['connectivity_factor_PD_condition']

dt = global_parameters['dt']
sim_steps = global_parameters['simulation_steps']
time = global_parameters['time_vector']
vr = global_parameters['vr']
vp = global_parameters['vp']
chop_till = global_parameters['chop_till']
Idc = global_parameters['Idc']

n = neuron_quantities['TR']
n_s = neuron_quantities['S']
n_m = neuron_quantities['M']
n_d = neuron_quantities['D']
n_ci = neuron_quantities['CI']
n_tc = neuron_quantities['TC']

neuron_params = neuron_params['TR1']

v = vr*np.ones((n,sim_steps))
u = 0*v
r = np.zeros((3,len(time)))
x = np.ones((3,len(time)))
I = np.zeros((3,len(time)))
PSC_self = np.zeros((1,sim_steps))
PSC_S = np.zeros((1,sim_steps))
PSC_M = np.zeros((1,sim_steps))
PSC_D = np.zeros((1,sim_steps))
PSC_TC = np.zeros((1,sim_steps))
PSC_CI = np.zeros((1,sim_steps))

W_N = coupling_matrix_normal(facilitating_factor_N, n_s, n_m, n_d, n_ci, n_tc, n)['weights']

SW_self = W_N['W_II_tr']
SW_S = W_N['W_IE_tr_s']
SW_M = W_N['W_IE_tr_m']
SW_D = W_N['W_IE_tr_d']
SW_TC = W_N['W_IE_tr_tc']
SW_CI = W_N['W_II_tr_ci']

Ib = currents['I_TR_1'] + Idc*np.ones(n)

# =============================================================================
# CALCULATING THE NEW VALUE
# =============================================================================

# Izhikevich neuron equations
def dvdt(v, u, I):
    return 0.04*v**2 + 5*v + 140 - u + I

def dudt(v, u, a, b):
    return a*(b*v - u)

# TM synapse
def r_eq(r, t_f, U, fired):
    # fraction of available neurotransmitter resources ready to be used
    return -(r/t_f) + U*(1 - r)*fired

def x_eq(x, t_d, r, U, fired):
    # fraction of the neurotransmitter resources that remain available after synaptic transmission
    return (1 - x)/t_d - (r + U*(1 - r))*x*fired
    
def I_eq(I, t_s, A, U, x, r, fired):
    # post-synaptic current
    return -(I/t_s) + A*(r + U*(1 - r))*x*fired

def getParamaters(synapse_type: str):
    if (synapse_type == 'excitatory'):
        return {
            # [Facilitating, Depressing, Pseudo-linear]
            't_f': [670, 17, 326],
            't_d': [138, 671, 329],
            'U': [0.09, 0.5, 0.29],
            'distribution': [0.2, 0.63, 0.17],
            't_s': 11,
        };
    elif (synapse_type == 'inhibitory'):
        return {
            # [Facilitating, Depressing, Pseudo-linear]
            't_f': [376, 21, 62],
            't_d': [45, 706, 144],
            'U': [0.016, 0.25, 0.32],
            'distribution': [0.08, 0.75, 0.17],
            't_s': 11,
        };
    
    else:
        return 'Invalid synapse_type. Synapse_type must be excitatory or inhibitory.'
    

AP = np.zeros((1,len(time)))

for t in range(1, len(time)):
    AP_aux = AP[0][t]
    for k in range(1, n):        
        v_aux = v[k - 1][t - 1]
        u_aux = u[k - 1][t - 1]
        
        if (v_aux >= vp):
            AP_aux = 1
            v[k][t] = vp
            v[k][t] = neuron_params['c']
            u[k][t] = u[k][t] + neuron_params['d']
        else:
            neuron_contribution = dvdt(v_aux, u_aux, Ib[k])
            self_feedback = SW_self[k][0]*PSC_self[0][t]/n
            layer_S = SW_S[k][0]*PSC_S[0][t]/n
            layer_M = SW_M[k][0]*PSC_M[0][t]/n
            layer_D = SW_D[k][0]*PSC_D[0][t]/n
            layer_TC = SW_TC[k][0]*PSC_TC[0][t]/n
            layer_CI = SW_CI[k][0]*PSC_CI[0][t]/n
            noise = 0
            
            v[k][t] = v_aux + dt*(neuron_contribution + self_feedback + layer_S + layer_M + layer_D + layer_TC + layer_CI + noise)
            u[k][t] = u_aux + dt*dudt(v_aux, u_aux, neuron_params['a'], neuron_params['b'])
            
        # TM parameters
        tau_f = getParamaters('inhibitory')['t_f']
        tau_d = getParamaters('inhibitory')['t_d']
        U = getParamaters('inhibitory')['U']
        A = getParamaters('inhibitory')['distribution']
        tau_s = getParamaters('inhibitory')['t_s']
        parameters_length = len(tau_f)
        
        # Loop trhough the parameters
        for p in range(1, parameters_length):
            r_aux = r[p - 1][t - 1]
            x_aux = x[p - 1][t - 1]
            I_aux = I[p - 1][t - 1]
            # Solve EDOs using Euler method
            r[p][t] = r_aux + dt*r_eq(r_aux, tau_f[p], U[p], AP_aux)
            x[p][t] = x_aux + dt*x_eq(x_aux, tau_d[p], r_aux, U[p], AP_aux)
            I[p][t] = I_aux + dt*I_eq(I_aux, tau_s, A[p], U[p], x_aux, r_aux, AP_aux)
            
        # Concatenate the final current
        I_post_synaptic = np.concatenate(I, axis=None)
        
        if (np.isnan(v[k][t]) or np.isnan(u[k][t]) or np.isinf(v[k][t]) or np.isinf(u[k][t])):
            print('NaN or inf in t = ', t)
            break

PSC_self = I_post_synaptic
    
# Plotting
# indexes = np.arange(0,40, dtype=object)
# for k in range(n):
#     indexes[k] = "neuron " + str(k)
        
# v_RT = pd.DataFrame(v.transpose())

# sns.stripplot(data=v_RT, palette="deep")
