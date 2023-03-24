# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 14:59:45 2023

@author: Avell
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

sns.set()

from model_parameters import TCM_model_parameters, coupling_matrix_normal

from tr_as_func import tr_cells
from tc_as_func import tc_cells
from ci_as_func import ci_cells
from s_as_func import s_cells

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

n_s = neuron_quantities['S']
n_m = neuron_quantities['M']
n_d = neuron_quantities['D']
n_ci = neuron_quantities['CI']
n_tr = neuron_quantities['TR']
n_tc = neuron_quantities['TC']

W_N = coupling_matrix_normal(    
    facilitating_factor = facilitating_factor_N, 
    n_s = n_s, 
    n_m = n_m, 
    n_d = n_d, 
    n_ci = n_ci, 
    n_tc = n_tc, 
    n_tr = n_tr)['weights']

PSC_S = np.zeros((1,sim_steps))
PSC_M = np.zeros((1,sim_steps))
PSC_D = np.zeros((1,sim_steps))
PSC_TC = np.zeros((1,sim_steps))
PSC_TR = np.zeros((1,sim_steps))
PSC_CI = np.zeros((1,sim_steps))

# =============================================================================
# EQUATIONS
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

def get_parameters(synapse_type: str):
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
    
print("----- Thalamic Reticular Nucleus (TR)")

PSC_TR, I_TR, AP_TR, v_tr, u_tr, r_tr, x_tr = tr_cells(
    time_vector = time, 
    number_neurons = n_tr, 
    simulation_steps = sim_steps, 
    coupling_matrix = W_N, 
    neuron_params = neuron_params['TR1'], 
    current = currents['I_TR_1'], 
    vr = vr, 
    vp = vp, 
    dt = dt, 
    Idc = Idc, 
    dvdt = dvdt, 
    dudt = dudt, 
    r_eq = r_eq, 
    x_eq = x_eq, 
    I_eq = I_eq, 
    synapse_parameters = get_parameters('inhibitory'), 
    PSC_S = PSC_S, 
    PSC_M = PSC_M, 
    PSC_D = PSC_D, 
    PSC_TR = PSC_TR, 
    PSC_TC = PSC_TC, 
    PSC_CI = PSC_CI)

print("----- Thalamo-Cortical Relay Nucleus (TC)")

PSC_TC, AP_TC, v_tc, u_tc, r_tc, x_tc = tc_cells(
    time_vector = time, 
    number_neurons = n_tc, 
    simulation_steps = sim_steps, 
    coupling_matrix = W_N, 
    neuron_params = neuron_params['TC1'], 
    current = currents['I_TC_1'], 
    vr = vr, 
    vp = vp, 
    dt = dt, 
    Idc = Idc, 
    dvdt = dvdt, 
    dudt = dudt, 
    r_eq = r_eq, 
    x_eq = x_eq, 
    I_eq = I_eq, 
    synapse_parameters = get_parameters('excitatory'), 
    PSC_S = PSC_S, 
    PSC_M = PSC_M, 
    PSC_D = PSC_D, 
    PSC_TR = PSC_TR, 
    PSC_TC = PSC_TC, 
    PSC_CI = PSC_CI)

print("----- Cortical Interneurons (CI)")

PSC_CI, AP_CI, v_ci, u_ci, r_ci, x_ci = ci_cells(
    time_vector = time, 
    number_neurons = n_ci, 
    simulation_steps = sim_steps, 
    coupling_matrix = W_N, 
    neuron_params = neuron_params['CI1'], 
    current = currents['I_CI_1'], 
    vr = vr, 
    vp = vp, 
    dt = dt, 
    Idc = Idc, 
    dvdt = dvdt, 
    dudt = dudt, 
    r_eq = r_eq, 
    x_eq = x_eq, 
    I_eq = I_eq, 
    synapse_parameters = get_parameters('excitatory'), 
    PSC_S = PSC_S, 
    PSC_M = PSC_M, 
    PSC_D = PSC_D, 
    PSC_TR = PSC_TR, 
    PSC_TC = PSC_TC, 
    PSC_CI = PSC_CI)

print("----- Superficial layer (S)")

PSC_S, AP_S, v_s, u_s, r_s, x_s = s_cells(
    time_vector = time, 
    number_neurons = n_s, 
    simulation_steps = sim_steps, 
    coupling_matrix = W_N, 
    neuron_params = neuron_params['S1'], 
    current = currents['I_S_1'], 
    vr = vr, 
    vp = vp, 
    dt = dt, 
    Idc = Idc, 
    dvdt = dvdt, 
    dudt = dudt, 
    r_eq = r_eq, 
    x_eq = x_eq, 
    I_eq = I_eq, 
    synapse_parameters = get_parameters('excitatory'), 
    PSC_S = PSC_S, 
    PSC_M = PSC_M, 
    PSC_D = PSC_D, 
    PSC_TR = PSC_TR, 
    PSC_TC = PSC_TC, 
    PSC_CI = PSC_CI)
