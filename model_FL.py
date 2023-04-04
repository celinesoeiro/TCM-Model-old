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

from model_parameters import TCM_model_parameters, coupling_matrix_normal, coupling_matrix_PD

from tr_as_func import tr_cells
from tc_as_func import tc_cells
from ci_as_func import ci_cells
from s_as_func import s_cells
from m_as_func import m_cells
from d_as_func import d_cells

# =============================================================================
# INITIAL VALUES
# =============================================================================
print("-- Initializing the global values")

global_parameters = TCM_model_parameters()['model_global_parameters']
neuron_quantities = TCM_model_parameters()['neuron_quantities']
neuron_params = TCM_model_parameters()['neuron_paramaters']
currents = TCM_model_parameters()['currents_per_structure']
TCM_model = TCM_model_parameters()['model_global_parameters']
random_factor = TCM_model_parameters()['random_factor']

facilitating_factor_N = global_parameters['connectivity_factor_normal_condition']
facilitating_factor_PD = global_parameters['connectivity_factor_PD_condition']

dt = global_parameters['dt']
sim_steps = global_parameters['simulation_steps']
time = global_parameters['time_vector']
vr = global_parameters['vr']
vp = global_parameters['vp']
chop_till = global_parameters['chop_till']
Idc_tune = global_parameters['Idc_tune']

# Neuron quantities
n_s = neuron_quantities['S']
n_m = neuron_quantities['M']
n_d = neuron_quantities['D']
n_ci = neuron_quantities['CI']
n_tr = neuron_quantities['TR']
n_tc = neuron_quantities['TC']

# Weight Matrix Normal Condition
W_N = coupling_matrix_normal(    
    facilitating_factor = facilitating_factor_N, 
    n_s = n_s, 
    n_m = n_m, 
    n_d = n_d, 
    n_ci = n_ci, 
    n_tc = n_tc, 
    n_tr = n_tr)['weights']

# Weight Matrix Parkinsonian Desease Condition
W_PD = coupling_matrix_PD(
    facilitating_factor = facilitating_factor_PD, 
    n_s = n_s, 
    n_m = n_m, 
    n_d = n_d, 
    n_ci = n_ci, 
    n_tc = n_tc, 
    n_tr = n_tr)['weights']

# Post Synaptic Current
PSC_S = np.zeros((1,sim_steps))
PSC_M = np.zeros((1,sim_steps))
PSC_D = np.zeros((1,sim_steps))
PSC_TC = np.zeros((1,sim_steps))
PSC_TR = np.zeros((1,sim_steps))
PSC_CI = np.zeros((1,sim_steps))

# =============================================================================
# COUPLING MATRIXES
# =============================================================================
# Weight Matrix Normal Condition
Z_N = coupling_matrix_normal(    
    facilitating_factor = facilitating_factor_N, 
    n_s = n_s, 
    n_m = n_m, 
    n_d = n_d, 
    n_ci = n_ci, 
    n_tc = n_tc, 
    n_tr = n_tr)['matrix']

# Weight Matrix Parkinsonian Desease Condition
Z_PD = coupling_matrix_PD(
    facilitating_factor = facilitating_factor_PD, 
    n_s = n_s, 
    n_m = n_m, 
    n_d = n_d, 
    n_ci = n_ci, 
    n_tc = n_tc, 
    n_tr = n_tr)['matrix']

# normalizing Normal coupling matrix
Z_N_norm = Z_N/np.linalg.norm(Z_N)

# normalizing PD coupling matrix
Z_PD_norm = Z_PD/np.linalg.norm(Z_PD)

# =============================================================================
# Graphs - Coupling Matrixes - Normal vs Parkinsonian
# =============================================================================
print("-- Printing the coupling matrixes")

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(17,7))

fig.subplots_adjust(wspace=0.3)
fig.suptitle('Matriz de conexão')

CM_Normal = pd.DataFrame(Z_N_norm, columns=['S', 'M', 'D', 'CI', 'TC', 'TR'])
CM_PD = pd.DataFrame(Z_PD_norm, columns=['S', 'M', 'D', 'CI', 'TC', 'TR'])

sns.heatmap(CM_Normal, 
            vmin=-1, vmax=1, 
            yticklabels=['S', 'M', 'D', 'CI', 'TC', 'TR'], 
            annot=True, 
            fmt=".3f", 
            linewidth=.75,
            cmap=sns.color_palette("coolwarm", as_cmap=True),
            ax=ax1,
            )
ax1.set(xlabel="", ylabel="")
ax1.xaxis.tick_top()
ax1.set_title('Condição normal')

sns.heatmap(CM_PD, 
            vmin=-1, vmax=1, 
            yticklabels=['S', 'M', 'D', 'CI', 'TC', 'TR'], 
            annot=True, 
            fmt=".3f", 
            linewidth=.75,
            cmap=sns.color_palette("coolwarm", as_cmap=True),
            ax=ax2,
            )
ax2.set(xlabel="", ylabel="")
ax2.xaxis.tick_top()
ax2.set_title('Condição parkinsoniana')

plt.show()

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
    
print("-- Initializing model")
    
# print("----- Thalamic Reticular Nucleus (TR)")

# PSC_TR, I_TR, AP_TR, v_tr, u_tr, r_tr, x_tr = tr_cells(
#     time_vector = time, 
#     number_neurons = n_tr, 
#     simulation_steps = sim_steps, 
#     coupling_matrix = W_N, 
#     neuron_params = neuron_params['TR1'], 
#     current = currents['I_TR_1'], 
#     vr = vr, 
#     vp = vp, 
#     dt = dt, 
#     Idc = Idc, 
#     dvdt = dvdt, 
#     dudt = dudt, 
#     r_eq = r_eq, 
#     x_eq = x_eq, 
#     I_eq = I_eq, 
#     synapse_parameters = get_parameters('inhibitory'), 
#     PSC_S = PSC_S, 
#     PSC_M = PSC_M, 
#     PSC_D = PSC_D, 
#     PSC_TR = PSC_TR, 
#     PSC_TC = PSC_TC, 
#     PSC_CI = PSC_CI,
#     neuron_type = "inhibitory",
#     random_factor = random_factor
#     )

# print("----- Thalamo-Cortical Relay Nucleus (TC)")

# PSC_TC, AP_TC, v_tc, u_tc, r_tc, x_tc = tc_cells(
#     time_vector = time, 
#     number_neurons = n_tc, 
#     simulation_steps = sim_steps, 
#     coupling_matrix = W_N, 
#     neuron_params = neuron_params['TC1'], 
#     current = currents['I_TC_1'], 
#     vr = vr, 
#     vp = vp, 
#     dt = dt, 
#     Idc = Idc, 
#     dvdt = dvdt, 
#     dudt = dudt, 
#     r_eq = r_eq, 
#     x_eq = x_eq, 
#     I_eq = I_eq, 
#     synapse_parameters = get_parameters('excitatory'), 
#     PSC_S = PSC_S, 
#     PSC_M = PSC_M, 
#     PSC_D = PSC_D, 
#     PSC_TR = PSC_TR, 
#     PSC_TC = PSC_TC, 
#     PSC_CI = PSC_CI,
#     neuron_type = "excitatory",
#     random_factor = random_factor
#     )

# print("----- Cortical Interneurons (CI)")

# PSC_CI, AP_CI, v_ci, u_ci, r_ci, x_ci = ci_cells(
#     time_vector = time, 
#     number_neurons = n_ci, 
#     simulation_steps = sim_steps, 
#     coupling_matrix = W_N, 
#     neuron_params = neuron_params['CI1'], 
#     current = currents['I_CI_1'], 
#     vr = vr, 
#     vp = vp, 
#     dt = dt, 
#     Idc = Idc, 
#     dvdt = dvdt, 
#     dudt = dudt, 
#     r_eq = r_eq, 
#     x_eq = x_eq, 
#     I_eq = I_eq, 
#     synapse_parameters = get_parameters('inhibitory'), 
#     PSC_S = PSC_S, 
#     PSC_M = PSC_M, 
#     PSC_D = PSC_D, 
#     PSC_TR = PSC_TR, 
#     PSC_TC = PSC_TC, 
#     PSC_CI = PSC_CI,
#     neuron_type = "inhibitory",
#     random_factor = random_factor
#     )

# print("----- Superficial layer (S)")

# PSC_S, AP_S, v_s, u_s, r_s, x_s = s_cells(
#     time_vector = time, 
#     number_neurons = n_s, 
#     simulation_steps = sim_steps, 
#     coupling_matrix = W_N, 
#     neuron_params = neuron_params['S1'], 
#     current = currents['I_S_1'], 
#     vr = vr, 
#     vp = vp, 
#     dt = dt, 
#     Idc = Idc, 
#     dvdt = dvdt, 
#     dudt = dudt, 
#     r_eq = r_eq, 
#     x_eq = x_eq, 
#     I_eq = I_eq, 
#     synapse_parameters = get_parameters('excitatory'), 
#     PSC_S = PSC_S, 
#     PSC_M = PSC_M, 
#     PSC_D = PSC_D, 
#     PSC_TR = PSC_TR, 
#     PSC_TC = PSC_TC, 
#     PSC_CI = PSC_CI,
#     neuron_type = "excitatory",
#     random_factor = random_factor
#     )

# print("----- Middle layer (M)")

# PSC_M, AP_M, v_m, u_m, r_m, x_m = m_cells(
#     time_vector = time, 
#     number_neurons = n_m, 
#     simulation_steps = sim_steps, 
#     coupling_matrix = W_N, 
#     neuron_params = neuron_params['M1'], 
#     current = currents['I_M_1'], 
#     vr = vr, 
#     vp = vp, 
#     dt = dt, 
#     Idc = Idc, 
#     dvdt = dvdt, 
#     dudt = dudt, 
#     r_eq = r_eq, 
#     x_eq = x_eq, 
#     I_eq = I_eq, 
#     synapse_parameters = get_parameters('excitatory'), 
#     PSC_S = PSC_S, 
#     PSC_M = PSC_M, 
#     PSC_D = PSC_D, 
#     PSC_TR = PSC_TR, 
#     PSC_TC = PSC_TC, 
#     PSC_CI = PSC_CI,
#     neuron_type = "excitatory",
#     random_factor = random_factor
#     )

# print("----- Deep layer (D)")

# PSC_D, AP_D, v_d, u_d, r_d, x_d = d_cells(
#     time_vector = time, 
#     number_neurons = n_d, 
#     simulation_steps = sim_steps, 
#     coupling_matrix = W_N, 
#     neuron_params = neuron_params['D1'], 
#     current = currents['I_D_1'], 
#     vr = vr, 
#     vp = vp, 
#     dt = dt, 
#     Idc = Idc, 
#     dvdt = dvdt, 
#     dudt = dudt, 
#     r_eq = r_eq, 
#     x_eq = x_eq, 
#     I_eq = I_eq, 
#     synapse_parameters = get_parameters('excitatory'), 
#     PSC_S = PSC_S, 
#     PSC_M = PSC_M, 
#     PSC_D = PSC_D, 
#     PSC_TR = PSC_TR, 
#     PSC_TC = PSC_TC, 
#     PSC_CI = PSC_CI,
#     neuron_type = "excitatory",
#     random_factor = random_factor
#     )