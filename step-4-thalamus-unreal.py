"""
Created on Sat Feb  3 11:12:30 2024

@author: celinesoeiro
"""

import numpy as np
import pandas as pd
import gc # Garbage Collector
from random import seed, random

seed(1)
random_factor = random()

from model_parameters import TCM_model_parameters, coupling_matrix_normal, coupling_matrix_PD
# from model_functions import izhikevich_dudt, izhikevich_dvdt, tm_synapse_eq, poisson_spike_generator, tm_synapse_poisson_eq
from model_plots import plot_heat_map, layer_raster_plot, plot_voltages, plot_raster

from TR_nucleus import TR_nucleus
from TC_nucleus import TC_nucleus
from S_nucleus import S_nucleus
from M_nucleus import M_nucleus
from D_nucleus import D_nucleus
from CI_nucleus import CI_nucleus

# =============================================================================
# INITIAL VALUES
# =============================================================================
print("-- Initializing the global values")
global_parameters = TCM_model_parameters()['model_global_parameters']
neuron_quantities = TCM_model_parameters()['neuron_quantities']
neuron_per_structure = TCM_model_parameters()['neuron_per_structure']
neuron_params = TCM_model_parameters()['neuron_paramaters']
currents = TCM_model_parameters()['currents_per_structure']
tm_synapse_params_inhibitory = TCM_model_parameters()['tm_synapse_params_inhibitory']
tm_synapse_params_excitatory = TCM_model_parameters()['tm_synapse_params_excitatory']

neuron_types_per_structure = global_parameters['neuron_types_per_structure']

noise = TCM_model_parameters()['noise']

# Neuron quantities
n_S = neuron_quantities['S']
n_M = neuron_quantities['M']
n_D = neuron_quantities['D']
n_CI = neuron_quantities['CI']
n_TR = neuron_quantities['TR']
n_TC = neuron_quantities['TC']
n_Hyper = neuron_quantities['HD']
n_total = neuron_quantities['total']

n_CI_FS = neuron_per_structure['neurons_ci_1']
n_CI_LTS = neuron_per_structure['neurons_ci_2']
n_D_RS = neuron_per_structure['neurons_d_1']
n_D_IB = neuron_per_structure['neurons_d_2']
n_S_RS = neuron_per_structure['neurons_s_1']
n_S_IB = neuron_per_structure['neurons_s_2']

vr = global_parameters['vr']
vp = global_parameters['vp']

facilitating_factor_N = global_parameters['connectivity_factor_normal_condition']
facilitating_factor_PD = global_parameters['connectivity_factor_PD_condition']
td_wl = global_parameters['time_delay_within_layers']
td_bl = global_parameters['time_delay_between_layers']
td_tc = global_parameters['time_delay_thalamus_cortex']
td_ct = global_parameters['time_delay_cortex_thalamus']
td_syn = global_parameters['transmission_delay_synapse']

synapse_initial_values = TCM_model_parameters()['synapse_initial_values']

dt = global_parameters['dt']
sim_time = global_parameters['simulation_time']
T = global_parameters['simulation_time_ms']
# sim_steps = global_parameters['simulation_steps']
# time = global_parameters['time_vector']
# Idc_tune = global_parameters['Idc_tune']

sim_steps = int(sim_time/dt)    # 1 second in miliseconds
time = np.arange(1, sim_steps)

# =============================================================================
# COUPLING MATRIXES
# =============================================================================
# Weight Matrix Normal Condition
Z_N = coupling_matrix_normal(    
    facilitating_factor = facilitating_factor_N, 
    n_s = n_S, 
    n_m = n_M, 
    n_d = n_D, 
    n_ci = n_CI, 
    n_tc = n_TC, 
    n_tr = n_TR)['matrix']

# Weight Matrix Parkinsonian Desease Condition
Z_PD = coupling_matrix_PD(
    facilitating_factor = facilitating_factor_PD, 
    n_s = n_S, 
    n_m = n_M, 
    n_d = n_D, 
    n_ci = n_CI, 
    n_tc = n_TC, 
    n_tr = n_TR)['matrix']

# normalizing Normal coupling matrix
Z_N_norm = Z_N/np.linalg.norm(Z_N)

# normalizing PD coupling matrix
Z_PD_norm = Z_PD/np.linalg.norm(Z_PD)

# =============================================================================
# Graphs - Coupling Matrixes - Normal vs Parkinsonian
# =============================================================================
print("-- Printing the coupling matrixes")

CM_Normal = pd.DataFrame(Z_N_norm, columns=['S', 'M', 'D', 'CI', 'TC', 'TR'])
CM_PD = pd.DataFrame(Z_PD_norm, columns=['S', 'M', 'D', 'CI', 'TC', 'TR'])

plot_heat_map(matrix_normal = CM_Normal, matrix_PD = CM_PD)

# =============================================================================
# NEURON PARAMS
# =============================================================================
a_S = neuron_params['a_S']
b_S = neuron_params['b_S']
c_S = neuron_params['c_S']
d_S = neuron_params['d_S']

a_M = neuron_params['a_M']
b_M = neuron_params['b_M']
c_M = neuron_params['c_M']
d_M = neuron_params['d_M']

a_D = neuron_params['a_D']
b_D = neuron_params['b_D']
c_D = neuron_params['c_D']
d_D = neuron_params['d_D']

a_CI = neuron_params['a_CI']
b_CI = neuron_params['b_CI']
c_CI = neuron_params['c_CI']
d_CI = neuron_params['d_CI']

a_TR = neuron_params['a_TR']
b_TR = neuron_params['b_TR']
c_TR = neuron_params['c_TR']
d_TR = neuron_params['d_TR']

a_TC = neuron_params['a_TC']
b_TC = neuron_params['b_TC']
c_TC = neuron_params['c_TC']
d_TC = neuron_params['d_TC']

# =============================================================================
# SYNAPTIC WEIGHTS AND PARAMETERS
# =============================================================================
# Weight Matrix Normal Condition
W_N = coupling_matrix_normal(    
    facilitating_factor = facilitating_factor_N, 
    n_s = n_S, 
    n_m = n_M, 
    n_d = n_D, 
    n_ci = n_CI, 
    n_tc = n_TC, 
    n_tr = n_TR)['weights']

# Weight Matrix Parkinsonian Desease Condition
W_PD = coupling_matrix_PD(
    facilitating_factor = facilitating_factor_PD, 
    n_s = n_S, 
    n_m = n_M, 
    n_d = n_D, 
    n_ci = n_CI, 
    n_tc = n_TC, 
    n_tr = n_TR)['weights']

# TM Synapse Initial Values
p = 3

# =============================================================================
# NEURON VARIABELS
# =============================================================================
## Post-Synaptic Currents
PSC_S = np.zeros((1, sim_steps))
PSC_M = np.zeros((1, sim_steps))
PSC_D = np.zeros((1, sim_steps))
PSC_CI = np.zeros((1, sim_steps))
PSC_TC = np.zeros((1, sim_steps))
PSC_TR = np.zeros((1, sim_steps))
## Thalamus-Cortex coupling params
PSC_T_D = np.zeros((1, sim_steps)) # from Thalamus to D
PSC_D_T = np.zeros((1, sim_steps)) # from D to Thalamus

## S
AP_S = np.zeros((n_S, sim_steps))

v_S = np.zeros((n_S, sim_steps))
u_S = np.zeros((n_S, sim_steps)) 
for i in range(n_S):    
    v_S[i][0] = vr
    u_S[i][0] = b_S[0][0]*vr

u_S_syn = np.zeros((1, p))
R_S_syn = np.ones((1, p))
I_S_syn = np.zeros((1, p))

del i

## M
AP_M = np.zeros((n_M, sim_steps))

v_M = np.zeros((n_M, sim_steps))
u_M = np.zeros((n_M, sim_steps)) 
for i in range(n_M):    
    v_M[i][0] = vr
    u_M[i][0] = b_M[0][0]*vr

u_M_syn = np.zeros((1, p))
R_M_syn = np.ones((1, p))
I_M_syn = np.zeros((1, p))

del i

## D
AP_D = np.zeros((n_D, sim_steps))

v_D = np.zeros((n_D, sim_steps))
u_D = np.zeros((n_D, sim_steps)) 
for i in range(n_D):    
    v_D[i][0] = vr
    u_D[i][0] = b_D[0][0]*vr

## D - Self
u_D_syn = np.zeros((1, p))
R_D_syn = np.ones((1, p))
I_D_syn = np.zeros((1, p))

del i

## CI
AP_CI = np.zeros((n_CI, sim_steps))

v_CI = np.zeros((n_CI, sim_steps))
u_CI = np.zeros((n_CI, sim_steps)) 
for i in range(n_CI):    
    v_CI[i][0] = vr
    u_CI[i][0] = b_CI[0][0]*vr
    
u_CI_syn = np.zeros((1, p))
R_CI_syn = np.ones((1, p))
I_CI_syn = np.zeros((1, p))

del i

## TC
AP_TC = np.zeros((n_TC, sim_steps))

v_TC = np.zeros((n_TC, sim_steps))
u_TC = np.zeros((n_TC, sim_steps)) 
for i in range(n_TC):    
    v_TC[i][0] = vr
    u_TC[i][0] = b_TC[0][0]*vr
    
u_TC_syn = np.zeros((1, p))
R_TC_syn = np.ones((1, p))
I_TC_syn = np.zeros((1, p))

del i

## TR
AP_TR = np.zeros((n_TR, sim_steps))

v_TR = np.zeros((n_TR, sim_steps))
u_TR = np.zeros((n_TR, sim_steps)) 
for i in range(n_TR):    
    v_TR[i][0] = vr
    u_TR[i][0] = b_TR[0][0]*vr
    
u_TR_syn = np.zeros((1, p))
R_TR_syn = np.ones((1, p))
I_TR_syn = np.zeros((1, p))

# =============================================================================
# MAIN
# =============================================================================
print("-- Running model")
for t in time:
    print('time = ', t)
# =============================================================================
#     TR
# =============================================================================
    print('----------------------------------- TR')
    v_TR, u_TR, PSC_TR, u_TR_syn, I_TR_syn, R_TR_syn = TR_nucleus(t, v_TR, u_TR, AP_TR, PSC_TR, PSC_TC, PSC_CI, PSC_D_T, PSC_M, PSC_S, u_TR_syn, R_TR_syn, I_TR_syn)
# =============================================================================
#     TC
# =============================================================================
    print('----------------------------------- TC')
    v_TC, u_TC, PSC_TC, u_TC_syn, I_TC_syn, R_TC_syn, PSC_T_D = TC_nucleus(t, v_TC, u_TC, AP_TC, PSC_TC, PSC_S, PSC_M, PSC_D_T, PSC_TR, PSC_CI, PSC_T_D, R_TC_syn, u_TC_syn, I_TC_syn)
        
# =============================================================================
#     S
# =============================================================================
    print('----------------------------------- S')
    v_S, u_S, PSC_S, u_S_syn, I_S_syn, R_S_syn = S_nucleus(t, v_S, u_S, AP_S, PSC_S, PSC_M, PSC_D, PSC_CI, PSC_TC, PSC_TR, u_S_syn, R_S_syn, I_S_syn)
        
# =============================================================================
#     M
# =============================================================================
    print('----------------------------------- M')
    v_M, u_M, PSC_M, u_M_syn, I_M_syn, R_M_syn = M_nucleus(t, v_M, u_M, AP_M, PSC_M, PSC_S, PSC_D, PSC_CI, PSC_TC, PSC_TR, u_M_syn, R_M_syn, I_M_syn)
            
# =============================================================================
#     D
# =============================================================================
    print('----------------------------------- D')
    v_D, u_D, PSC_D, u_D_syn, I_D_syn, R_D_syn, PSC_D_T = D_nucleus(t, v_D, u_D, AP_D, PSC_D, PSC_S, PSC_M, PSC_T_D, PSC_CI, PSC_TR, PSC_D_T, u_D_syn, R_D_syn, I_D_syn)
        
# =============================================================================
#     CI
# =============================================================================
    print('----------------------------------- CI')
    v_CI, u_CI, PSC_CI, u_CI_syn, I_CI_syn, R_CI_syn = CI_nucleus(t, v_CI, u_CI, AP_CI, PSC_CI, PSC_D, PSC_M, PSC_S, PSC_TC, PSC_TR, u_CI_syn, R_CI_syn, I_CI_syn)
    
# =============================================================================
# PLOTS
# =============================================================================
print("-- Plotting results")

plot_voltages(n_neurons = n_S, voltage = v_S, title = "v - Layer S", neuron_types = neuron_types_per_structure['S'])
layer_raster_plot(n = n_S, AP = AP_S, sim_steps = sim_steps, layer_name = 'S', dt = dt)
print('APs in S layer = ', np.count_nonzero(AP_S))

plot_voltages(n_neurons = n_M, voltage = v_M, title = "v - Layer M", neuron_types = neuron_types_per_structure['M'])
layer_raster_plot(n = n_M, AP = AP_M, sim_steps = sim_steps, layer_name = 'M', dt = dt)
print('APs in M layer = ', np.count_nonzero(AP_M))

plot_voltages(n_neurons = n_D, voltage = v_D, title = "v - Layer D", neuron_types=neuron_types_per_structure['D'])
layer_raster_plot(n = n_D, AP = AP_D, sim_steps = sim_steps, layer_name = 'D', dt = dt)
print('APs in D layer = ', np.count_nonzero(AP_D))

plot_voltages(n_neurons = n_CI, voltage = v_CI, title = "Layer CI", neuron_types=neuron_types_per_structure['CI'])
layer_raster_plot(n = n_CI, AP = AP_CI, sim_steps = sim_steps, layer_name = 'CI', dt = dt)
print('APS in CI layer = ',np.count_nonzero(AP_CI))

plot_voltages(n_neurons = n_TC, voltage = v_TC, title = "TC", neuron_types=neuron_types_per_structure['TC'])
layer_raster_plot(n = n_TC, AP = AP_TC, sim_steps = sim_steps, layer_name = 'TC', dt = dt)
print('APS in TC layer = ',np.count_nonzero(AP_TC))

plot_voltages(n_neurons = n_TR, voltage = v_TR, title = "TR", neuron_types=neuron_types_per_structure['TR'])
layer_raster_plot(n = n_TR, AP = AP_TR, sim_steps = sim_steps, layer_name = 'TR', dt = dt)
print('APS in TR layer = ',np.count_nonzero(AP_TR))

plot_raster(dbs=0,
            sim_steps=sim_steps,
            sim_time=sim_time,
            dt = dt,
            chop_till = 0, 
            n_TR = n_TR, 
            n_TC = n_TC, 
            n_CI = n_CI, 
            n_D = n_D, 
            n_M = n_M, 
            n_S = n_S, 
            n_total = n_total,
            n_CI_FS = n_CI_FS,
            n_CI_LTS = n_CI_LTS,
            n_D_RS = n_D_RS,
            n_D_IB = n_D_IB,
            n_S_RS = n_S_RS,
            n_S_IB = n_S_IB,
            spike_times_TR = AP_TR, 
            spike_times_TC = AP_TC, 
            spike_times_CI = AP_CI, 
            spike_times_D = AP_D, 
            spike_times_M = AP_M,
            spike_times_S = AP_S
            )

print("-- Done!")