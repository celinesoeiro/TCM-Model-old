"""
Created on Sun Jan 28 18:35:18 2024

@author: celinesoeiro
"""
import random
import numpy as np
import pandas as pd
import gc # Garbage Collector
from random import seed, random

from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

seed(1)
random_factor = random()

from model_parameters import TCM_model_parameters, coupling_matrix_normal, coupling_matrix_PD
from model_functions import izhikevich_dudt, izhikevich_dvdt, tm_synapse_eq, poisson_spike_generator
from model_plots import plot_heat_map, layer_raster_plot, plot_voltages

# =============================================================================
# INITIAL VALUES
# =============================================================================
print("-- Initializing the global values")
global_parameters = TCM_model_parameters()['model_global_parameters']
neuron_quantities = TCM_model_parameters()['neuron_quantities']
neuron_params = TCM_model_parameters()['neuron_paramaters']
currents = TCM_model_parameters()['currents_per_structure']
tm_synapse_params_inhibitory = TCM_model_parameters()['tm_synapse_params_inhibitory']
tm_synapse_params_excitatory = TCM_model_parameters()['tm_synapse_params_excitatory']

neuron_types_per_structure = global_parameters['neuron_types_per_structure']

# Neuron quantities
n_S = neuron_quantities['S']
n_M = neuron_quantities['M']
n_D = neuron_quantities['D']
n_CI = neuron_quantities['CI']
n_TR = neuron_quantities['TR']
n_TC = neuron_quantities['TC']
n_Hyper = neuron_quantities['HD']
n_total = neuron_quantities['total']

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
a = [0.02, 0.02, 0.1, 0.02, 0.02, 0.02]
b = [0.2,  0.2,  0.2, 0.25, 0.25, 0.25]
c = [-65,  -55,  -65, -65,   -65,  -65]
d = [8,    4,      2,   2,  0.05, 2.05]

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

I_S = currents['S']
I_M = currents['M']
I_D = currents['D']
I_CI = currents['CI']

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

# S
W_S_self = W_N['W_EE_s']
W_S_M = W_N['W_EE_s_m']
W_S_D = W_N['W_EE_s_d']
W_S_CI = W_N['W_EI_s_ci']
W_S_TR = W_N['W_EI_s_tr']
W_S_TC = W_N['W_EE_s_tc']

#M
W_M_self = W_N['W_EE_m']
W_M_S = W_N['W_EE_m_s']
W_M_D = W_N['W_EE_m_d']
W_M_CI = W_N['W_EI_m_ci']
W_M_TR = W_N['W_EI_m_tr']
W_M_TC = W_N['W_EE_m_tc']

# CI
W_CI_self = W_N['W_II_ci']
W_CI_S = W_N['W_IE_ci_s']
W_CI_M = W_N['W_IE_ci_m']
W_CI_D = W_N['W_IE_ci_d']
W_CI_TR = W_N['W_II_ci_tr']
W_CI_TC = W_N['W_IE_ci_tc']

# D
W_D_self = W_N['W_EE_d']
W_D_S = W_N['W_EE_d_s']
W_D_M = W_N['W_EE_d_m']
W_D_CI = W_N['W_EI_d_ci']
W_D_TR = W_N['W_EI_d_tr']
W_D_TC = W_N['W_EE_d_tc']

t_f_I = tm_synapse_params_inhibitory['t_f']
t_d_I = tm_synapse_params_inhibitory['t_d']
t_s_I = tm_synapse_params_inhibitory['t_s']
U_I = tm_synapse_params_inhibitory['U']
A_I = tm_synapse_params_inhibitory['distribution']

t_f_E = tm_synapse_params_excitatory['t_f']
t_d_E = tm_synapse_params_excitatory['t_d']
t_s_E = tm_synapse_params_excitatory['t_s']
U_E = tm_synapse_params_excitatory['U']
A_E = tm_synapse_params_excitatory['distribution']
A_E_T_D = tm_synapse_params_excitatory['distribution_T_D']
A_E_D_T = tm_synapse_params_excitatory['distribution_D_T']

# TM Synapse Initial Values
p = 3

# =============================================================================
# NEURON VARIABELS
# =============================================================================
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

PSC_S = np.zeros((1, sim_steps))

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

PSC_M = np.zeros((1, sim_steps))

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

PSC_D = np.zeros((1, sim_steps))

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

PSC_CI = np.zeros((1, sim_steps))

[spike_T, I_T] = poisson_spike_generator(num_steps = sim_steps, dt = dt, num_neurons = 1, thalamic_firing_rate = 20, current_value=None)

# =============================================================================
# MAIN
# =============================================================================
print("-- Running model")
for t in time:
# =============================================================================
#     S
# =============================================================================
    for s in range(n_S):
        v_S_aux = 1*v_S[s][t - 1]
        u_S_aux = 1*u_S[s][t - 1]
        AP_S_aux = 0
                
        if (v_S_aux >= vp):
            AP_S_aux = 1
            AP_S[s][t] = t
            v_S_aux = v_S[s][t]
            v_S[s][t] = c_S[0][s]
            u_S[s][t] = u_S_aux + d_S[0][s]
        else:
            AP_S[s][t] = 0
            AP_S_aux = 0
            
            # Self feedback - Inhibitory
            coupling_S_S = W_S_self[s][0]*1*PSC_S[0][t - td_wl - td_syn - 1]
            # Coupling S to M - Excitatory 
            coupling_S_M = W_S_M[s][0]*1*PSC_M[0][t - td_wl - td_syn - 1]
            # Coupling S to D - Excitatory 
            coupling_S_D = W_S_D[s][0]*1*PSC_D[0][t - td_wl - td_syn - 1]
            # Coupling S to CI - Inhibitory 
            coupling_S_CI = W_S_CI[s][0]*1*PSC_CI[0][t - td_wl - td_syn - 1]
            # Coupling S to T - Excitatory
            coupling_S_T = W_S_TC[s][0]*I_T[0][t - td_tc - td_syn - 1]
            
            dv_S = izhikevich_dvdt(v = v_S_aux, u = u_S_aux, I = I_S[s])
            du_S = izhikevich_dudt(v = v_S_aux, u = u_S_aux, a = a_S[0][s], b = b_S[0][s])
        
            v_S[s][t] = v_S_aux + dt*(dv_S + coupling_S_S + coupling_S_M + coupling_S_D + coupling_S_CI + coupling_S_T)
            u_S[s][t] = u_S_aux + dt*du_S
            
        # Synapse - Within cortex  
        syn_S = tm_synapse_eq(u = u_S_syn, 
                              R = R_S_syn, 
                              I = I_S_syn, 
                              AP = AP_S_aux, 
                              t_f = t_f_E, 
                              t_d = t_d_E, 
                              t_s = t_s_E, 
                              U = U_E, 
                              A = A_E, 
                              dt = dt, 
                              p = p)
        
        R_S_syn = 1*syn_S['R']
        u_S_syn = 1*syn_S['u']
        I_S_syn = 1*syn_S['I']
        PSC_S[0][t] = 1*syn_S['Ipost']
        
# =============================================================================
#     M
# =============================================================================
    for m in range(n_M):
        v_M_aux = 1*v_M[m][t - 1]
        u_M_aux = 1*u_M[m][t - 1]
        AP_M_aux = 0
                
        if (v_M_aux >= vp):
            AP_M_aux = 1
            AP_M[m][t] = t
            v_M_aux = v_M[m][t]
            v_M[m][t] = c_M[0][m]
            u_M[m][t] = u_M_aux + d_M[0][m]
        else:
            AP_M[m][t] = 0
            AP_M_aux = 0
            
            # Self feedback - Inhibitory
            coupling_M_M = W_M_self[m][0]*1*PSC_M[0][t - td_wl - td_syn - 1]
            # Coupling M to S - Excitatory 
            coupling_M_S = W_M_S[m][0]*1*PSC_S[0][t - td_wl - td_syn - 1]
            # Coupling M to D - Excitatory 
            coupling_M_D = W_M_D[m][0]*1*PSC_D[0][t - td_wl - td_syn - 1]
            # Coupling M to CI - Inhibitory 
            coupling_M_CI = W_M_CI[m][0]*1*PSC_CI[0][t - td_wl - td_syn - 1]
            # Coupling M to T - Excitatory
            coupling_M_T = W_M_TC[m][0]*I_T[0][t - td_tc - td_syn - 1]
            
            dv_M = izhikevich_dvdt(v = v_M_aux, u = u_M_aux, I = I_M[m])
            du_M = izhikevich_dudt(v = v_M_aux, u = u_M_aux, a = a_M[0][m], b = b_M[0][m])
        
            v_M[m][t] = v_M_aux + dt*(dv_M + coupling_M_M + coupling_M_S + coupling_M_D + coupling_M_CI + coupling_M_T)
            u_M[m][t] = u_M_aux + dt*du_M
            
        # Synapse - Within cortex  
        syn_M = tm_synapse_eq(u = u_M_syn, 
                              R = R_M_syn, 
                              I = I_M_syn, 
                              AP = AP_M_aux, 
                              t_f = t_f_E, 
                              t_d = t_d_E, 
                              t_s = t_s_E, 
                              U = U_E, 
                              A = A_E, 
                              dt = dt, 
                              p = p)
        
        R_M_syn = 1*syn_M['R']
        u_M_syn = 1*syn_M['u']
        I_M_syn = 1*syn_M['I']
        PSC_M[0][t] = 1*syn_M['Ipost']
            
# =============================================================================
#     D
# =============================================================================
    for d in range(n_D):
        v_D_aux = 1*v_D[d][t - 1]
        u_D_aux = 1*u_D[d][t - 1]
        AP_D_aux = 0
                
        if (v_D_aux >= vp):
            AP_D_aux = 1
            AP_D[d][t] = t
            v_D_aux = v_D[d][t]
            v_D[d][t] = c_D[0][d]
            u_D[d][t] = u_D_aux + d_D[0][d]
        else:
            AP_D[d][t] = 0
            AP_D_aux = 0
            
            # Self feedback - Inhibitory
            coupling_D_D = W_D_self[d][0]*1*PSC_D[0][t - td_wl - td_syn - 1]
            # Coupling D to S - Excitatory 
            coupling_D_S = W_D_S[d][0]*1*PSC_S[0][t - td_wl - td_syn - 1]
            # Coupling D to M - Excitatory 
            coupling_D_M = W_D_M[d][0]*1*PSC_M[0][t - td_wl - td_syn - 1]
            # Coupling D to CI - Inhibitory 
            coupling_D_CI = W_D_CI[d][0]*1*PSC_CI[0][t - td_wl - td_syn - 1]
            # Coupling D to T - Excitatory
            coupling_D_T = W_D_TC[d][0]*I_T[0][t - td_tc - td_syn - 1]
            
            dv_D = izhikevich_dvdt(v = v_D_aux, u = u_D_aux, I = I_D[d])
            du_D = izhikevich_dudt(v = v_D_aux, u = u_D_aux, a = a_D[0][d], b = b_D[0][d])
        
            v_D[d][t] = v_D_aux + dt*(dv_D + coupling_D_S + coupling_D_M + coupling_D_D + coupling_D_CI + coupling_D_T)
            u_D[d][t] = u_D_aux + dt*du_D
            
        # Synapse - Within cortex  
        syn_D = tm_synapse_eq(u = u_D_syn, 
                              R = R_D_syn, 
                              I = I_D_syn, 
                              AP = AP_D_aux, 
                              t_f = t_f_E, 
                              t_d = t_d_E, 
                              t_s = t_s_E, 
                              U = U_E, 
                              A = A_E, 
                              dt = dt, 
                              p = p)
        
        R_D_syn = 1*syn_D['R']
        u_D_syn = 1*syn_D['u']
        I_D_syn = 1*syn_D['I']
        PSC_D[0][t] = 1*syn_D['Ipost']
        
# =============================================================================
#     CI
# =============================================================================
    for ci in range(n_CI):
        v_CI_aux = 1*v_CI[ci][t - 1]
        u_CI_aux = 1*u_CI[ci][t - 1]
        AP_C_aux = 0
                
        if (v_CI_aux >= vp):
            AP_CI[ci][t] = t
            AP_C_aux = 1
            v_CI_aux = v_CI[ci][t]
            v_CI[ci][t] = c_CI[0][ci]
            u_CI[ci][t] = u_CI_aux + d_CI[0][ci]
        else:
            AP_C_aux = 0
            AP_CI[ci][t] = 0
            
            # Self feeback - Inhibitory
            coupling_CI_CI = W_CI_self[ci][0]*PSC_CI[0][t - td_wl - td_syn - 1]
            # Coupling CI to S - Inhibitory
            coupling_CI_S = W_CI_S[ci][0]*PSC_S[0][t - td_wl - td_syn - 1]
            # Coupling CI to M - Inhibitory
            coupling_CI_M = W_CI_M[ci][0]*PSC_M[0][t - td_wl - td_syn - 1]
            # Coupling CI to D - Inhibitory
            coupling_CI_D = W_CI_D[ci][0]*PSC_D[0][t - td_wl - td_syn - 1]
            # Coupling CI to T - Inhibitory
            coupling_CI_T = W_CI_TC[ci][0]*I_T[0][t - td_tc - td_syn - 1]
            
            dv_CI = izhikevich_dvdt(v = v_CI_aux, u = u_CI_aux, I = I_CI[ci])
            du_CI = izhikevich_dudt(v = v_CI_aux, u = u_CI_aux, a = a_CI[0][ci], b = b_CI[0][ci])
        
            v_CI[ci][t] = v_CI_aux + dt*(dv_CI + coupling_CI_M + coupling_CI_S + coupling_CI_CI + coupling_CI_D + coupling_CI_T)
            u_CI[ci][t] = u_CI_aux + dt*du_CI
            
        # Synapse        
        syn_CI = tm_synapse_eq(u = u_CI_syn, 
                              R = R_CI_syn, 
                              I = I_CI_syn, 
                              AP = AP_C_aux, 
                              t_f = t_f_I, 
                              t_d = t_d_I, 
                              t_s = t_s_I, 
                              U = U_I, 
                              A = A_I, 
                              dt = dt, 
                              p = p)
        
        R_CI_syn = 1*syn_CI['R']
        u_CI_syn = 1*syn_CI['u']
        I_CI_syn = 1*syn_CI['I']
        PSC_CI[0][t] = 1*syn_CI['Ipost']
    
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

print("-- Done!")