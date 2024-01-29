"""
Created on Sun Jan 28 18:35:18 2024

@author: celinesoeiro
"""
import random
import numpy as np
import pandas as pd
import gc # Garbage Collector

from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

random.seed(0)

from model_parameters import TCM_model_parameters, coupling_matrix_normal, coupling_matrix_PD
from model_functions import izhikevich_dudt, izhikevich_dvdt, tm_synapse_eq
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

# Neuron quantities
n_S = neuron_quantities['S']
n_M = neuron_quantities['M']
n_D = neuron_quantities['D']
n_CI = neuron_quantities['CI']
n_TR = neuron_quantities['TR']
n_TC = neuron_quantities['TC']
n_Hyper = neuron_quantities['HD']
n_total = neuron_quantities['total']

facilitating_factor_N = global_parameters['connectivity_factor_normal_condition']
facilitating_factor_PD = global_parameters['connectivity_factor_PD_condition']
td_wl = global_parameters['time_delay_within_layers']
td_bl = global_parameters['time_delay_between_layers']
td_tc = global_parameters['time_delay_thalamus_cortex']
td_ct = global_parameters['time_delay_cortex_thalamus']
td_syn = global_parameters['transmission_delay_synapse']

synapse_initial_values = TCM_model_parameters()['synapse_initial_values']

dt = global_parameters['dt']
sim_steps = global_parameters['simulation_steps']
time = global_parameters['time_vector']
vr = global_parameters['vr']
vp = global_parameters['vp']
chop_till = global_parameters['chop_till']
Idc_tune = global_parameters['Idc_tune']
samp_freq = global_parameters['sampling_frequency']
T = global_parameters['simulation_time_ms']
sim_time = global_parameters['simulation_time']

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
a_D = neuron_params['a_D']
b_D = neuron_params['b_D']
c_D = neuron_params['c_D']
d_D = neuron_params['d_D']

a_CI = neuron_params['a_CI']
b_CI = neuron_params['b_CI']
c_CI = neuron_params['c_CI']
d_CI = neuron_params['d_CI']

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

# CI
W_CI_D = W_N['W_IE_ci_d']

# D
W_D_CI = W_N['W_EI_d_ci']


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
#D D
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

PSC_D = np.zeros((n_D, sim_steps))

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

PSC_CI = np.zeros((n_CI, sim_steps))

# =============================================================================
# MAIN
# =============================================================================
print("-- Running model")
for t in time:
    for d in range(n_D):
        v_D_aux = 1*v_D[d][t - 1]
        u_D_aux = 1*u_D[d][t - 1]
                
        if (v_D_aux >= vp):
            AP_D[d][t] = 1
            v_D_aux = v_D[d][t]
            v_D[d][t] = c_D[0][d]
            u_D[d][t] = u_D_aux + d_D[0][d]
        else:
            AP_D[d][t] = 0
            dv_D = izhikevich_dvdt(v = v_D_aux, u = u_D_aux, I = I_D[d])
            du_D = izhikevich_dudt(v = v_D_aux, u = u_D_aux, a = a_D[0][d], b = b_D[0][d])
        
            v_D[d][t] = v_D_aux + dt*dv_D
            u_D[d][t] = u_D_aux + dt*du_D
            
        # Synapse        
        syn_D = tm_synapse_eq(u = u_D_syn, 
                              R = R_D_syn, 
                              I = I_D_syn, 
                              AP = AP_D[d][t], 
                              t_f = t_f_E, 
                              t_d = t_d_E, 
                              t_s = t_s_E, 
                              U = U_E, 
                              A = A_E, 
                              dt = dt, 
                              p = p)
        
        R_D_syn = syn_D['R']
        u_D_syn = syn_D['u']
        I_D_syn = syn_D['I']
        PSC_D[d][t] = syn_D['Ipost']
        
    for ci in range(n_CI):
        v_CI_aux = 1*v_CI[ci][t - 1]
        u_CI_aux = 1*u_CI[ci][t - 1]
                
        if (v_CI_aux >= vp):
            AP_CI[ci][t] = 1
            v_CI_aux = v_CI[ci][t]
            v_CI[ci][t] = c_CI[0][ci]
            u_CI[ci][t] = u_CI_aux + d_CI[0][ci]
        else:
            AP_CI[ci][t] = 0
            dv_CI = izhikevich_dvdt(v = v_CI_aux, u = u_CI_aux, I = I_CI[ci])
            du_CI = izhikevich_dudt(v = v_CI_aux, u = u_CI_aux, a = a_CI[0][ci], b = b_CI[0][ci])
        
            v_CI[ci][t] = v_CI_aux + dt*dv_CI
            u_CI[ci][t] = u_CI_aux + dt*du_CI
            
        # Synapse        
        syn_CI = tm_synapse_eq(u = u_CI_syn, 
                              R = R_CI_syn, 
                              I = I_CI_syn, 
                              AP = AP_CI[ci][t], 
                              t_f = t_f_I, 
                              t_d = t_d_I, 
                              t_s = t_s_I, 
                              U = U_I, 
                              A = A_I, 
                              dt = dt, 
                              p = p)
        
        R_CI_syn = syn_CI['R']
        u_CI_syn = syn_CI['u']
        I_CI_syn = syn_CI['I']
        PSC_CI[ci][t] = syn_CI['Ipost']
    
    
print("-- Plotting results")
plot_voltages(n_neurons = n_D, voltage = v_D, title = "Layer D")
layer_raster_plot(n = n_D, AP = AP_D, sim_steps = sim_steps, layer_name = 'D')

plot_voltages(n_neurons = n_CI, voltage = v_CI, title = "Layer CI")
layer_raster_plot(n = n_CI, AP = AP_CI, sim_steps = sim_steps, layer_name = 'CI')


print("-- Done!")