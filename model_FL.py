# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 14:59:45 2023

@author: Avell
"""

import numpy as np
import pandas as pd
import seaborn as sns
import gc # Garbage Collector

sns.set()

from model_parameters import TCM_model_parameters, coupling_matrix_normal, coupling_matrix_PD

from model_functions import izhikevich_dvdt, izhikevich_dudt
from model_functions import tm_synapse_eq, tm_synapse_dbs_eq, DBS_delta

from model_plots import plot_heat_map, plot_voltages

from tr_as_func import tr_cells
# from tc_as_func import tc_cells
# from ci_as_func import ci_cells
# from s_as_func import s_cells
# from m_as_func import m_cells
# from d_as_func import d_cells

# =============================================================================
# INITIAL VALUES
# =============================================================================
print("-- Initializing the global values")

global_parameters = TCM_model_parameters()['model_global_parameters']
neuron_quantities = TCM_model_parameters()['neuron_quantities']
neurons_connected_with_hyperdirect_neurons = TCM_model_parameters()['neurons_connected_with_hyperdirect_neurons']
neuron_params = TCM_model_parameters()['neuron_paramaters']
currents = TCM_model_parameters()['currents_per_structure']
noise = TCM_model_parameters()['noise']
TCM_model = TCM_model_parameters()['model_global_parameters']
random_factor = TCM_model_parameters()['random_factor']
synapse_initial_values = TCM_model_parameters()['synapse_initial_values']

tm_synapse_params_inhibitory = TCM_model_parameters()['tm_synapse_params_inhibitory']
tm_synapse_params_excitatory = TCM_model_parameters()['tm_synapse_params_excitatory']

facilitating_factor_N = global_parameters['connectivity_factor_normal_condition']
facilitating_factor_PD = global_parameters['connectivity_factor_PD_condition']
td_wl = global_parameters['time_delay_within_layers']
td_bl = global_parameters['time_delay_between_layers']
td_tc = global_parameters['time_delay_thalamus_cortex']
td_ct = global_parameters['time_delay_cortex_thalamus']
td_syn = global_parameters['transmission_delay_synapse']

dt = global_parameters['dt']
sim_steps = global_parameters['simulation_steps']
time = global_parameters['time_vector']
vr = global_parameters['vr']
vp = global_parameters['vp']
chop_till = global_parameters['chop_till']
Idc_tune = global_parameters['Idc_tune']
Fs = global_parameters['sampling_frequency']
synaptic_fidelity = global_parameters['synaptic_fidelity']

# Neuron quantities
n_S = neuron_quantities['S']
n_M = neuron_quantities['M']
n_D = neuron_quantities['D']
n_CI = neuron_quantities['CI']
n_TR = neuron_quantities['TR']
n_TC = neuron_quantities['TC']

# Affected neurons
n_TR_affected = neurons_connected_with_hyperdirect_neurons['TR']

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

# TM Synapse
x_TR = synapse_initial_values['x_TR']
r_TR = synapse_initial_values['r_TR']
I_syn_TR = synapse_initial_values['I_syn_TR']

tau_f_I = tm_synapse_params_inhibitory['t_f']
tau_d_I = tm_synapse_params_inhibitory['t_d']
tau_s_I = tm_synapse_params_inhibitory['t_s']
U_I = tm_synapse_params_inhibitory['U']
A_I = tm_synapse_params_inhibitory['distribution']

tau_f_E = tm_synapse_params_excitatory['t_f']
tau_d_E = tm_synapse_params_excitatory['t_d']
tau_s_E = tm_synapse_params_excitatory['t_s']
U_E = tm_synapse_params_excitatory['U']
A_E = tm_synapse_params_excitatory['distribution']

# =============================================================================
# NOISE TERMS
# =============================================================================
# additive white Gaussian noise 
# kisi_S_E = noise['kisi_S_E']
# kisi_M_E = noise['kisi_M_E']
# kisi_D_E = noise['kisi_D_E']
# kisi_CI_I = noise['kisi_CI_I']
# kisi_TC_E = noise['kisi_TC_E']
kisi_TR_I = noise['kisi_TR_I']
# threshold white Gaussian noise
# zeta_S_E = noise['zeta_S_E']
# zeta_M_E = noise['zeta_M_E']
# zeta_D_E = noise['zeta_D_E']
# zeta_CI_I = noise['zeta_CI_I']
# zeta_TC_E = noise['zeta_TC_E']
zeta_TR_I = noise['zeta_TR_I']

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
# WEIGHTS
# =============================================================================
W_TR_self = W_N['W_II_tr']
W_TR_S = W_N['W_IE_tr_s']
W_TR_M = W_N['W_IE_tr_m']
W_TR_D = W_N['W_IE_tr_d']
W_TR_TC = W_N['W_IE_tr_tc']
W_TR_CI = W_N['W_II_tr_ci']

# =============================================================================
# CURRENTS
# =============================================================================
# Bias Currents
I_S = currents['S']
I_M = currents['M']
I_D = currents['D']
I_CI = currents['CI']
I_TR = currents['TR']
I_TC = currents['TC']

# Post Synaptic Currents
I_PSC_S = np.zeros((1, sim_steps))
I_PSC_M = np.zeros((1, sim_steps))
I_PSC_D = np.zeros((1, sim_steps))
I_PSC_TC = np.zeros((1, sim_steps))
I_PSC_TR = np.zeros((1, sim_steps))
I_PSC_CI = np.zeros((1, sim_steps))

# =============================================================================
# VOLTAGES
# =============================================================================

v_TR = vr*np.ones((n_TR, sim_steps))
u_TR = 0*v_TR

v_TC = vr*np.ones((n_TC, sim_steps))
u_TC = 0*v_TC

# v_CI = vr*np.ones((n_CI, sim_steps))
# u_CI = 0*v_CI

# v_S = vr*np.ones((n_S, sim_steps))
# u_S = 0*v_S

# v_M = vr*np.ones((n_M, sim_steps))
# u_M = 0*v_M

# v_D = vr*np.ones((n_D, sim_steps))
# u_D = 0*v_D

# =============================================================================
# NEURON PARAMS
# =============================================================================
# a_S = neuron_params['a_S']
# b_S = neuron_params['b_S']
# c_S = neuron_params['c_S']
# d_S = neuron_params['d_S']

# a_M = neuron_params['a_M']
# b_M = neuron_params['b_M']
# c_M = neuron_params['c_M']
# d_M = neuron_params['d_M']

# a_D = neuron_params['a_D']
# b_D = neuron_params['b_D']
# c_D = neuron_params['c_D']
# d_D = neuron_params['d_D']

# a_CI = neuron_params['a_CI']
# b_CI = neuron_params['b_CI']
# c_CI = neuron_params['c_CI']
# d_CI = neuron_params['d_CI']

a_TR = neuron_params['a_TR']
b_TR = neuron_params['b_TR']
c_TR = neuron_params['c_TR']
d_TR = neuron_params['d_TR']

a_TC = neuron_params['a_TC']
b_TC = neuron_params['b_TC']
c_TC = neuron_params['c_TC']
d_TC = neuron_params['d_TC']
    
# =============================================================================
# DBS
# =============================================================================
I_dbs = np.zeros((2, sim_steps))

f_dbs = 130
dev = 1 # devide the total simulation time in dev sections

if (synaptic_fidelity != 0):
    dev = 3

# for DBS on all the time
if (dev == 1):
    dbs_duration = sim_steps
    dbs_amplitude = 0.02
    
    I_dbs_pre = DBS_delta(f_dbs = f_dbs, 
                          dbs_duration = dbs_duration, 
                          dev = dev, 
                          sim_steps = sim_steps, 
                          Fs=Fs, 
                          dbs_amplitude=dbs_amplitude, 
                          cut=chop_till)
else:
    dbs_duration = (sim_steps - chop_till)/dev
    dbs_amplitude = 1
    
    I_dbs_pre = DBS_delta(f_dbs = f_dbs, 
                          dbs_duration = dbs_duration, 
                          dev = dev, 
                          sim_steps = sim_steps, 
                          Fs=Fs, 
                          dbs_amplitude=dbs_amplitude, 
                          cut=chop_till)

I_dbs_post = tm_synapse_dbs_eq(dbs = I_dbs_pre, 
                               t_delay = td_syn, 
                               dt = dt,
                               tau_f = tau_f_E,
                               tau_d = tau_d_E,
                               U = U_E,
                               A = A_E,
                               tau_s = tau_s_E,
                               sim_steps = sim_steps)

I_dbs[0][:] = I_dbs_pre
I_dbs[1][:] = I_dbs_post[0]

# =============================================================================
# INITIALIZING MODEL
# =============================================================================
        
print("-- Initializing model")

Isi = np.zeros((1,n_TR))
fired = np.zeros((n_TR,sim_steps))

for t in range(1, sim_steps):
    [Ipost, r, x, Is, I_PSC_TR, voltage, u] = tr_cells(
        t = t,
        n_neurons = n_TR, 
        sim_steps = sim_steps,
        voltage = v_TR,
        u = u_TR,
        current = I_TR, 
        a_wg_noise = zeta_TR_I,
        t_wg_noise = kisi_TR_I,
        n_affected = n_TR_affected,
        synaptic_fidelity = synaptic_fidelity,
        I_dbs = I_dbs,
        W_TR_self = W_TR_self,
        W_TR_S = W_TR_S,
        W_TR_M = W_TR_M,
        W_TR_D = W_TR_D,
        W_TR_TC = W_TR_TC,
        W_TR_CI = W_TR_CI,
        I_PSC_S = I_PSC_S,
        I_PSC_M = I_PSC_M,
        I_PSC_D = I_PSC_D,
        I_PSC_TC = I_PSC_TC,
        I_PSC_TR = I_PSC_TR,
        I_PSC_CI = I_PSC_CI,
        td_wl = td_wl,
        td_syn = td_syn,
        td_ct = td_ct,
        td_bl = td_bl,
        a = a_TR,
        b = b_TR,
        c = c_TR,
        d = d_TR,
        r = r_TR,
        x = x_TR,
        Is = I_syn_TR,
        tau_f = tau_f_I,
        tau_d = tau_d_I,
        tau_s = tau_s_I,
        U = U_I,
        A = A_I,
        vr = vr, 
        vp = vp,
        dt = dt,
    )
    
    # print("----- Thalamic Reticular Nucleus (TR) - t = %d" %t)
    # for k in range(0, n_TR):   
    #     AP_aux = 0
    #     v_aux = v_TR[k][t - 1]
    #     u_aux = u_TR[k][t - 1]
    #     I_aux = I_TR[k]
    #     white_gausian_aux = zeta_TR_I[k][t - 1]
        
    #     if (k >= 1 and k <= n_TR_affected):
    #         I_dbss = synaptic_fidelity*I_dbs[1][t - 1]
    #     else:
    #         I_dbss = 0
            
    #     neuron_contribution = izhikevich_dvdt(v = v_aux, u = u_aux, I = I_aux)
    #     self_feedback = W_TR_self[k][0]*I_PSC_TR[0][t - td_wl - td_syn]/n_TR
    #     layer_S = W_TR_S[k][0]*I_PSC_S[0][t - td_ct - td_syn]/n_TR
    #     layer_M = W_TR_M[k][0]*I_PSC_M[0][t - td_ct - td_syn]/n_TR
    #     layer_D = W_TR_D[k][0]*I_PSC_D[0][t - td_ct - td_syn]/n_TR
    #     layer_TC = W_TR_TC[k][0]*I_PSC_TC[0][t - td_bl - td_syn]/n_TR
    #     layer_CI = W_TR_CI[k][0]*I_PSC_CI[0][t - td_ct - td_syn]/n_TR
    #     noise = I_dbss + kisi_TR_I[k][t - 1]
        
    #     v_TR[k][t] = v_aux + dt*(
    #         neuron_contribution + 
    #         self_feedback + 
    #         layer_S + layer_M + layer_D + layer_TC + layer_CI + 
    #         noise
    #         )
    #     u_TR[k][t] = u_aux + dt*izhikevich_dudt(v = v_aux, u = u_aux, a = a_TR[0][k], b = b_TR[0][k])
        
    #     if (v_aux >= (vp + white_gausian_aux)):
    #         AP_aux = 1
    #         v_aux = vp + white_gausian_aux
    #         v_TR[k][t] = c_TR[0][k]
    #         u_TR[k][t] = u_aux + d_TR[0][k]
    #         fired[k][t] = 1
        
    #     [rs, xs, Isyn, Ipost] = tm_synapse_eq(r = r_TR, 
    #                                           x = x_TR, 
    #                                           Is = I_syn_TR, 
    #                                           AP = AP_aux, 
    #                                           tau_f = tau_f_I, 
    #                                           tau_d = tau_d_I, 
    #                                           tau_s = tau_s_I, 
    #                                           U = U_I, 
    #                                           A = A_I,
    #                                           dt = dt)
    #     r_TR = rs
    #     x_TR = xs
    #     I_syn_TR = Isyn
            
    #     Isi[0][k] = Ipost 
        
    I_PSC_TR[0][t] = np.sum(Ipost)
    
    gc.collect()
    
# =============================================================================
# CLEANING THE DATA
# =============================================================================
plot_voltages(n_neurons = n_TR, voltage = v_TR, chop_till = chop_till, sim_steps = sim_steps)

v_TR_clean = np.transpose(v_TR[:,chop_till:sim_steps])
I_PSC_TR_clean = I_PSC_TR[:, chop_till:sim_steps]



# gc.collect()
    
    # PSC_TR, Is, AP, Inhibitory_AP, Inhibitory_aux, r, x
    # v_tr, u_tr, rI, xI, IsI, IPSC_TR = tr_cells(
    #     time_vector = time, 
    #     number_neurons = n_tr, 
    #     simulation_steps = sim_steps, 
    #     coupling_matrix = W_N, 
    #     current = I_TR, 
    #     vr = vr, 
    #     vp = vp, 
    #     dt = dt, 
    #     t = t,
    #     v = v_TR,
    #     u = u_TR,
    #     Idc_tune = Idc_tune, 
    #     dvdt = dvdt, 
    #     dudt = dudt, 
    #     r_eq = r_eq, 
    #     x_eq = x_eq, 
    #     I_eq = I_eq, 
    #     tm_synapse_eq = tm_synapse_eq,
    #     synapse_parameters = tm_synapse_params_inhibitory, 
    #     r = r,
    #     x = x,
    #     Is = Is,
    #     PSC_S = PSC_S[0][t], 
    #     PSC_M = PSC_M[0][t], 
    #     PSC_D = PSC_D[0][t], 
    #     PSC_TR = PSC_TR[0][t], 
    #     PSC_TC = PSC_TC[0][t], 
    #     PSC_CI = PSC_CI[0][t],
    #     neuron_type = "inhibitory",
    #     random_factor = random_factor,
    #     a = a_TR,
    #     b = b_TR,
    #     c = c_TR,
    #     d = d_TR,
    # )
        
    # print("----- Thalamo-Cortical Relay Nucleus (TC)")
    
    # PSC_TC, I_TC, AP_TC, v_tc, u_tc, r_tc, x_tc = tc_cells(
    #     time_vector = time, 
    #     number_neurons = n_tc, 
    #     simulation_steps = sim_steps, 
    #     coupling_matrix = W_N, 
    #     neuron_params = neuron_params['TC1'], 
    #     current = currents['TC'], 
    #     vr = vr, 
    #     vp = vp, 
    #     dt = dt, 
    #     Idc = Idc_tune, 
    #     dvdt = dvdt, 
    #     dudt = dudt, 
    #     r_eq = r_eq, 
    #     x_eq = x_eq, 
    #     I_eq = I_eq, 
    #     tm_synapse_eq = tm_synapse_eq,
    #     synapse_parameters = tm_synapse_params_excitatory, 
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
    
    # PSC_CI, I_CI, AP_CI, v_ci, u_ci, r_ci, x_ci = ci_cells(
    #     time_vector = time, 
    #     number_neurons = n_ci, 
    #     simulation_steps = sim_steps, 
    #     coupling_matrix = W_N, 
    #     neuron_params = neuron_params['CI1'], 
    #     current = currents['CI'], 
    #     vr = vr, 
    #     vp = vp, 
    #     dt = dt, 
    #     Idc = Idc_tune, 
    #     dvdt = dvdt, 
    #     dudt = dudt, 
    #     r_eq = r_eq, 
    #     x_eq = x_eq, 
    #     I_eq = I_eq, 
    #     tm_synapse_eq = tm_synapse_eq,
    #     synapse_parameters = tm_synapse_params_inhibitory, 
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
    
    # PSC_S, I_S, AP_S, v_s, u_s, r_s, x_s = s_cells(
    #     time_vector = time, 
    #     number_neurons = n_s, 
    #     simulation_steps = sim_steps, 
    #     coupling_matrix = W_N, 
    #     neuron_params = neuron_params['S1'], 
    #     current = currents['S'], 
    #     vr = vr, 
    #     vp = vp, 
    #     dt = dt, 
    #     Idc = Idc_tune, 
    #     dvdt = dvdt, 
    #     dudt = dudt, 
    #     r_eq = r_eq, 
    #     x_eq = x_eq, 
    #     I_eq = I_eq, 
    #     tm_synapse_eq = tm_synapse_eq,
    #     synapse_parameters = tm_synapse_params_excitatory, 
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
    
    # PSC_M, I_M, AP_M, v_m, u_m, r_m, x_m = m_cells(
    #     time_vector = time, 
    #     number_neurons = n_m, 
    #     simulation_steps = sim_steps, 
    #     coupling_matrix = W_N, 
    #     neuron_params = neuron_params['M1'], 
    #     current = currents['M'], 
    #     vr = vr, 
    #     vp = vp, 
    #     dt = dt, 
    #     Idc = Idc_tune, 
    #     dvdt = dvdt, 
    #     dudt = dudt, 
    #     r_eq = r_eq, 
    #     x_eq = x_eq, 
    #     I_eq = I_eq, 
    #     tm_synapse_eq = tm_synapse_eq,
    #     synapse_parameters = tm_synapse_params_excitatory, 
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
    
    # PSC_D, I_D, AP_D, v_d, u_d, r_d, x_d = d_cells(
    #     time_vector = time, 
    #     number_neurons = n_d, 
    #     simulation_steps = sim_steps, 
    #     coupling_matrix = W_N, 
    #     neuron_params = neuron_params['D1'], 
    #     current = currents['D'], 
    #     vr = vr, 
    #     vp = vp, 
    #     dt = dt, 
    #     Idc = Idc_tune, 
    #     dvdt = dvdt, 
    #     dudt = dudt, 
    #     r_eq = r_eq, 
    #     x_eq = x_eq, 
    #     I_eq = I_eq, 
    #     tm_synapse_eq = tm_synapse_eq,
    #     synapse_parameters = tm_synapse_params_excitatory, 
    #     PSC_S = PSC_S, 
    #     PSC_M = PSC_M, 
    #     PSC_D = PSC_D, 
    #     PSC_TR = PSC_TR, 
    #     PSC_TC = PSC_TC, 
    #     PSC_CI = PSC_CI,
    #     neuron_type = "excitatory",
    #     random_factor = random_factor
    #     )