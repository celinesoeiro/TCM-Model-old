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

from model_functions import tm_synapse_dbs_eq, DBS_delta, tm_synapse_poisson_eq, poissonSpikeGen

from model_plots import plot_heat_map, plot_voltages, plot_raster

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
neuron_per_structure = TCM_model_parameters()['neuron_per_structure']
neurons_connected_with_hyperdirect_neurons = TCM_model_parameters()['neurons_connected_with_hyperdirect_neurons']
neuron_params = TCM_model_parameters()['neuron_paramaters']
currents = TCM_model_parameters()['currents_per_structure']
noise = TCM_model_parameters()['noise']
TCM_model = TCM_model_parameters()['model_global_parameters']
synapse_initial_values = TCM_model_parameters()['synapse_initial_values']
dbs_modes = TCM_model_parameters()['dbs']

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
T = global_parameters['simulation_time_ms']
sim_time = global_parameters['simulation_time']

# Neuron quantities
n_S = neuron_quantities['S']
n_M = neuron_quantities['M']
n_D = neuron_quantities['D']
n_CI = neuron_quantities['CI']
n_TR = neuron_quantities['TR']
n_TC = neuron_quantities['TC']
n_Hyper = neuron_quantities['HD']
n_total = neuron_quantities['total']

# Quantity of neurons of different types per structure (FS, LTS, RS, IB)
n_CI_FS = neuron_per_structure['neurons_ci_1']
n_CI_LTS = neuron_per_structure['neurons_ci_2']
n_D_RS = neuron_per_structure['neurons_d_1']
n_D_IB = neuron_per_structure['neurons_d_2']
n_S_RS = neuron_per_structure['neurons_s_1']
n_S_IB = neuron_per_structure['neurons_s_2']

# Affected neurons
n_TR_affected = neurons_connected_with_hyperdirect_neurons['TR']
n_TC_affected = neurons_connected_with_hyperdirect_neurons['TC']
n_CI_affected = neurons_connected_with_hyperdirect_neurons['CI']
n_S_affected = neurons_connected_with_hyperdirect_neurons['S']
n_M_affected = neurons_connected_with_hyperdirect_neurons['M']

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

x_TC = synapse_initial_values['x_TC']
r_TC = synapse_initial_values['r_TC']
I_syn_TC = synapse_initial_values['I_syn_TC']

x_CI = synapse_initial_values['x_CI']
r_CI = synapse_initial_values['r_CI']
I_syn_CI = synapse_initial_values['I_syn_CI']

x_D = synapse_initial_values['x_D']
r_D = synapse_initial_values['r_D']
I_syn_D = synapse_initial_values['I_syn_D']

x_M = synapse_initial_values['x_M']
r_M = synapse_initial_values['r_M']
I_syn_M = synapse_initial_values['I_syn_M']

x_S = synapse_initial_values['x_S']
r_S = synapse_initial_values['r_S']
I_syn_S = synapse_initial_values['I_syn_S']

x_D_F = synapse_initial_values['x_D_F']
r_D_F = synapse_initial_values['r_D_F']
I_syn_D_F = synapse_initial_values['I_syn_D_F']

x_D_TC = synapse_initial_values['x_D_TC']
r_D_TC = synapse_initial_values['r_D_TC']
I_syn_D_TC = synapse_initial_values['I_syn_D_TC']

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
A_E_D = tm_synapse_params_excitatory['distribution_D']
A_E_D_F = tm_synapse_params_excitatory['distribution_D_F']

# =============================================================================
# NOISE TERMS
# =============================================================================
# additive white Gaussian noise 
kisi_S_E = noise['kisi_S_E']
kisi_M_E = noise['kisi_M_E']
kisi_D_E = noise['kisi_D_E']
kisi_CI_I = noise['kisi_CI_I']
kisi_TC_E = noise['kisi_TC_E']
kisi_TR_I = noise['kisi_TR_I']
# threshold white Gaussian noise
zeta_S_E = noise['zeta_S_E']
zeta_M_E = noise['zeta_M_E']
zeta_D_E = noise['zeta_D_E']
zeta_CI_I = noise['zeta_CI_I']
zeta_TC_E = noise['zeta_TC_E']
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
print("-- Initializing model variables")
W_TR_self = W_N['W_II_tr']
W_TR_S = W_N['W_IE_tr_s']
W_TR_M = W_N['W_IE_tr_m']
W_TR_D = W_N['W_IE_tr_d']
W_TR_TC = W_N['W_IE_tr_tc']
W_TR_CI = W_N['W_II_tr_ci']

W_TC_self = W_N['W_EE_tc']
W_TC_S = W_N['W_EE_tc_s']
W_TC_M = W_N['W_EE_tc_m']
W_TC_D = W_N['W_EE_tc_d']
W_TC_TR = W_N['W_EI_tc_tr']
W_TC_CI = W_N['W_EI_tc_ci']

W_CI_self = W_N['W_II_ci']
W_CI_S = W_N['W_IE_ci_s']
W_CI_M = W_N['W_IE_ci_m']
W_CI_D = W_N['W_IE_ci_d']
W_CI_TR = W_N['W_II_ci_tr']
W_CI_TC = W_N['W_IE_ci_tc']

W_S_self = W_N['W_EE_s']
W_S_CI = W_N['W_EI_s_ci']
W_S_M = W_N['W_EE_s_m']
W_S_D = W_N['W_EE_s_d']
W_S_TR = W_N['W_EI_s_tr']
W_S_TC = W_N['W_EE_s_tc']

W_M_self = W_N['W_EE_m']
W_M_S = W_N['W_EE_m_s']
W_M_D = W_N['W_EE_m_d']
W_M_CI = W_N['W_EI_m_ci']
W_M_TR = W_N['W_EI_m_tr']
W_M_TC = W_N['W_EE_m_tc']

W_D_self = W_N['W_EE_d']
W_D_S = W_N['W_EE_d_s']
W_D_M = W_N['W_EE_d_m']
W_D_CI = W_N['W_EI_d_ci']
W_D_TR = W_N['W_EI_d_tr']
W_D_TC = W_N['W_EE_d_tc']

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
PSC_S = np.zeros((1, sim_steps))
PSC_M = np.zeros((1, sim_steps))
PSC_D = np.zeros((1, sim_steps))
PSC_TC = np.zeros((1, sim_steps))
PSC_TR = np.zeros((1, sim_steps))
PSC_CI = np.zeros((1, sim_steps))
PSC_D_F = np.zeros((1, sim_steps))
PSC_D_TC = np.zeros((1, sim_steps))
PSC_D_D = np.zeros((1, sim_steps))

# =============================================================================
# VOLTAGES
# =============================================================================
v_TR = vr*np.ones((n_TR, sim_steps))
u_TR = 0*v_TR

v_TC = vr*np.ones((n_TC, sim_steps))
u_TC = 0*v_TC

v_CI = vr*np.ones((n_CI, sim_steps))
u_CI = 0*v_CI

v_S = vr*np.ones((n_S, sim_steps))
u_S = 0*v_S

v_M = vr*np.ones((n_M, sim_steps))
u_M = 0*v_M

v_D = vr*np.ones((n_D, sim_steps))
u_D = 0*v_D

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
# POISSONIAN BACKGROUND ACTIVITY
# =============================================================================
w_ps = 1
I_ps_S = np.zeros((2, sim_steps))
I_ps_M = np.zeros((2, sim_steps))
I_ps_D = np.zeros((2, sim_steps))
I_ps_CI = np.zeros((2, sim_steps))
I_ps_TR = np.zeros((2, sim_steps))
I_ps_TC = np.zeros((2, sim_steps))

if (w_ps != 0):
    W_ps = w_ps*np.random.rand(6,2)
    for l in range(5):
        poisson_firing = 20 + 2*np.random.randn()
        [poisson_spikes, poisson_time_vector] = poissonSpikeGen(poisson_firing, T/1000, 1, dt/1000)
        
    for i in range (len(poisson_time_vector)):
        fired = poisson_spikes[0][i]

        if (fired == True):
            tm_syn_E = tm_synapse_poisson_eq(i, sim_steps, td_syn, dt, tau_f_E, tau_d_E, tau_s_E, U_E, A_E)
            tm_syn_I = tm_synapse_poisson_eq(i, sim_steps, td_syn, dt, tau_f_I, tau_d_I, tau_s_I, U_I, A_I)

            I_ps_S[0][i] = W_ps[0][0]*tm_syn_E[3][i]
            I_ps_S[1][i] = W_ps[0][1]*tm_syn_I[3][i]
            I_ps_M[0][i] = W_ps[1][0]*tm_syn_E[3][i]
            I_ps_M[1][i] = W_ps[1][1]*tm_syn_I[3][i]
            I_ps_D[0][i] = W_ps[2][0]*tm_syn_E[3][i]
            I_ps_D[1][i] = W_ps[2][1]*tm_syn_I[3][i]
            I_ps_CI[0][i] = W_ps[3][0]*tm_syn_E[3][i]
            I_ps_CI[1][i] = W_ps[3][1]*tm_syn_I[3][i]
            I_ps_TR[0][i] = W_ps[4][0]*tm_syn_E[3][i]
            I_ps_TR[1][i] = W_ps[4][1]*tm_syn_I[3][i]
            I_ps_TC[0][i] = W_ps[5][0]*tm_syn_E[3][i]
            I_ps_TC[1][i] = W_ps[5][1]*tm_syn_I[3][i]
            
# =============================================================================
# FIRED NEURONS
# =============================================================================
fired_TR = np.zeros((n_TR, sim_steps))
fired_TC = np.zeros((n_TC, sim_steps))
fired_CI = np.zeros((n_CI, sim_steps))
fired_D = np.zeros((n_D, sim_steps))
fired_M = np.zeros((n_M, sim_steps))
fired_S = np.zeros((n_S, sim_steps))

spike_times_TR = np.zeros((n_TR, sim_steps))
spike_times_TC = np.zeros((n_TC, sim_steps))
spike_times_CI = np.zeros((n_CI, sim_steps))
spike_times_D = np.zeros((n_D, sim_steps))
spike_times_M = np.zeros((n_M, sim_steps))
spike_times_S = np.zeros((n_S, sim_steps))

# =============================================================================
# INITIALIZING MODEL
# =============================================================================
Isi = np.zeros((1,n_TR))
fired = np.zeros((n_TR,sim_steps))

for dbs in dbs_modes:
    print(f'-- Running the model for dbs = {dbs}')
    # Impact of DBS on other cortical structures via D PNs axons
    syn_fid_CI = dbs
    syn_fid_D = dbs
    syn_fid_M = 0*dbs
    syn_fid_S = dbs
    syn_fid_TC = dbs
    syn_fid_TR = dbs
    
    # =============================================================================
    # DBS
    # =============================================================================
    I_dbs = np.zeros((2, sim_steps))

    f_dbs = 130
    dev = 1 # devide the total simulation time in dev sections

    if (dbs != 0):
        dev = 3

    # for DBS on all the time
    if (dev == 1):
        dbs_duration = sim_steps
        dbs_amplitude = 0.02
    else:
        dbs_duration = int(np.round((sim_steps - chop_till)/dev))
        dbs_amplitude = 1
        
    I_dbs_pre = DBS_delta(f_dbs, 
                          dbs_duration, 
                          dev, 
                          sim_steps, 
                          Fs, 
                          dbs_amplitude, 
                          chop_till)

    I_dbs_post = tm_synapse_dbs_eq(I_dbs = I_dbs_pre, 
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

    for t in range(1, sim_steps):
        # TR
        [r_I, x_I, I_syn_I, PSC_TR, v_TR, u_TR, fired_TR] = tr_cells(
            t = t,
            n_neurons = n_TR, 
            sim_steps = sim_steps,
            voltage = v_TR,
            u = u_TR,
            current = I_TR, 
            a_wg_noise = zeta_TR_I,
            t_wg_noise = kisi_TR_I,
            poisson_background_E = I_ps_TR[0],
            poisson_background_I  = I_ps_TR[1],
            n_affected = n_TR_affected,
            synaptic_fidelity = dbs,
            I_dbs = syn_fid_TR*I_dbs,
            W_TR = W_TR_self,
            W_S = W_TR_S,
            W_M = W_TR_M,
            W_D = W_TR_D,
            W_TC = W_TR_TC,
            W_CI = W_TR_CI,
            PSC_S = PSC_S,
            PSC_M = PSC_M,
            PSC_D = PSC_D_F,
            PSC_TC = PSC_TC,
            PSC_TR = PSC_TR,
            PSC_CI = PSC_CI,
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
            fired = fired_TR,
            spikes = spike_times_TR,
        )
        
        r_TR = r_I; x_TR = x_I; I_syn_TR = I_syn_I;
            
        # TC    
        [r_E, x_E, I_syn_E, PSC_TC, v_TC, u_TC, fired_TC, r_d, x_d, I_syn_d, PSC_D_TC] = tc_cells(
            t = t,
            n_neurons = n_TC, 
            sim_steps = sim_steps,
            voltage = v_TC,
            u = u_TC,
            current = I_TC, 
            a_wg_noise = zeta_TC_E,
            t_wg_noise = kisi_TC_E,
            poisson_background_E = I_ps_TC[0],
            poisson_background_I  = I_ps_TC[1],
            n_affected = n_TC_affected,
            synaptic_fidelity = dbs,
            I_dbs = syn_fid_TC*I_dbs,
            W_S = W_TC_S,
            W_M = W_TC_M,
            W_D = W_TC_D,
            W_TR = W_TC_TR,
            W_TC = W_TC_self,
            W_CI = W_TC_CI,
            PSC_S = PSC_S,
            PSC_M = PSC_M,
            PSC_D = PSC_D_F,
            PSC_TC = PSC_TC,
            PSC_TR = PSC_TR,
            PSC_CI = PSC_CI,
            td_wl = td_wl,
            td_syn = td_syn,
            td_ct = td_ct,
            td_bl = td_bl,
            a = a_TC,
            b = b_TC,
            c = c_TC,
            d = d_TC,
            r = r_TC,
            x = x_TC,
            Is = I_syn_TC,
            tau_f = tau_f_E,
            tau_d = tau_d_E,
            tau_s = tau_s_E,
            U = U_E,
            A = A_E,
            vr = vr, 
            vp = vp,
            dt = dt,
            r_D = r_D_TC,
            x_D = x_D_TC,
            I_syn_D = I_syn_D_TC,
            tau_f_D = tau_f_E,
            tau_d_D = tau_d_E,
            tau_s_D = tau_s_E,
            U_D = U_E,
            A_D = A_E_D,
            fired = fired_TC,
            spikes = spike_times_TC
        )
        
        r_TC = r_E; x_TC = x_E; I_syn_TC = I_syn_E;
        r_D_TC = r_d; x_D_TC = x_d; I_syn_D_TC = I_syn_d;
        
        # CI
        [r_I, x_I, I_syn_I, PSC_CI, v_CI, u_CI, fired_CI] = ci_cells(
            t = t,
            n_neurons = n_CI, 
            sim_steps = sim_steps,
            voltage = v_CI,
            u = u_CI,
            current = I_CI, 
            a_wg_noise = zeta_CI_I,
            t_wg_noise = kisi_CI_I,
            poisson_background_E = I_ps_CI[0],
            poisson_background_I  = I_ps_CI[1],
            n_affected = n_CI_affected,
            synaptic_fidelity = dbs,
            I_dbs = syn_fid_CI*I_dbs,
            W_S = W_CI_S,
            W_M = W_CI_M,
            W_D = W_CI_D,
            W_TR = W_CI_TR,
            W_TC = W_CI_TC,
            W_CI = W_CI_self,
            PSC_S = PSC_S,
            PSC_M = PSC_M,
            PSC_D = PSC_D,
            PSC_TC = PSC_TC,
            PSC_TR = PSC_TR,
            PSC_CI = PSC_CI,
            td_wl = td_wl,
            td_syn = td_syn,
            td_ct = td_ct,
            td_bl = td_bl,
            td_tc = td_tc,
            a = a_CI,
            b = b_CI,
            c = c_CI,
            d = d_CI,
            r = r_CI,
            x = x_CI,
            Is = I_syn_CI,
            tau_f = tau_f_I,
            tau_d = tau_d_I,
            tau_s = tau_s_I,
            U = U_I,
            A = A_I,
            vr = vr, 
            vp = vp,
            dt = dt,
            fired = fired_CI,
            spikes = spike_times_CI
        )
        
        r_CI = r_I; x_CI = x_I; I_syn_CI = I_syn_I;
        
        # S
        [r_E, x_E, I_syn_E, PSC_S, v_S, u_S, fired_S] = s_cells(
            t = t,
            n_neurons = n_S, 
            sim_steps = sim_steps,
            voltage = v_S,
            u = u_S,
            current = I_S, 
            a_wg_noise = zeta_S_E,
            t_wg_noise = kisi_S_E,
            poisson_background_E = I_ps_S[0],
            poisson_background_I  = I_ps_S[1],
            n_affected = n_S_affected,
            synaptic_fidelity = dbs,
            I_dbs = syn_fid_S*I_dbs,
            W_S = W_S_self,
            W_M = W_S_M,
            W_D = W_S_D,
            W_TR = W_S_TR,
            W_TC = W_S_TC,
            W_CI = W_S_CI,
            PSC_S = PSC_S,
            PSC_M = PSC_M,
            PSC_D = PSC_D,
            PSC_TC = PSC_TC,
            PSC_TR = PSC_TR,
            PSC_CI = PSC_CI,
            td_wl = td_wl,
            td_syn = td_syn,
            td_ct = td_ct,
            td_bl = td_bl,
            td_tc = td_tc,
            a = a_S,
            b = b_S,
            c = c_S,
            d = d_S,
            r = r_S,
            x = x_S,
            Is = I_syn_S,
            tau_f = tau_f_E,
            tau_d = tau_d_E,
            tau_s = tau_s_E,
            U = U_E,
            A = A_E,
            vr = vr, 
            vp = vp,
            dt = dt,
            fired = fired_S,
            spikes = spike_times_S
        )
        
        r_S = r_E; x_S = x_E; I_syn_S = I_syn_E;
        
        # M
        [r_E, x_E, I_syn_E, PSC_M, v_M, u_M, fired_M] = m_cells(
            t = t,
            n_neurons = n_M, 
            sim_steps = sim_steps,
            voltage = v_M,
            u = u_M,
            current = I_M, 
            a_wg_noise = zeta_M_E,
            t_wg_noise = kisi_M_E,
            poisson_background_E = I_ps_M[0],
            poisson_background_I  = I_ps_M[1],
            n_affected = n_M_affected,
            synaptic_fidelity = dbs,
            I_dbs = syn_fid_M*I_dbs,
            W_M = W_M_self,
            W_S = W_M_S,
            W_D = W_M_D,
            W_TR = W_M_TR,
            W_TC = W_M_TC,
            W_CI = W_M_CI,
            PSC_S = PSC_S,
            PSC_M = PSC_M,
            PSC_D = PSC_D,
            PSC_TC = PSC_TC,
            PSC_TR = PSC_TR,
            PSC_CI = PSC_CI,
            td_wl = td_wl,
            td_syn = td_syn,
            td_ct = td_ct,
            td_bl = td_bl,
            td_tc = td_tc,
            a = a_M,
            b = b_M,
            c = c_M,
            d = d_M,
            r = r_M,
            x = x_M,
            Is = I_syn_M,
            tau_f = tau_f_E,
            tau_d = tau_d_E,
            tau_s = tau_s_E,
            U = U_E,
            A = A_E,
            vr = vr, 
            vp = vp,
            dt = dt,
            fired = fired_M,
            spikes = spike_times_M
        )
        
        r_M = r_E; x_M = x_E; I_syn_M = I_syn_E;
        
        # D
        [r_E, x_E, I_syn_E, r_F, x_F, I_syn_F, v_D, u_D, fired_D, PSC_D, PSC_D_F, PSC_D_D] = d_cells(
            t = t,
            n_neurons = n_D, 
            sim_steps = sim_steps,
            voltage = v_D,
            u = u_D,
            current = I_D, 
            a_wg_noise = zeta_S_E,
            t_wg_noise = kisi_S_E,
            poisson_background_E = I_ps_D[0],
            poisson_background_I  = I_ps_D[1],
            n_affected = n_Hyper,
            synaptic_fidelity = dbs,
            I_dbs = syn_fid_D*I_dbs,
            W_M = W_D_M,
            W_S = W_D_S,
            W_D = W_D_self,
            W_TR = W_D_TR,
            W_TC = W_D_TC,
            W_CI = W_D_CI,
            PSC_S = PSC_S,
            PSC_M = PSC_M,
            PSC_D = PSC_D,
            PSC_TR = PSC_TR,
            PSC_CI = PSC_CI,
            PSC_D_TC = PSC_D_TC,
            PSC_D_D = PSC_D_D,
            PSC_D_F = PSC_D_F,
            td_wl = td_wl,
            td_syn = td_syn,
            td_ct = td_ct,
            td_bl = td_bl,
            td_tc = td_tc,
            a = a_D,
            b = b_D,
            c = c_D,
            d = d_D,
            r = r_D,
            x = x_D,
            Is = I_syn_D,
            r_F = r_D_F,
            x_F = x_D_F,
            Is_F = I_syn_D_F,
            tau_f = tau_f_E,
            tau_d = tau_d_E,
            tau_s = tau_s_E,
            U = U_E,
            A = A_E,
            A_F = A_E_D_F,
            A_D = A_E_D,
            vr = vr, 
            vp = vp,
            dt = dt,
            fired = fired_D,
            spikes = spike_times_D
        )
        
        r_D = r_E; x_D = x_E; I_syn_D = I_syn_E;
        r_D_F = r_F; x_D_F = x_F; I_syn_D_F = I_syn_F;
        
        gc.collect()
        
    # =============================================================================
    # CLEANING THE DATA
    # =============================================================================
    v_TR_clean = np.transpose(v_TR[:,chop_till:sim_steps])
    PSC_TR_clean = PSC_TR[:, chop_till:sim_steps]
    
    v_TC_clean = np.transpose(v_TC[:,chop_till:sim_steps])
    PSC_TC_clean = PSC_TC[:, chop_till:sim_steps]
    
    v_CI_clean = np.transpose(v_CI[:,chop_till:sim_steps])
    PSC_CI_clean = PSC_CI[:, chop_till:sim_steps]
    
    v_S_clean = np.transpose(v_S[:,chop_till:sim_steps])
    PSC_S_clean = PSC_S[:, chop_till:sim_steps]
    
    v_M_clean = np.transpose(v_M[:,chop_till:sim_steps])
    PSC_M_clean = PSC_M[:, chop_till:sim_steps]
    
    v_D_clean = np.transpose(v_D[:,chop_till:sim_steps])
    PSC_D_clean = PSC_D[:, chop_till:sim_steps]
        
    # =============================================================================
    # PLOTING THE VOLTAGES - CLEAN
    # =============================================================================
    print("--- Printing membrane potentials")
    plot_voltages(n_neurons = n_TR, voltage = v_TR_clean, chop_till = chop_till, sim_steps = sim_steps, title="V - TR Nucleus")
    plot_voltages(n_neurons = n_TC, voltage = v_TC_clean, chop_till = chop_till, sim_steps = sim_steps, title="V - TC Nucleus")
    plot_voltages(n_neurons = n_CI, voltage = v_CI_clean, chop_till = chop_till, sim_steps = sim_steps, title="V - CI")
    plot_voltages(n_neurons = n_S, voltage = v_S_clean, chop_till = chop_till, sim_steps = sim_steps, title="V - Layer S")
    plot_voltages(n_neurons = n_M, voltage = v_M_clean, chop_till = chop_till, sim_steps = sim_steps, title="V - Layer M")
    plot_voltages(n_neurons = n_D, voltage = v_D_clean, chop_till = chop_till, sim_steps = sim_steps, title="V - Layer D")
    
    print("--- Printing Post Synaptic Currents")
    plot_voltages(n_neurons = n_TR, voltage = v_TR_clean, chop_till = chop_till, sim_steps = sim_steps, title="PSC - TR Nucleus")
    plot_voltages(n_neurons = n_TC, voltage = v_TC_clean, chop_till = chop_till, sim_steps = sim_steps, title="PSC - TC Nucleus")
    plot_voltages(n_neurons = n_CI, voltage = v_CI_clean, chop_till = chop_till, sim_steps = sim_steps, title="PSC - CI")
    plot_voltages(n_neurons = n_S, voltage = v_S_clean, chop_till = chop_till, sim_steps = sim_steps, title="PSC - Layer S")
    plot_voltages(n_neurons = n_M, voltage = v_M_clean, chop_till = chop_till, sim_steps = sim_steps, title="PSC - Layer M")
    plot_voltages(n_neurons = n_D, voltage = v_D_clean, chop_till = chop_till, sim_steps = sim_steps, title="PSC - Layer D")
    
    # =============================================================================
    # MAKING RASTER PLOT
    # =============================================================================
    
    
    print("--- Printing Raster Plot")
    plot_raster(
        dbs,
        sim_steps, 
        sim_time,
        dt,
        chop_till, 
        n_TR, 
        n_TC,
        n_CI, 
        n_D, 
        n_M, 
        n_S, 
        n_total, 
        n_CI_FS,
        n_CI_LTS,
        n_D_RS,
        n_D_IB,
        n_S_RS,
        n_S_IB,
        spike_times_TR, 
        spike_times_TC, 
        spike_times_CI, 
        spike_times_D, 
        spike_times_M, 
        spike_times_S
        )

