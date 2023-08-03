# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 14:59:45 2023

@author: Avell
"""

import random
import numpy as np
import pandas as pd
import seaborn as sns
import gc # Garbage Collector

random.seed(0)
sns.set()

from model_parameters import TCM_model_parameters, coupling_matrix_normal, coupling_matrix_PD

from model_functions import tm_synapse_dbs_eq, DBS_delta, tm_synapse_poisson_eq, poissonSpikeGen

from model_plots import plot_heat_map, plot_raster_comparison, plot_LFPs

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
samp_freq = global_parameters['sampling_frequency']
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

# TM Synapse Initial Values
## TR
x_TR = synapse_initial_values['x_TR']
r_TR = synapse_initial_values['r_TR']
I_syn_TR = synapse_initial_values['I_syn_TR']
## TC
x_TC = synapse_initial_values['x_TC']
r_TC = synapse_initial_values['r_TC']
I_syn_TC = synapse_initial_values['I_syn_TC']
## CI
x_CI = synapse_initial_values['x_CI']
r_CI = synapse_initial_values['r_CI']
I_syn_CI = synapse_initial_values['I_syn_CI']
## CI
x_S = synapse_initial_values['x_S']
r_S = synapse_initial_values['r_S']
I_syn_S = synapse_initial_values['I_syn_S']
## M
x_M = synapse_initial_values['x_M']
r_M = synapse_initial_values['r_M']
I_syn_M = synapse_initial_values['I_syn_M']
## D- Self
x_D = synapse_initial_values['x_D']
r_D = synapse_initial_values['r_D']
I_syn_D = synapse_initial_values['I_syn_D']
## D - Thamalus (Facilitating)
x_D_T = synapse_initial_values['x_D_T']
r_D_T = synapse_initial_values['r_D_T']
I_syn_D_T = synapse_initial_values['I_syn_D_T']
## Thalamus - D (Depressing)
x_T_D = synapse_initial_values['x_T_D']
r_T_D = synapse_initial_values['r_T_D']
I_syn_T_D = synapse_initial_values['I_syn_T_D']

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
A_E_T_D = tm_synapse_params_excitatory['distribution_T_D']
A_E_D_T = tm_synapse_params_excitatory['distribution_D_T']

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

# =============================================================================
# POST SYNAPTIC CURRENTS (Local Field Potentials)
# =============================================================================
PSC_S_off = np.zeros((1, sim_steps))
PSC_M_off = np.zeros((1, sim_steps))
PSC_D_off = np.zeros((1, sim_steps))
PSC_TC_off = np.zeros((1, sim_steps))
PSC_TR_off = np.zeros((1, sim_steps))
PSC_CI_off = np.zeros((1, sim_steps))
PSC_D_T_off = np.zeros((1, sim_steps))  # Facilitating (D to Thalamus)
PSC_T_D_off = np.zeros((1, sim_steps))  # Depressing (Thalamus to D)
PSC_D_TC_off = np.zeros((1, sim_steps)) # TC
# =============================================================================
PSC_S_on = np.zeros((1, sim_steps))
PSC_M_on = np.zeros((1, sim_steps))
PSC_D_on = np.zeros((1, sim_steps))
PSC_TC_on = np.zeros((1, sim_steps))
PSC_TR_on = np.zeros((1, sim_steps))
PSC_CI_on = np.zeros((1, sim_steps))
PSC_D_T_on = np.zeros((1, sim_steps))  # Facilitating (D to Thalamus)
PSC_T_D_on = np.zeros((1, sim_steps))  # Depressing (Thalamus to D)
PSC_D_TC_on = np.zeros((1, sim_steps)) # TC

# =============================================================================
# VOLTAGES
# =============================================================================
v_TR_off = vr*np.ones((n_TR, sim_steps))
u_TR_off = 0*v_TR_off

v_TC_off = vr*np.ones((n_TC, sim_steps))
u_TC_off = 0*v_TC_off

v_CI_off = vr*np.ones((n_CI, sim_steps))
u_CI_off = 0*v_CI_off

v_S_off = vr*np.ones((n_S, sim_steps))
u_S_off = 0*v_S_off

v_M_off = vr*np.ones((n_M, sim_steps))
u_M_off = 0*v_M_off

v_D_off = vr*np.ones((n_D, sim_steps))
u_D_off = 0*v_D_off
# =============================================================================
v_TR_on = vr*np.ones((n_TR, sim_steps))
u_TR_on = 0*v_TR_on

v_TC_on = vr*np.ones((n_TC, sim_steps))
u_TC_on = 0*v_TC_on

v_CI_on = vr*np.ones((n_CI, sim_steps))
u_CI_on = 0*v_CI_on

v_S_on = vr*np.ones((n_S, sim_steps))
u_S_on = 0*v_S_on

v_M_on = vr*np.ones((n_M, sim_steps))
u_M_on = 0*v_M_on

v_D_on = vr*np.ones((n_D, sim_steps))
u_D_on = 0*v_D_on

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

W_ps = [[w_ps * random.random() for _ in range(2)] for _ in range(6)]
poisson_firing = 20 + 2 * random.random()
[poisson_spikes, poisson_time_vector] = poissonSpikeGen(poisson_firing, T/1000, 1, dt/1000)
    
for i, fired in enumerate(poisson_spikes[0]):
    if (fired):
        tm_syn_E = tm_synapse_poisson_eq(AP_position=i, 
                                         sim_steps=sim_steps, 
                                         t_delay=td_syn, 
                                         dt=dt, 
                                         tau_f=tau_f_E, 
                                         tau_d=tau_d_E, 
                                         tau_s=tau_s_E, 
                                         U=U_E, 
                                         A=A_E)
        tm_syn_I = tm_synapse_poisson_eq(AP_position=i, 
                                         sim_steps=sim_steps, 
                                         t_delay=td_syn, 
                                         dt=dt, 
                                         tau_f=tau_f_I, 
                                         tau_d=tau_d_I, 
                                         tau_s=tau_s_I, 
                                         U=U_I, 
                                         A=A_I)
        
        # Excitatory
        I_ps_S[0][i] = W_ps[0][0]*tm_syn_E[i]
        I_ps_M[0][i] = W_ps[1][0]*tm_syn_E[i]
        I_ps_D[0][i] = W_ps[2][0]*tm_syn_E[i]
        I_ps_CI[0][i] = W_ps[3][0]*tm_syn_E[i]
        I_ps_TR[0][i] = W_ps[4][0]*tm_syn_E[i]
        I_ps_TC[0][i] = W_ps[5][0]*tm_syn_E[i]
        
        # Inhibitory
        I_ps_S[1][i] = W_ps[0][1]*tm_syn_I[i]
        I_ps_M[1][i] = W_ps[1][1]*tm_syn_I[i]
        I_ps_D[1][i] = W_ps[2][1]*tm_syn_I[i]
        I_ps_CI[1][i] = W_ps[3][1]*tm_syn_I[i]
        I_ps_TR[1][i] = W_ps[4][1]*tm_syn_I[i]
        I_ps_TC[1][i] = W_ps[5][1]*tm_syn_I[i]
            
# =============================================================================
# FIRED NEURONS
# =============================================================================
spike_times_TR_off = np.zeros((n_TR, sim_steps))
spike_times_TC_off = np.zeros((n_TC, sim_steps))
spike_times_CI_off = np.zeros((n_CI, sim_steps))
spike_times_D_off = np.zeros((n_D, sim_steps))
spike_times_M_off = np.zeros((n_M, sim_steps))
spike_times_S_off = np.zeros((n_S, sim_steps))
# =============================================================================
spike_times_TR_on = np.zeros((n_TR, sim_steps))
spike_times_TC_on = np.zeros((n_TC, sim_steps))
spike_times_CI_on = np.zeros((n_CI, sim_steps))
spike_times_D_on = np.zeros((n_D, sim_steps))
spike_times_M_on = np.zeros((n_M, sim_steps))
spike_times_S_on = np.zeros((n_S, sim_steps))

# =============================================================================
# INITIALIZING MODEL - DBS OFF
# =============================================================================

dbs = dbs_modes[0]
print('-- Running the model for DBS OFF')
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

# for DBS on all the time
dbs_duration = sim_steps
dbs_amplitude = 0.02

I_dbs_pre = DBS_delta(f_dbs, 
                      dbs_duration, 
                      dev, 
                      sim_steps, 
                      samp_freq, 
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
I_dbs[1][:] = I_dbs_post

for t in range(1, sim_steps):
    # TR
    tr_neurons = tr_cells(
        t = t,
        dt = dt,
        n_neurons = n_TR, 
        sim_steps = sim_steps,
        v = v_TR_off,
        u = u_TR_off,
        I_dc = I_TR, 
        a = a_TR,
        b = b_TR,
        c = c_TR,
        d = d_TR,
        vp = vp,
        r = r_TR,
        x = x_TR,
        I_syn = I_syn_TR,
        tau_f = tau_f_I,
        tau_d = tau_d_I,
        tau_s = tau_s_I,
        U = U_I,
        A = A_I,
        PSC_S = PSC_S_off[0][t - 1 - td_ct - td_syn],
        PSC_M = PSC_M_off[0][t - 1 - td_ct - td_syn],
        PSC_D = PSC_D_T_off[0][t - 1 - td_ct - td_syn], # Facilitating
        PSC_TC = PSC_TC_off[0][t - 1 - td_bl - td_syn],
        PSC_TR = PSC_TR_off[0][t - 1 - td_wl - td_syn],
        PSC_CI = PSC_CI_off[0][t - 1 - td_ct - td_syn],
        W_TR = W_TR_self,
        W_S = W_TR_S,
        W_M = W_TR_M,
        W_D = W_TR_D,
        W_TC = W_TR_TC,
        W_CI = W_TR_CI,
        a_wg_noise = kisi_TR_I,
        t_wg_noise = zeta_TR_I,
        poisson_background_E = I_ps_TR[0][t - 1 - td_wl - td_syn],
        poisson_background_I  = I_ps_TR[1][t - 1 - td_wl - td_syn],
        n_affected = n_TR_affected,
        I_dbs = syn_fid_TR*I_dbs[1][t - 1],
        spikes = spike_times_TR_off,
    )
    r_TR = tr_neurons['r'];             x_TR = tr_neurons['x'];
    I_syn_TR = tr_neurons['I_syn'];     PSC_TR_off[0][t] = tr_neurons['PSC_TR'];
    v_TR_off = tr_neurons['v'];         u_TR_off = tr_neurons['u']; 
    print(f'PSC_TR_off[0][{t}] = ',PSC_TR_off[0][t])
        
    # TC    
    tc_neurons = tc_cells(
        t = t,
        dt = dt,
        n_neurons = n_TC, 
        sim_steps = sim_steps,
        v = v_TC_off,
        u = u_TC_off,
        I_dc = I_TC, 
        a = a_TC,
        b = b_TC,
        c = c_TC,
        d = d_TC,
        vp = vp,
        r = r_TC,
        x = x_TC,
        I_syn = I_syn_TC,
        tau_f = tau_f_E,
        tau_d = tau_d_E,
        tau_s = tau_s_E,
        U = U_E,
        A = A_E,
        r_D = r_T_D,
        x_D = x_T_D,
        I_syn_D = I_syn_T_D,
        tau_f_D = tau_f_E,
        tau_d_D = tau_d_E,
        tau_s_D = tau_s_E,
        U_D = U_E,
        A_D = A_E_T_D,
        PSC_S = PSC_S_off[0][t - 1 - td_ct - td_syn],
        PSC_M = PSC_M_off[0][t - 1 - td_ct - td_syn],
        PSC_D = PSC_D_T_off[0][t - 1 - td_ct - td_syn],
        PSC_TC = PSC_TC_off[0][t - 1 - td_wl - td_syn],
        PSC_TR = PSC_TR_off[0][t - 1 - td_bl - td_syn],
        PSC_CI = PSC_CI_off[0][t - 1 - td_ct - td_syn],
        W_S = W_TC_S,
        W_M = W_TC_M,
        W_D = W_TC_D,
        W_TR = W_TC_TR,
        W_TC = W_TC_self,
        W_CI = W_TC_CI,
        a_wg_noise = kisi_TC_E,
        t_wg_noise = zeta_TC_E,
        poisson_background_E = I_ps_TC[0][t - 1 - td_wl - td_syn],
        poisson_background_I  = I_ps_TC[1][t - 1 - td_wl - td_syn],
        n_affected = n_TC_affected,
        I_dbs = syn_fid_TC*I_dbs[1][t - 1],
        spikes = spike_times_TC_off
    )
    
    r_TC = tc_neurons['r'];             x_TC = tc_neurons['x'];
    I_syn_TC = tc_neurons['I_syn'];     PSC_TC_off[0][t] = tc_neurons['PSC_TC'];
    v_TC_off = tc_neurons['v'];         u_TC_off = tc_neurons['u'];
    r_T_D = tc_neurons['r_D'];          x_T_D = tc_neurons['x_D'];
    I_syn_T_D = tc_neurons['I_syn_D'];  PSC_D_TC_off[0][t] = tc_neurons['PSC_D'];
    
    # S
    s_neurons = s_cells(
        t = t,
        dt = dt,
        n_neurons = n_S, 
        sim_steps = sim_steps,
        v = v_S_off,
        u = u_S_off,
        I_dc = I_S, 
        a = a_S,
        b = b_S,
        c = c_S,
        d = d_S,
        r = r_S,
        x = x_S,
        I_syn = I_syn_S,
        tau_f = tau_f_E,
        tau_d = tau_d_E,
        tau_s = tau_s_E,
        U = U_E,
        A = A_E,
        PSC_S = PSC_S_off[0][t - 1 - td_wl - td_syn],
        PSC_M = PSC_M_off[0][t - 1 - td_bl - td_syn],
        PSC_D = PSC_D_off[0][t - 1 - td_bl - td_syn],
        PSC_TC = PSC_TC_off[0][t - 1 - td_tc - td_syn],
        PSC_TR = PSC_TR_off[0][t - 1 - td_tc - td_syn],
        PSC_CI = PSC_CI_off[0][t - 1 - td_wl - td_syn],
        W_S = W_S_self,
        W_M = W_S_M,
        W_D = W_S_D,
        W_TR = W_S_TR,
        W_TC = W_S_TC,
        W_CI = W_S_CI,
        a_wg_noise = kisi_S_E,
        t_wg_noise = zeta_S_E,
        poisson_background_E = I_ps_S[0][t - 1 - td_wl - td_syn],
        poisson_background_I  = I_ps_S[1][t - 1 - td_wl - td_syn],
        n_affected = n_S_affected,
        I_dbs = syn_fid_S*I_dbs[1][t - 1],
        vp = vp,
        spikes = spike_times_S_off
    )
    
    r_S = s_neurons['r'];           x_S = s_neurons['x'];
    I_syn_S = s_neurons['I_syn'];   PSC_S_off[0][t] = s_neurons['PSC_S']; 
    v_S_off = s_neurons['v'];       u_S_off = s_neurons['u'];
    
    # M
    m_neurons = m_cells(
        t = t,
        dt = dt,
        n_neurons = n_M, 
        sim_steps = sim_steps,
        v = v_M_off,
        u = u_M_off,
        I_dc = I_M, 
        a = a_M,
        b = b_M,
        c = c_M,
        d = d_M,
        r = r_M,
        x = x_M,
        I_syn = I_syn_M,
        tau_f = tau_f_E,
        tau_d = tau_d_E,
        tau_s = tau_s_E,
        U = U_E,
        A = A_E,
        PSC_S = PSC_S_off[0][t - 1 - td_bl - td_syn],
        PSC_M = PSC_M_off[0][t - 1 - td_wl - td_syn],
        PSC_D = PSC_D_off[0][t - 1 - td_bl - td_syn],
        PSC_TC = PSC_TC_off[0][t - 1 - td_tc - td_syn],
        PSC_TR = PSC_TR_off[0][t - 1 - td_tc - td_syn],
        PSC_CI = PSC_CI_off[0][t - 1 - td_wl - td_syn],
        W_M = W_M_self,
        W_S = W_M_S,
        W_D = W_M_D,
        W_TR = W_M_TR,
        W_TC = W_M_TC,
        W_CI = W_M_CI,
        a_wg_noise = kisi_M_E,
        t_wg_noise = zeta_M_E,
        poisson_background_E = I_ps_M[0][t - 1 - td_wl - td_syn],
        poisson_background_I  = I_ps_M[1][t - 1 - td_wl - td_syn],
        n_affected = n_M_affected,
        I_dbs = syn_fid_M*I_dbs[1][t - 1],
        vp = vp,
        spikes = spike_times_M_off
    )
    
    r_M = m_neurons['r'];           x_M = m_neurons['x'];
    I_syn_M = m_neurons['I_syn'];   PSC_M_off[0][t] = m_neurons['PSC_M']; 
    v_M_off = m_neurons['v'];       u_M_off = m_neurons['u'];
    
    # D
    d_neurons = d_cells(
        t = t,
        dt = dt,
        n_neurons = n_D, 
        sim_steps = sim_steps,
        v = v_D_off,
        u = u_D_off,
        I_dc = I_D, 
        a = a_D,
        b = b_D,
        c = c_D,
        d = d_D,
        r = r_D,
        x = x_D,
        I_syn = I_syn_D,
        U = U_E,
        A = A_E,
        r_F = r_D_T,
        x_F = x_D_T,
        I_syn_F = I_syn_D_T,
        A_F = A_E_D_T,
        A_D = A_E_T_D,
        tau_f = tau_f_E,
        tau_d = tau_d_E,
        tau_s = tau_s_E,
        PSC_S = PSC_S_off[0][t - 1 - td_bl - td_syn],
        PSC_M = PSC_M_off[0][t - 1 - td_bl - td_syn],
        PSC_D = PSC_D_off[0][t - 1 - td_wl - td_syn],
        PSC_TR = PSC_TR_off[0][t - 1 - td_tc - td_syn],
        PSC_CI = PSC_CI_off[0][t - 1 - td_wl - td_syn],
        PSC_D_TC = PSC_D_TC_off[0][t - 1 - td_tc - td_syn],
        W_M = W_D_M,
        W_S = W_D_S,
        W_D = W_D_self,
        W_TR = W_D_TR,
        W_TC = W_D_TC,
        W_CI = W_D_CI,
        a_wg_noise = kisi_D_E,
        t_wg_noise = zeta_D_E,
        poisson_background_E = I_ps_D[0][t - 1 - td_wl - td_syn],
        poisson_background_I  = I_ps_D[1][t - 1 - td_wl - td_syn],
        n_affected = n_Hyper,
        I_dbs = syn_fid_D*I_dbs,
        vp = vp,
        spikes = spike_times_D_off
    )
    
    v_D_off = d_neurons['v'];              u_D_off = d_neurons['u'];
    r_D = d_neurons['r'];                  x_D = d_neurons['x'];
    I_syn_D = d_neurons['I_syn'];          PSC_D_off[0][t] = d_neurons['PSC_D'];
    r_D_T = d_neurons['r_F'];              x_D_T = d_neurons['x_F'];
    I_syn_D_T = d_neurons['I_syn_F'];      PSC_D_T_off[0][t] = d_neurons['PSC_D_T'];
    PSC_T_D_off[0][t] = d_neurons['PSC_T_D'];
    
    # CI
    ci_neurons = ci_cells(
        t = t,
        dt = dt,
        n_neurons = n_CI, 
        sim_steps = sim_steps,
        v = v_CI_off,
        u = u_CI_off,
        I_dc = I_CI, 
        a = a_CI,
        b = b_CI,
        c = c_CI,
        d = d_CI,
        r = r_CI,
        x = x_CI,
        I_syn = I_syn_CI,
        tau_f = tau_f_I,
        tau_d = tau_d_I,
        tau_s = tau_s_I,
        U = U_I,
        A = A_I,
        W_S = W_CI_S,
        W_M = W_CI_M,
        W_D = W_CI_D,
        W_TR = W_CI_TR,
        W_TC = W_CI_TC,
        W_CI = W_CI_self,
        PSC_S = PSC_S_off[0][t - 1 - td_wl - td_syn],
        PSC_M = PSC_M_off[0][t - 1 - td_wl - td_syn],
        PSC_D = PSC_D_off[0][t - 1 - td_wl - td_syn],
        PSC_TC = PSC_TC_off[0][t - 1 - td_tc - td_syn],
        PSC_TR = PSC_TR_off[0][t - 1 - td_tc - td_syn],
        PSC_CI = PSC_CI_off[0][t - 1 - td_wl - td_syn],          
        a_wg_noise = kisi_CI_I,
        t_wg_noise = zeta_CI_I,
        poisson_background_E = I_ps_CI[0][t - 1- td_wl - td_syn],
        poisson_background_I  = I_ps_CI[1][t - 1 - td_wl - td_syn],
        n_affected = n_CI_affected,
        I_dbs = syn_fid_CI*I_dbs[1][t - 1],
        vp = vp,
        spikes = spike_times_CI_off
    )
    
    r_CI = ci_neurons['r'];             x_CI = ci_neurons['x']; 
    I_syn_CI = ci_neurons['I_syn'];     PSC_CI_off[0][t] = ci_neurons['PSC_CI'];  
    v_CI_off = ci_neurons['v'];         u_CI_off = ci_neurons['u'];
    
    gc.collect()


# =============================================================================
# INITIALIZING MODEL - DBS ON
# =============================================================================

dbs = dbs_modes[1]
print('-- Running the model for DBS ON')
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

dev = 3

dbs_duration = int(np.round((sim_steps - chop_till)/dev))
dbs_amplitude = 1
    
I_dbs_pre = DBS_delta(f_dbs, 
                      dbs_duration, 
                      dev, 
                      sim_steps, 
                      samp_freq, 
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
    tr_neurons = tr_cells(
        t = t,
        dt = dt,
        n_neurons = n_TR, 
        sim_steps = sim_steps,
        v = v_TR_on,
        u = u_TR_on,
        I_dc = I_TR, 
        a = a_TR,
        b = b_TR,
        c = c_TR,
        d = d_TR,
        vp = vp,
        r = r_TR,
        x = x_TR,
        I_syn = I_syn_TR,
        tau_f = tau_f_I,
        tau_d = tau_d_I,
        tau_s = tau_s_I,
        U = U_I,
        A = A_I,
        PSC_S = PSC_S_on[0][t - 1 - td_ct - td_syn],
        PSC_M = PSC_M_on[0][t - 1 - td_ct - td_syn],
        PSC_D = PSC_D_T_on[0][t - 1 - td_ct - td_syn],
        PSC_TC = PSC_TC_on[0][t - 1 - td_bl - td_syn],
        PSC_TR = PSC_TR_on[0][t - 1 - td_wl - td_syn],
        PSC_CI = PSC_CI_on[0][t - 1 - td_ct - td_syn],
        W_TR = W_TR_self,
        W_S = W_TR_S,
        W_M = W_TR_M,
        W_D = W_TR_D,
        W_TC = W_TR_TC,
        W_CI = W_TR_CI,
        a_wg_noise = kisi_TR_I,
        t_wg_noise = zeta_TR_I,
        poisson_background_E = I_ps_TR[0][t - 1 - td_wl - td_syn],
        poisson_background_I  = I_ps_TR[1][t - 1 - td_wl - td_syn],
        n_affected = n_TR_affected,
        I_dbs = syn_fid_TR*I_dbs[1][t - 1],
        spikes = spike_times_TR_on,
    )
    r_TR = tr_neurons['r'];            x_TR = tr_neurons['x'];
    I_syn_TR = tr_neurons['I_syn'];    PSC_TR_on[0][t] = tr_neurons['PSC_TR'];
    v_TR_on = tr_neurons['v'];         u_TR_on = tr_neurons['u'];
        
    # TC    
    tc_neurons = tc_cells(
        t = t,
        dt = dt,
        n_neurons = n_TC, 
        sim_steps = sim_steps,
        v = v_TC_on,
        u = u_TC_on,
        I_dc = I_TC, 
        a = a_TC,
        b = b_TC,
        c = c_TC,
        d = d_TC,
        vp = vp,
        r = r_TC,
        x = x_TC,
        I_syn = I_syn_TC,
        tau_f = tau_f_E,
        tau_d = tau_d_E,
        tau_s = tau_s_E,
        U = U_E,
        A = A_E,
        r_D = r_T_D,
        x_D = x_T_D,
        I_syn_D = I_syn_T_D,
        tau_f_D = tau_f_E,
        tau_d_D = tau_d_E,
        tau_s_D = tau_s_E,
        U_D = U_E,
        A_D = A_E_T_D,
        PSC_S = PSC_S_on[0][t - 1 - td_ct - td_syn],
        PSC_M = PSC_M_on[0][t - 1 - td_ct - td_syn],
        PSC_D = PSC_D_T_on[0][t - 1 - td_ct - td_syn],
        PSC_TC = PSC_TC_on[0][t - 1 - td_wl - td_syn],
        PSC_TR = PSC_TR_on[0][t - 1 - td_bl - td_syn],
        PSC_CI = PSC_CI_on[0][t - 1 - td_ct - td_syn],
        W_S = W_TC_S,
        W_M = W_TC_M,
        W_D = W_TC_D,
        W_TR = W_TC_TR,
        W_TC = W_TC_self,
        W_CI = W_TC_CI,
        a_wg_noise = kisi_TC_E,
        t_wg_noise = zeta_TC_E,
        poisson_background_E = I_ps_TC[0][t - 1 - td_wl - td_syn],
        poisson_background_I  = I_ps_TC[1][t - 1 - td_wl - td_syn],
        n_affected = n_TC_affected,
        I_dbs = syn_fid_TC*I_dbs[1][t - 1],
        spikes = spike_times_TC_on
    )
    
    r_TC = tc_neurons['r'];            x_TC = tc_neurons['x'];
    I_syn_TC = tc_neurons['I_syn'];    PSC_TC_on[0][t] = tc_neurons['PSC_TC'];
    v_TC_on = tc_neurons['v'];         u_TC_on = tc_neurons['u'];
    r_T_D = tc_neurons['r_D'];         x_T_D = tc_neurons['x_D'];
    I_syn_T_D = tc_neurons['I_syn_D']; PSC_D_TC_on[0][t] = tc_neurons['PSC_D'];
    
    # S
    s_neurons = s_cells(
        t = t,
        dt = dt,
        n_neurons = n_S, 
        sim_steps = sim_steps,
        v = v_S_on,
        u = u_S_on,
        I_dc = I_S, 
        a = a_S,
        b = b_S,
        c = c_S,
        d = d_S,
        r = r_S,
        x = x_S,
        I_syn = I_syn_S,
        tau_f = tau_f_E,
        tau_d = tau_d_E,
        tau_s = tau_s_E,
        U = U_E,
        A = A_E,
        PSC_S = PSC_S_on[0][t - 1 - td_wl - td_syn],
        PSC_M = PSC_M_on[0][t - 1 - td_bl - td_syn],
        PSC_D = PSC_D_on[0][t - 1 - td_bl - td_syn],
        PSC_TC = PSC_TC_on[0][t - 1 - td_tc - td_syn],
        PSC_TR = PSC_TR_on[0][t - 1 - td_tc - td_syn],
        PSC_CI = PSC_CI_on[0][t - 1 - td_wl - td_syn],
        W_S = W_S_self,
        W_M = W_S_M,
        W_D = W_S_D,
        W_TR = W_S_TR,
        W_TC = W_S_TC,
        W_CI = W_S_CI,
        a_wg_noise = kisi_S_E,
        t_wg_noise = zeta_S_E,
        poisson_background_E = I_ps_S[0][t - 1 - td_wl - td_syn],
        poisson_background_I  = I_ps_S[1][t - 1 - td_wl - td_syn],
        n_affected = n_S_affected,
        I_dbs = syn_fid_S*I_dbs[1][t - 1],
        vp = vp,
        spikes = spike_times_S_on
    )
    
    r_S = s_neurons['r'];         x_S = s_neurons['x'];
    I_syn_S = s_neurons['I_syn']; PSC_S_on[0][t] = s_neurons['PSC_S']; 
    v_S_on = s_neurons['v'];      u_S_on = s_neurons['u'];
    
    # M
    m_neurons = m_cells(
        t = t,
        dt = dt,
        n_neurons = n_M, 
        sim_steps = sim_steps,
        v = v_M_on,
        u = u_M_on,
        I_dc = I_M, 
        a = a_M,
        b = b_M,
        c = c_M,
        d = d_M,
        r = r_M,
        x = x_M,
        I_syn = I_syn_M,
        tau_f = tau_f_E,
        tau_d = tau_d_E,
        tau_s = tau_s_E,
        U = U_E,
        A = A_E,
        PSC_S = PSC_S_on[0][t - 1 - td_bl - td_syn],
        PSC_M = PSC_M_on[0][t - 1 - td_wl - td_syn],
        PSC_D = PSC_D_on[0][t - 1 - td_bl - td_syn],
        PSC_TC = PSC_TC_on[0][t - 1 - td_tc - td_syn],
        PSC_TR = PSC_TR_on[0][t - 1 - td_tc - td_syn],
        PSC_CI = PSC_CI_on[0][t - 1 - td_wl - td_syn],
        W_M = W_M_self,
        W_S = W_M_S,
        W_D = W_M_D,
        W_TR = W_M_TR,
        W_TC = W_M_TC,
        W_CI = W_M_CI,
        a_wg_noise = kisi_M_E,
        t_wg_noise = zeta_M_E,
        poisson_background_E = I_ps_M[0][t - 1 - td_wl - td_syn],
        poisson_background_I  = I_ps_M[1][t - 1 - td_wl - td_syn],
        n_affected = n_M_affected,
        I_dbs = syn_fid_M*I_dbs[1][t - 1],
        vp = vp,
        spikes = spike_times_M_on
    )
    
    r_M = m_neurons['r'];             x_M = m_neurons['x'];
    I_syn_M = m_neurons['I_syn'];     PSC_M_on[0][t] = m_neurons['PSC_M']; 
    v_M_on = m_neurons['v'];          u_M_on = m_neurons['u'];
    
    # D
    d_neurons = d_cells(
        t = t,
        dt = dt,
        n_neurons = n_D, 
        sim_steps = sim_steps,
        v = v_D_on,
        u = u_D_on,
        I_dc = I_D, 
        a = a_D,
        b = b_D,
        c = c_D,
        d = d_D,
        r = r_D,
        x = x_D,
        I_syn = I_syn_D,
        U = U_E,
        A = A_E,
        r_F = r_D_T,
        x_F = x_D_T,
        I_syn_F = I_syn_D_T,
        A_F = A_E_D_T,
        A_D = A_E_T_D,
        tau_f = tau_f_E,
        tau_d = tau_d_E,
        tau_s = tau_s_E,
        PSC_S = PSC_S_on[0][t - 1 - td_bl - td_syn],
        PSC_M = PSC_M_on[0][t - 1 - td_bl - td_syn],
        PSC_D = PSC_D_on[0][t - 1 - td_wl - td_syn],
        PSC_TR = PSC_TR_on[0][t - 1 - td_tc - td_syn],
        PSC_CI = PSC_CI_on[0][t - 1 - td_wl - td_syn],
        PSC_D_TC = PSC_D_TC_on[0][t - 1 - td_tc - td_syn],
        W_M = W_D_M,
        W_S = W_D_S,
        W_D = W_D_self,
        W_TR = W_D_TR,
        W_TC = W_D_TC,
        W_CI = W_D_CI,
        a_wg_noise = kisi_D_E,
        t_wg_noise = zeta_D_E,
        poisson_background_E = I_ps_D[0][t - 1 - td_wl - td_syn],
        poisson_background_I  = I_ps_D[1][t - 1 - td_wl - td_syn],
        n_affected = n_Hyper,
        I_dbs = syn_fid_D*I_dbs,
        vp = vp,
        spikes = spike_times_D_on
    )
    
    v_D_on = d_neurons['v'];            u_D_on = d_neurons['u'];
    r_D = d_neurons['r'];               x_D = d_neurons['x'];
    I_syn_D = d_neurons['I_syn'];       PSC_D_on[0][t] = d_neurons['PSC_D'];
    r_D_T = d_neurons['r_F'];           x_D_T = d_neurons['x_F'];
    I_syn_D_T = d_neurons['I_syn_F'];   PSC_D_T_on[0][t] = d_neurons['PSC_D_T'];
    PSC_T_D_on[0][t] = d_neurons['PSC_T_D'];
    
    # CI
    ci_neurons = ci_cells(
        t = t,
        dt = dt,
        n_neurons = n_CI, 
        sim_steps = sim_steps,
        v = v_CI_on,
        u = u_CI_on,
        I_dc = I_CI, 
        a = a_CI,
        b = b_CI,
        c = c_CI,
        d = d_CI,
        r = r_CI,
        x = x_CI,
        I_syn = I_syn_CI,
        tau_f = tau_f_I,
        tau_d = tau_d_I,
        tau_s = tau_s_I,
        U = U_I,
        A = A_I,
        W_S = W_CI_S,
        W_M = W_CI_M,
        W_D = W_CI_D,
        W_TR = W_CI_TR,
        W_TC = W_CI_TC,
        W_CI = W_CI_self,
        PSC_S = PSC_S_on[0][t - 1 - td_wl - td_syn],
        PSC_M = PSC_M_on[0][t - 1 - td_wl - td_syn],
        PSC_D = PSC_D_on[0][t - 1 - td_wl - td_syn],
        PSC_TC = PSC_TC_on[0][t - 1 - td_tc - td_syn],
        PSC_TR = PSC_TR_on[0][t - 1 - td_tc - td_syn],
        PSC_CI = PSC_CI_on[0][t - 1 - td_wl - td_syn],          
        a_wg_noise = kisi_CI_I,
        t_wg_noise = zeta_CI_I,
        poisson_background_E = I_ps_CI[0][t - 1 - td_wl - td_syn],
        poisson_background_I  = I_ps_CI[1][t - 1 - td_wl - td_syn],
        n_affected = n_CI_affected,
        I_dbs = syn_fid_CI*I_dbs[1][t - 1],
        vp = vp,
        spikes = spike_times_CI_on
    )
    
    r_CI = ci_neurons['r'];           x_CI = ci_neurons['x']; 
    I_syn_CI = ci_neurons['I_syn'];   PSC_CI_on[0][t] = ci_neurons['PSC_CI'];  
    v_CI_on = ci_neurons['v'];        u_CI_on = ci_neurons['u'];
    
    gc.collect()
    
# =============================================================================
# CLEANING THE DATA
# =============================================================================
PSC_TR_off = np.transpose(PSC_TR_off[:, chop_till:sim_steps])
PSC_TC_off = np.transpose(PSC_TC_off[:, chop_till:sim_steps])
PSC_CI_off = np.transpose(PSC_CI_off[:, chop_till:sim_steps])
PSC_D_off = np.transpose(PSC_D_off[:, chop_till:sim_steps])
PSC_M_off = np.transpose(PSC_M_off[:, chop_till:sim_steps])
PSC_S_off = np.transpose(PSC_S_off[:, chop_till:sim_steps])

spike_TR_OFF = spike_times_TR_off[:, chop_till:sim_steps]
spike_TC_OFF = spike_times_TC_off[:, chop_till:sim_steps]
spike_CI_OFF = spike_times_CI_off[:, chop_till:sim_steps]
spike_D_OFF = spike_times_D_off[:, chop_till:sim_steps]
spike_M_OFF = spike_times_M_off[:, chop_till:sim_steps]
spike_S_OFF = spike_times_S_off[:, chop_till:sim_steps]

PSC_TR_on = np.transpose(PSC_TR_on[:, chop_till:sim_steps])
PSC_TC_on = np.transpose(PSC_TC_on[:, chop_till:sim_steps])
PSC_CI_on = np.transpose(PSC_CI_on[:, chop_till:sim_steps])
PSC_D_on = np.transpose(PSC_D_on[:, chop_till:sim_steps])
PSC_M_on = np.transpose(PSC_M_on[:, chop_till:sim_steps])
PSC_S_on = np.transpose(PSC_S_on[:, chop_till:sim_steps])
    
spike_TR_ON = spike_times_TR_on[:, chop_till:sim_steps]
spike_TC_ON = spike_times_TC_on[:, chop_till:sim_steps]
spike_CI_ON = spike_times_CI_on[:, chop_till:sim_steps]
spike_D_ON = spike_times_D_on[:, chop_till:sim_steps]
spike_M_ON = spike_times_M_on[:, chop_till:sim_steps]
spike_S_ON = spike_times_S_on[:, chop_till:sim_steps]
    
# =============================================================================
# PLOTING THE LOCAL FIELD POTENTIALS (LFP) - DBS OFF
# =============================================================================
print("--- Printing Post Synaptic Currents - LFP - DBS OFF")
plot_LFPs(PSC_S_off, PSC_M_off, PSC_D_off, PSC_CI_off, PSC_TC_off, PSC_TR_off, chop_till, sim_steps, 'DBS OFF')

print("--- Printing Post Synaptic Currents - LFP - DBS ON")
# Layer V LFP are simulated as the sum of all excitatory PSCs (EPCs) within Layer D
plot_LFPs(PSC_S_on, PSC_M_on, PSC_D_on, PSC_CI_on, PSC_TC_on, PSC_TR_on, chop_till, sim_steps, 'DBS ON')
          
# =============================================================================
# MAKING RASTER PLOTS 
# =============================================================================
plot_raster_comparison(
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
    spike_TR_ON, 
    spike_TC_ON, 
    spike_CI_ON, 
    spike_D_ON, 
    spike_M_ON, 
    spike_S_ON, 
    spike_TR_OFF, 
    spike_TC_OFF, 
    spike_CI_OFF, 
    spike_D_OFF, 
    spike_M_OFF, 
    spike_S_OFF
    )

# =============================================================================
# MAKING POWER SPECTRAL DENSITY PLOT (PSD)
# PSD shows how the power of a signal is distributed over frequencies. 
# =============================================================================
import matplotlib.pyplot as plt

rho = 0.27
dist = 100^-6

LFP_off = np.transpose(PSC_D_off)

(S, f) = plt.psd(LFP_off, Fs=samp_freq)

plt.semilogy(f, S)
plt.xlim([0, 100])
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.title('PSD OFF')
plt.show()


LFP_on = np.transpose(PSC_D_on)

(S, f) = plt.psd(LFP_on, Fs=samp_freq)

plt.semilogy(f, S)
plt.xlim([0, 100])
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.title('PSD ON')
plt.show()
