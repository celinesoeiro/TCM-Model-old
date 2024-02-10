#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 00:51:58 2024

@author: celinesoeiro
"""

print("-- Importing packages and functions")
import numpy as np
import pandas as pd

from tcm_params import TCM_model_parameters, coupling_matrix_normal, coupling_matrix_PD
from model_plots import plot_heat_map, plot_raster_2
from model_functions import LFP, butter_bandpass_filter, PSD

from TR_nucleus_DBS import TR_nucleus
from TC_nucleus_DBS import TC_nucleus
from S_nucleus_DBS import S_nucleus
from M_nucleus_DBS import M_nucleus
from D_nucleus_DBS import D_nucleus
from CI_nucleus_DBS import CI_nucleus

# =============================================================================
# INITIAL VALUES
# =============================================================================
print("-- Initializing the global values")
neuron_quantities = TCM_model_parameters()['neuron_quantities']
neuron_per_structure = TCM_model_parameters()['neuron_per_structure']
neuron_params = TCM_model_parameters()['neuron_paramaters']
neuron_types_per_structure = TCM_model_parameters()['neuron_types_per_structure']
syn_params = TCM_model_parameters()['synapse_params_excitatory']

currents = TCM_model_parameters()['currents_per_structure']
p = TCM_model_parameters()['synapse_total_params']

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

vr = TCM_model_parameters()['vr']

dt = TCM_model_parameters()['dt']
sim_time = TCM_model_parameters()['simulation_time']
T = TCM_model_parameters()['simulation_time_ms']
fs = TCM_model_parameters()['sampling_frequency']

dbs = TCM_model_parameters()['dbs'][1]

sim_steps = TCM_model_parameters()['simulation_steps']
time_v = TCM_model_parameters()['time_vector']
time = np.arange(1, sim_steps)

t_f_E = syn_params['t_f']
t_d_E = syn_params['t_d']
t_s_E = syn_params['t_s']
U_E = syn_params['U']
A_E = syn_params['distribution']
td_syn = TCM_model_parameters()['time_delay_synapse']

I_dbs = np.zeros((2, sim_steps))
dev = 1 # divide the total simulation time in dev 
f_dbs = 160

# =============================================================================
# COUPLING MATRIXES
# =============================================================================
# Weight Matrix Normal Condition
Z_N = coupling_matrix_normal()['matrix']

# Weight Matrix Parkinsonian Desease Condition
Z_PD = coupling_matrix_PD()['matrix']

# normalizing Normal coupling matrix
Z_N_norm = Z_N/np.linalg.norm(Z_N)

# normalizing PD coupling matrix
Z_PD_norm = Z_PD/np.linalg.norm(Z_PD)

print("-- Printing the coupling matrixes")

CM_Normal = pd.DataFrame(Z_N_norm, columns=['S', 'M', 'D', 'CI', 'TC', 'TR'])
CM_PD = pd.DataFrame(Z_PD_norm, columns=['S', 'M', 'D', 'CI', 'TC', 'TR'])

plot_heat_map(matrix_normal = CM_Normal, matrix_PD = CM_PD)


# =============================================================================
# NEURON VARIABELS
# =============================================================================
def dbs(freq):
    p = 3
    print("-- Initializing the neuron variables")
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
        u_S[i][0] = neuron_params['b_S'][0][0]*vr
    
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
        u_M[i][0] = neuron_params['b_M'][0][0]*vr
    
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
        u_D[i][0] = neuron_params['b_D'][0][0]*vr
    
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
        u_CI[i][0] = neuron_params['b_CI'][0][0]*vr
        
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
        u_TC[i][0] = neuron_params['b_TC'][0][0]*vr
        
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
        u_TR[i][0] = neuron_params['b_TR'][0][0]*vr
        
    u_TR_syn = np.zeros((1, p))
    R_TR_syn = np.ones((1, p))
    I_TR_syn = np.zeros((1, p))
    
    tr_aux = 0
    
    # =============================================================================
    # DBS
    # =============================================================================
    
    # Simulate 1/dev of DBS
    if (dbs != 0):
        dev = 3
        
    if (dev == 1):
        print('dbs off')
        dbs_duration = sim_steps
        dbs_amplitude = 0.02
    else:
        print('dbs on')
        dbs_duration = int(np.round((sim_steps)/dev))
        dbs_amplitude = 1
    
    T_dbs = np.round(fs/freq)
    dbs_arr = np.arange(0, dbs_duration, T_dbs)
    I_dbs_full = np.zeros((1, dbs_duration))
    
    for i in dbs_arr:
        I_dbs_full[0][int(i)] = dbs_amplitude 
        
    if (dev == 1):
        I_dbs_pre = 1*I_dbs_full
    else:
        I_dbs_pre = 1*np.concatenate((
            np.zeros((1, 1)), 
            np.zeros((1, dbs_duration)), 
            I_dbs_full, 
            np.zeros((1, dbs_duration))
            ),axis=1)
    
    R_dbs = np.zeros((3, sim_steps))
    u_dbs = np.ones((3, sim_steps))
    Is_dbs = np.zeros((3, sim_steps))
    
    for p in range(3):
        for i in range(td_syn, sim_steps - 1):
            # u -> utilization factor -> resources ready for use
            u_dbs[p][i] = u_dbs[p - 1][i - 1] + -dt*u_dbs[p - 1][i - 1]/t_f_E[p - 1] + U_E[p - 1]*(1 - u_dbs[p - 1][i - 1])*I_dbs_pre[0][i- td_syn]
            # x -> availabe resources -> Fraction of resources that remain available after neurotransmitter depletion
            R_dbs[p][i] = R_dbs[p - 1][i - 1] + dt*(1 - R_dbs[p - 1][i - 1])/t_d_E[p - 1] - u_dbs[p - 1][i - 1]*R_dbs[p - 1][i - 1]*I_dbs_pre[0][i- td_syn]
            # PSC
            Is_dbs[p][i] = Is_dbs[p - 1][i - 1] + -dt*Is_dbs[p - 1][i - 1]/t_s_E + A_E[p - 1]*R_dbs[p - 1][i - 1]*u_dbs[p - 1][i - 1]*I_dbs_pre[0][i- td_syn]
            
    I_dbs_post = np.sum(Is_dbs, 0)
    
    I_dbs[0] = I_dbs_pre
    I_dbs[1] = I_dbs_post
    
    # =============================================================================
    # MAIN
    # =============================================================================
    print("-- Running model")
    for t in time:
    # =============================================================================
    #     TR
    # =============================================================================
        v_TR, u_TR, PSC_TR, u_TR_syn, I_TR_syn, R_TR_syn, tr_aux = TR_nucleus(t, v_TR, u_TR, AP_TR, PSC_TR, PSC_TC, PSC_CI, PSC_D_T, PSC_M, PSC_S, u_TR_syn, R_TR_syn, I_TR_syn, tr_aux, I_dbs[1])

    # =============================================================================
    #     TC
    # =============================================================================
        v_TC, u_TC, PSC_TC, u_TC_syn, I_TC_syn, R_TC_syn, PSC_T_D = TC_nucleus(t, v_TC, u_TC, AP_TC, PSC_TC, PSC_S, PSC_M, PSC_D_T, PSC_TR, PSC_CI, PSC_T_D, R_TC_syn, u_TC_syn, I_TC_syn, I_dbs[1])
            
    # =============================================================================
    #     S
    # =============================================================================
        v_S, u_S, PSC_S, u_S_syn, I_S_syn, R_S_syn = S_nucleus(t, v_S, u_S, AP_S, PSC_S, PSC_M, PSC_D, PSC_CI, PSC_TC, PSC_TR, u_S_syn, R_S_syn, I_S_syn, I_dbs[1])
            
    # =============================================================================
    #     M
    # =============================================================================
        v_M, u_M, PSC_M, u_M_syn, I_M_syn, R_M_syn = M_nucleus(t, v_M, u_M, AP_M, PSC_M, PSC_S, PSC_D, PSC_CI, PSC_TC, PSC_TR, u_M_syn, R_M_syn, I_M_syn, I_dbs[1])
                
    # =============================================================================
    #     D
    # =============================================================================
        v_D, u_D, PSC_D, u_D_syn, I_D_syn, R_D_syn, PSC_D_T = D_nucleus(t, v_D, u_D, AP_D, PSC_D, PSC_S, PSC_M, PSC_T_D, PSC_CI, PSC_TR, PSC_D_T, u_D_syn, R_D_syn, I_D_syn, I_dbs)
            
    # =============================================================================
    #     CI
    # =============================================================================
        v_CI, u_CI, PSC_CI, u_CI_syn, I_CI_syn, R_CI_syn = CI_nucleus(t, v_CI, u_CI, AP_CI, PSC_CI, PSC_D, PSC_M, PSC_S, PSC_TC, PSC_TR, u_CI_syn, R_CI_syn, I_CI_syn, I_dbs[1])
        
    # =============================================================================
    # PLOTS
    # =============================================================================
    print("-- Plotting results")
    
    # plot_voltages(n_neurons = n_S, voltage = v_S, title = "v - Layer S", neuron_types = neuron_types_per_structure['S'])
    # layer_raster_plot(n = n_S, AP = AP_S, sim_steps = sim_steps, layer_name = 'S', dt = dt)
    print('APs in S layer = ', np.count_nonzero(AP_S))
    
    # plot_voltages(n_neurons = n_M, voltage = v_M, title = "v - Layer M", neuron_types = neuron_types_per_structure['M'])
    # layer_raster_plot(n = n_M, AP = AP_M, sim_steps = sim_steps, layer_name = 'M', dt = dt)
    print('APs in M layer = ', np.count_nonzero(AP_M))
    
    # plot_voltages(n_neurons = n_D, voltage = v_D, title = "v - Layer D", neuron_types=neuron_types_per_structure['D'])
    # layer_raster_plot(n = n_D, AP = AP_D, sim_steps = sim_steps, layer_name = 'D', dt = dt)
    print('APs in D layer = ', np.count_nonzero(AP_D))
    
    # plot_voltages(n_neurons = n_CI, voltage = v_CI, title = "Layer CI", neuron_types=neuron_types_per_structure['CI'])
    # layer_raster_plot(n = n_CI, AP = AP_CI, sim_steps = sim_steps, layer_name = 'CI', dt = dt)
    print('APS in CI layer = ',np.count_nonzero(AP_CI))
    
    # plot_voltages(n_neurons = n_TC, voltage = v_TC, title = "TC", neuron_types=neuron_types_per_structure['TC'])
    # layer_raster_plot(n = n_TC, AP = AP_TC, sim_steps = sim_steps, layer_name = 'TC', dt = dt)
    print('APS in TC layer = ',np.count_nonzero(AP_TC))
    
    # plot_voltages(n_neurons = n_TR, voltage = v_TR, title = "TR", neuron_types=neuron_types_per_structure['TR'])
    # layer_raster_plot(n = n_TR, AP = AP_TR, sim_steps = sim_steps, layer_name = 'TR', dt = dt)
    print('APS in TR layer = ',np.count_nonzero(AP_TR))
    
    plot_raster_2(
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
    
    # =============================================================================
    # Signal analysis
    # =============================================================================
    print("-- Signal analysis")
    
    ## Getting the Local Field Potential
    LFP_D = LFP(PSC_D[0], PSC_CI[0])
    
    ## Bandpass filtering the LFP to get the beta waves
    beta_waves = butter_bandpass_filter(LFP_D, lowcut=13, highcut=30, fs=fs)
    
    # Power Spectral Density
    return PSD(beta_waves, fs, freq)
    
    print("-- Done!")
    
f_60, s_60 = dbs(60)
f_80, s_80 = dbs(80)
f_100, s_100 = dbs(100)
f_120, s_120 = dbs(120)
f_140, s_140 = dbs(140)
f_160, s_160 = dbs(160)
f_180, s_180 = dbs(180)
f_200, s_200 = dbs(200)
f_220, s_220 = dbs(220)
f_240, s_240 = dbs(240)
f_260, s_260 = dbs(260)
f_280, s_280 = dbs(280)
f_300, s_300 = dbs(300)

from matplotlib import pyplot as plt

x_arr = np.arange(0, 101, 10)

plt.figure()
plt.semilogy(f_60, s_60)
plt.semilogy(f_80, s_80)
plt.semilogy(f_120, s_120)
plt.semilogy(f_140, s_140)
plt.semilogy(f_160, s_160)
plt.semilogy(f_180, s_180)
plt.semilogy(f_200, s_200)
plt.semilogy(f_220, s_220)
plt.semilogy(f_240, s_240)
plt.semilogy(f_260, s_260)
plt.semilogy(f_280, s_280)
plt.semilogy(f_300, s_300)
plt.ylim([1e-5, 1e5])
plt.xlim([0, 100])
plt.xticks(x_arr)
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.title('PSD - compare')
plt.show()
