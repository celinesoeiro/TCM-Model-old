# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 21:24:07 2023

@author: Avell
"""
import numpy as np
import pandas as pd

def trn_cells(
        time_vector, 
        number_neurons, 
        simulation_steps, 
        neuron_params, 
        coupling_matrix, 
        currents, 
        vr, 
        vp,
        dt,
        Idc,
        dvdt,
        dudt,
        r_eq,
        x_eq,
        I_eq,
        get_parameters,
        PSC_S,
        PSC_M,
        PSC_D,
        PSC_TR,
        PSC_TN,
        PSC_CI,
    ):
    
    v = vr*np.ones((number_neurons,simulation_steps))
    u = 0*v
    r = np.zeros((3,len(time_vector)))
    x = np.ones((3,len(time_vector)))
    I = np.zeros((3,len(time_vector)))
    
    SW_self = coupling_matrix['W_EE_tc']
    SW_S = coupling_matrix['W_EE_tc_s']
    SW_M = coupling_matrix['W_EE_tc_m']
    SW_D = coupling_matrix['W_EE_tc_d']
    SW_TR = coupling_matrix['W_EI_tc_tr']
    SW_CI = coupling_matrix['W_EI_tc_ci']

    Ib = currents['I_TC_1'] + Idc*np.ones(number_neurons)
    
    AP = np.zeros((1,len(time_vector)))
    
    for t in range(1, len(time_vector)):
        print("interaction: %d" %(t))
        AP_aux = AP[0][t]
        for k in range(1, number_neurons):       
            v_aux = v[k - 1][t - 1]
            u_aux = u[k - 1][t - 1]
            
            if (v_aux >= vp):
                AP_aux = 1
                v[k][t] = vp
                v[k][t] = neuron_params['c']
                u[k][t] = u[k][t] + neuron_params['d']
            else:
                neuron_contribution = dvdt(v_aux, u_aux, Ib[k])
                self_feedback = SW_self[k][0]*PSC_TN[0][t]/number_neurons
                layer_S = SW_S[k][0]*PSC_S[0][t]/number_neurons
                layer_M = SW_M[k][0]*PSC_M[0][t]/number_neurons
                layer_D = SW_D[k][0]*PSC_D[0][t]/number_neurons
                layer_TR = SW_TR[k][0]*PSC_TR[0][t]/number_neurons
                layer_CI = SW_CI[k][0]*PSC_CI[0][t]/number_neurons
                noise = 0
                
                v[k][t] = v_aux + dt*(neuron_contribution + self_feedback + layer_S + layer_M + layer_D + layer_TR + layer_CI + noise)
                u[k][t] = u_aux + dt*dudt(v_aux, u_aux, neuron_params['a'], neuron_params['b'])
                
            # TM parameters
            tau_f = get_parameters('excitatory')['t_f']
            tau_d = get_parameters('excitatory')['t_d']
            U = get_parameters('excitatory')['U']
            A = get_parameters('excitatory')['distribution']
            tau_s = get_parameters('excitatory')['t_s']
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

    PSC_TN = I_post_synaptic
    
    return PSC_TN, AP, v, u, r, x
    
    
    
    
    
    