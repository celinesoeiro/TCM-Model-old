"""
Created on Thu Mar 23 20:10:08 2023

@author: Celine Soeiro

@description: Thalamic Reticular Nucleus (TR) cells

# Abreviations:
    PSC: Post Synaptic Current
    IC: Intercortical Neurons
    SW: Synaptic Weight
    S: Surface (Supragranular) layer
    M: Middle (Granular) layer
    D: Deep (Infragranular) layer
    CI: Cortical Interneurons
    TR: Thalamic Reticular Nucleus
    TC: Thalamo-Cortical Relay nucleus
    PD: Parkinsonian Desease

Inputs:
    time step: dt
    peak voltage: vp
    rest voltage: vr
    simulation steps: sim_steps
    number of neurons: n
    neuron_params: a,b,c,d                                                          -> Izhikevich
    membrane recovery variable: v                                                   -> Izhikevich
    membrane potential of the neuron: u                                             -> Izhikevich
    available neurotransmitter resources ready to be used: r                        -> TM model (u in original article)
    neurotransmitter resources that remain available after synaptic transmission: x -> TM model
    post-synaptic current: I                                                        -> TM model
    PSC self contribution: PSC_self
    PSC layer S: PSC_S
    PSC layer M: PSC_M
    PSC layer D: PSC_D
    PSC layer TC: PSC_TC
    PSC CI: PSC_CI
    SW from self: SW_self
    SW from S: SW_S
    SW from M: SW_M
    SW from D: SW_D
    SW from CI: SW_CI
    SW from TC: SW_TC 
    bias current: Ib
    time vector: time
    
------------ OVERVIEW

Receive inhibitory stimulus from:
    - Self 

Receive excitatory stimulus from:
    - Thalamo-cortical relay nucleus (TRN)

Send inhibitory stimulus to:
    - Thalamo-cortical relay nucleus (TRN)
    
Send excitatory stimulus to:
    - None

"""

import numpy as np

def tr_cells(
       time_vector, 
       number_neurons, 
       simulation_steps, 
       neuron_params, 
       coupling_matrix, 
       current, 
       vr, 
       vp,
       dt,
       Idc,
       dvdt,
       dudt,
       r_eq,
       x_eq,
       I_eq,
       synapse_parameters,
       PSC_S,
       PSC_M,
       PSC_D,
       PSC_TR,
       PSC_TC,
       PSC_CI,     
    ):
    
    v = vr*np.ones((number_neurons,simulation_steps))
    u = 0*v
    r = np.zeros((3,len(time_vector)))
    x = np.ones((3,len(time_vector)))
    I = np.zeros((3,len(time_vector)))
    
    SW_self = coupling_matrix['W_II_tr']
    SW_S = coupling_matrix['W_IE_tr_s']
    SW_M = coupling_matrix['W_IE_tr_m']
    SW_D = coupling_matrix['W_IE_tr_d']
    SW_TC = coupling_matrix['W_IE_tr_tc']
    SW_CI = coupling_matrix['W_II_tr_ci']
 
    Ib = current + Idc*np.ones(number_neurons)
    
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
                self_feedback = SW_self[k][0]*PSC_TR[0][t]/number_neurons
                layer_S = SW_S[k][0]*PSC_S[0][t]/number_neurons
                layer_M = SW_M[k][0]*PSC_M[0][t]/number_neurons
                layer_D = SW_D[k][0]*PSC_D[0][t]/number_neurons
                layer_TC = SW_TC[k][0]*PSC_TC[0][t]/number_neurons
                layer_CI = SW_CI[k][0]*PSC_CI[0][t]/number_neurons
                noise = 0
                
                v[k][t] = v_aux + dt*(neuron_contribution + self_feedback + layer_S + layer_M + layer_D + layer_TC + layer_CI + noise)
                u[k][t] = u_aux + dt*dudt(v_aux, u_aux, neuron_params['a'], neuron_params['b'])
                
            # TM parameters
            tau_f = synapse_parameters['t_f']
            tau_d = synapse_parameters['t_d']
            U = synapse_parameters['U']
            A = synapse_parameters['distribution']
            tau_s = synapse_parameters['t_s']
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

    PSC_TR = I_post_synaptic
    
    return PSC_TR, I, AP, v, u, r, x
