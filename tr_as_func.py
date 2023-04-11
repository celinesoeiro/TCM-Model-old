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
       Idc_tune,
       dvdt,
       dudt,
       r_eq,
       x_eq,
       I_eq,
       tm_synapse_eq,
       synapse_parameters,
       PSC_S,
       PSC_M,
       PSC_D,
       PSC_TR,
       PSC_TC,
       PSC_CI,    
       neuron_type,
       random_factor
    ):
    
    v = vr*np.ones((number_neurons,simulation_steps))
    u = 0*v
    r = np.zeros((3,len(time_vector)))
    x = np.ones((3,len(time_vector)))
    Is = np.zeros((3,len(time_vector)))
    
    SW_self = coupling_matrix['W_II_tr']
    SW_S = coupling_matrix['W_IE_tr_s']
    SW_M = coupling_matrix['W_IE_tr_m']
    SW_D = coupling_matrix['W_IE_tr_d']
    SW_TC = coupling_matrix['W_IE_tr_tc']
    SW_CI = coupling_matrix['W_II_tr_ci']
 
    Isi = []
    Ib = current + Idc_tune*np.ones(number_neurons)
    
    AP = np.zeros((1,len(time_vector)))
    
    if (neuron_type == 'excitatory' or 'excit'):
        a = neuron_params['a']
        b = neuron_params['b']
        c = neuron_params['c'] + 15*random_factor**2
        d = neuron_params['d'] - 6*random_factor**2
    elif (neuron_type == 'inhibitory' or 'inhib'):
        a = neuron_params['a'] + 0.08*random_factor
        b = neuron_params['b'] - 0.05*random_factor
        c = neuron_params['c']
        d = neuron_params['d']
    else:
        return 'Neuron type must be excitatory or inhibitory'

    for t in range(1, len(time_vector)):
        AP_aux = AP[0][t]
        for k in range(1, number_neurons):        
            v_aux = v[k - 1][t - 1]
            u_aux = u[k - 1][t - 1]
            
            if (v_aux >= vp):
                AP_aux = 1
                v[k][t] = vp
                v[k][t] = c
                u[k][t] = u[k][t] + d
            else:
                neuron_contribution = dvdt(v_aux, u_aux, Ib[k])
                self_feedback = SW_self[0][k]*PSC_TR[0][t]/number_neurons
                layer_S = SW_S[0][k]*PSC_S[0][t]/number_neurons
                layer_M = SW_M[0][k]*PSC_M[0][t]/number_neurons
                layer_D = SW_D[0][k]*PSC_D[0][t]/number_neurons
                layer_TC = SW_TC[0][k]*PSC_TC[0][t]/number_neurons
                layer_CI = SW_CI[0][k]*PSC_CI[0][t]/number_neurons
                noise = 0
                
                v[k][t] = v_aux + dt*(neuron_contribution + self_feedback + layer_S + layer_M + layer_D + layer_TC + layer_CI + noise)
                u[k][t] = u_aux + dt*dudt(v_aux, u_aux, a, b)
                
            # TM parameters
            tau_f = synapse_parameters['t_f']
            tau_d = synapse_parameters['t_d']
            U = synapse_parameters['U']
            A = synapse_parameters['distribution']
            tau_s = synapse_parameters['t_s']
            
            [rs, xs, Isyn, Ipost] = tm_synapse_eq(r, x, Is, AP_aux, tau_f, tau_d, tau_s, U, A)
            r = rs
            x = xs
            Is = Isyn
            
            if (np.isnan(v[k][t]) or np.isnan(u[k][t]) or np.isinf(v[k][t]) or np.isinf(u[k][t])):
                print('NaN or inf in t = ', t)
                break
            
            Isi.append(Ipost) 
            
        PSC_TR = np.sum(Isi, axis=0).reshape(1,len(time_vector))
    
    return PSC_TR, Is, AP, v, u, r, x
