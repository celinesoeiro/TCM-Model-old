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
       coupling_matrix, 
       current, 
       vr, 
       vp,
       dt,
       t,
       v,
       u,
       Idc_tune,
       dvdt,
       dudt,
       r_eq,
       x_eq,
       I_eq,
       tm_synapse_eq,
       synapse_parameters,
       r,
       x,
       Is,
       PSC_S,
       PSC_M,
       PSC_D,
       PSC_TR,
       PSC_TC,
       PSC_CI,    
       neuron_type,
       random_factor,
       a,
       b,
       c,
       d
    ):
    
    SW_self = coupling_matrix['W_II_tr']
    SW_S = coupling_matrix['W_IE_tr_s']
    SW_M = coupling_matrix['W_IE_tr_m']
    SW_D = coupling_matrix['W_IE_tr_d']
    SW_TC = coupling_matrix['W_IE_tr_tc']
    SW_CI = coupling_matrix['W_II_tr_ci']
 
    Isi = []
    Ib = current + Idc_tune*np.ones(number_neurons)

    # for t in range(1, len(time_vector)):
    AP_aux = 0
    for k in range(1, number_neurons):        
        v_aux = v[k - 1][t - 1]
        u_aux = u[k - 1][t - 1]
        
        if (v_aux >= vp):
            AP_aux = 1
            v[k][t] = vp
            v[k][t] = c[0][k]
            u[k][t] = u[k][t] + d[0][k]
        else:
            neuron_contribution = dvdt(v_aux, u_aux, Ib[k])
            self_feedback = SW_self[0][k]*PSC_TR/number_neurons
            layer_S = SW_S[0][k]*PSC_S/number_neurons
            layer_M = SW_M[0][k]*PSC_M/number_neurons
            layer_D = SW_D[0][k]*PSC_D/number_neurons
            layer_TC = SW_TC[0][k]*PSC_TC/number_neurons
            layer_CI = SW_CI[0][k]*PSC_CI/number_neurons
            noise = 0
            
            v[k] = v_aux + dt*(neuron_contribution + self_feedback + layer_S + layer_M + layer_D + layer_TC + layer_CI + noise)
            u[k] = u_aux + dt*dudt(v_aux, u_aux, a[0][k], b[0][k])
            
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
        
        Isi.append(Ipost) 
            
    PSC_TR = np.sum(Isi)

    Inhibitory_AP = v
    Inhibitory_aux = u

    return Inhibitory_AP, Inhibitory_aux, r, x, Is, PSC_TR
