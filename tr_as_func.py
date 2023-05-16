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

from model_functions import izhikevich_dvdt, izhikevich_dudt, tm_synapse_eq

def tr_cells(
       t,
       n_neurons, 
       sim_steps,
       voltage,
       u,
       current, 
       a_wg_noise,
       t_wg_noise,
       n_affected,
       synaptic_fidelity,
       I_dbs,
       W_TR,
       W_S,
       W_M,
       W_D,
       W_TC,
       W_CI,
       PSC_S,
       PSC_M,
       PSC_D,
       PSC_TC,
       PSC_TR,
       PSC_CI,
       td_wl,
       td_syn,
       td_ct,
       td_bl,
       a,
       b,
       c,
       d,
       r,
       x,
       Is,
       tau_f,
       tau_d,
       tau_s,
       U,
       A,
       vr, 
       vp,
       dt,
    ):
     
    Isi = np.zeros((1,n_neurons))
    fired = np.zeros((n_neurons,sim_steps))

    for k in range(0, n_neurons):   
        AP_aux = 0
        v_aux = voltage[k][t - 1]
        u_aux = u[k][t - 1]
        I_aux = current[k]
        white_gausian_aux = a_wg_noise[k][t - 1]
        
        if (k >= 1 and k <= n_affected):
            I_dbss = synaptic_fidelity*I_dbs[1][t - 1]
        else:
            I_dbss = 0
            
        neuron_contribution = izhikevich_dvdt(v = v_aux, u = u_aux, I = I_aux)
        self_feedback = W_TR[k][0]*PSC_TR[0][t - td_wl - td_syn]/n_neurons
        layer_S = W_S[k][0]*PSC_S[0][t - td_ct - td_syn]/n_neurons
        layer_M = W_M[k][0]*PSC_M[0][t - td_ct - td_syn]/n_neurons
        layer_D = W_D[k][0]*PSC_D[0][t - td_ct - td_syn]/n_neurons
        layer_TC = W_TC[k][0]*PSC_TC[0][t - td_wl - td_syn]/n_neurons
        layer_CI = W_CI[k][0]*PSC_CI[0][t - td_ct - td_syn]/n_neurons
        noise = I_dbss + t_wg_noise[k][t - 1]
        
        voltage[k][t] = v_aux + dt*(
            neuron_contribution + 
            self_feedback + 
            layer_S + layer_M + layer_D + layer_TC + layer_CI + 
            noise
            )
        u[k][t] = u_aux + dt*izhikevich_dudt(v = v_aux, u = u_aux, a = a[0][k], b = b[0][k])
        
        if (v_aux >= (vp + white_gausian_aux)):
            AP_aux = 1
            v_aux = vp + white_gausian_aux
            voltage[k][t] = c[0][k]
            u[k][t] = u_aux + d[0][k]
            fired[k][t] = 1
        
        [rs, xs, Isyn, Ipost] = tm_synapse_eq(r = r, 
                                              x = x, 
                                              Is = Is, 
                                              AP = AP_aux, 
                                              tau_f = tau_f, 
                                              tau_d = tau_d, 
                                              tau_s = tau_s, 
                                              U = U, 
                                              A = A,
                                              dt = dt)
        r = rs
        x = xs
        Is = Isyn
            
        Isi[0][k] = Ipost 
    
    PSC_TR[0][t] = np.sum(Ipost)

    return r, x, Is, PSC_TR, voltage, u, fired
