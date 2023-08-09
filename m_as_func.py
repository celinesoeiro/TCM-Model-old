"""
@author: Celine Soeiro

@description: Layer M function
    
-- OVERVIEW
Receive inhibitory stimulus from:
    - Self 
    - CI

Receive excitatory stimulus from:
    - Layer S

Send inhibitory stimulus to:
    - None
    
Send excitatory stimulus to:
    - Layer S
    - CI

-- INPUTS:
    t: time
    dt: time step
    n_neurons: number of neurons in structure
    sim_steps: simulation steps
    ------------------------ Izhikevich Neuron Model
    v: membrane voltage 
    u: membrane recovery variable 
    I_dc: Bias current 
    a, b, c, d: Izhikevich neuron params
    vp: Peak voltage 
    ------------------------ TM Synapse Model
    r: available neurotransmitter resources ready to be used (u in original article)
    x: neurotransmitter resources that remain available after synaptic transmission 
    I_syn: post-synaptic current 
    tau_f: 
    tau_d:
    tau_s:
    U: 
    A: Distribution
    ------------------------ TCM Model
    PSC_S: Post Synaptic Current from layer S 
    PSC_M: Post Synaptic Current from Layer M (Self)
    PSC_D: Post Synaptic Current from layer D
    PSC_TC: Post Synaptic Current from TC
    PSC_TR: Post Synaptic Current from TR 
    PSC_CI: Post Synspatic Current frmo CI
    W_TR: Synaptic weight within TR neurons
    W_S: Synaptic weight between Layer S and TR neurons
    W_M: Synaptic weight between Layer M and TR neurons
    W_D: Synaptic weight between Layer D and TR neurons
    W_TC: Synaptic weight between TC and TR neurons
    W_CI: Synaptic weight between CI and TR neurons
    a_wg_noise: Additive white gaussian noise
    t_wg_noise: threshold white gaussian noise
    I_dbs: Postsynadisplay DBS pulses
    n_affected: Percentage of neurons that are connected with hyperdirect neurons

-- OUTPUTS:
    r: available neurotransmitter resources ready to be used (u in original article)
    x: neurotransmitter resources that remain available after synaptic transmission 
    I_syn: post-synaptic current
    PSC_M: Post Synaptic Current from Layer M (Self)
    v: membrane voltage 
    u: membrane recovery variable  
"""

import numpy as np

from model_functions import izhikevich_dvdt, izhikevich_dudt, tm_synapse_eq

def m_cells(
        t,
        dt,
        n_neurons, 
        sim_steps,
        v,
        u,
        I_dc, 
        a,
        b,
        c,
        d,
        r,
        x,
        I_syn,
        tau_f,
        tau_d,
        tau_s,
        U,
        A,
        PSC_S,
        PSC_M,
        PSC_D,
        PSC_TC,
        PSC_TR,
        PSC_CI,
        W_TR,
        W_S,
        W_M,
        W_D,
        W_TC,
        W_CI,
        a_wg_noise,
        t_wg_noise,
        poisson_background_E,
        poisson_background_I,
        n_affected,
        I_dbs,
        vp,
        spikes,
     ):
      
     Isi = np.zeros((1,n_neurons))

     for k in range(0, n_neurons):   
         AP_aux = 0
         v_aux = v[k][t - 1]
         u_aux = u[k][t - 1]
         I_aux = I_dc[k]
         white_gausian_aux = a_wg_noise[k][t - 1]
         
         if (k >= 1 and k <= (n_affected - 1)):
             I_dbss = I_dbs
         else:
             I_dbss = 0
             
         neuron_contribution = izhikevich_dvdt(v = v_aux, u = u_aux, I = I_aux)
         self_feedback = W_M[k][0]*PSC_M/n_neurons
         layer_S = W_S[k][0]*PSC_S/n_neurons
         layer_D = W_D[k][0]*PSC_D/n_neurons
         layer_TC = W_TC[k][0]*PSC_TC/n_neurons
         layer_TR = W_TR[k][0]*PSC_TR/n_neurons
         layer_CI = W_CI[k][0]*PSC_CI/n_neurons
         noise = t_wg_noise[k][t - 1] + poisson_background_E - poisson_background_I
         
         v[k][t] = v_aux + dt*(
             neuron_contribution + 
             self_feedback + 
             layer_TC + layer_S + layer_D + layer_TR + layer_CI + 
             noise +
             I_dbss
             )
         u[k][t] = u_aux + dt*izhikevich_dudt(v = v_aux, u = u_aux, a = a[0][k], b = b[0][k])
         
         if (v_aux >= (vp + white_gausian_aux)):
             AP_aux = 1
             v_aux = vp + white_gausian_aux
             v[k][t] = c[0][k]
             u[k][t] = u_aux + d[0][k]
             spikes[k][t] = t
         
         tm_syn_inst = tm_synapse_eq(r = r, 
                                    x = x, 
                                    Is = I_syn, 
                                    AP = AP_aux, 
                                    tau_f = tau_f, 
                                    tau_d = tau_d, 
                                    tau_s = tau_s, 
                                    U = U, 
                                    A = A,
                                    dt = dt)
         r = 1*tm_syn_inst['r']; 
         x = 1*tm_syn_inst['x']; 
         I_syn = 1*tm_syn_inst['Is'];
         Isi[0][k] = 1*tm_syn_inst['Ipost'];
     
     m_neurons = dict()
     
     m_neurons['r'] = r
     m_neurons['x'] = x
     m_neurons['I_syn'] = I_syn
     m_neurons['PSC_M'] = np.sum(Isi[0])
     m_neurons['v'] = v
     m_neurons['u'] = u

     return m_neurons