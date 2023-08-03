"""
@author: Celine Soeiro

@description: Layer D function
    
-- OVERVIEW
Receive inhibitory stimulus from:
    - Self 
    - CI

Receive excitatory stimulus from:
    - Layer S
    - TC nucleus

Send inhibitory stimulus to:
    - None
    
Send excitatory stimulus to:
    - Layer S
    - CI
    - TR nucleus
    - TC nucleus
Receive DBS input

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
    r_F: available neurotransmitter resources ready to be used from Layer D to Thalamus (Facilitating)
    x_F: neurotransmitter resources that remain available after synaptic transmission from Layer D to Thalamus (Facilitating)
    I_syn_F: post-synaptic current from Layer D to Thalamus (Facilitating)
    tau_f_F:
    tau_d_F:
    tau_s_F:
    U_F: 
    A_F: Distribution from Layer D to Thalamus (Facilitating)
    A_D: Distribution from Thalamus to Layer D (Depressing)
    ------------------------ TCM Model
    PSC_S: Post Synaptic Current from layer S 
    PSC_M: Post Synaptic Current from Layer M 
    PSC_D: Post Synaptic Current from layer D (Self)
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
    PSC_D: Post Synaptic Current from Layer D (Self)
    v: membrane voltage 
    u: membrane recovery variable  
"""
import numpy as np

from model_functions import izhikevich_dvdt, izhikevich_dudt, tm_synapse_eq

def d_cells(
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
        U,
        A,
        r_F,
        x_F,
        I_syn_F,
        A_F,
        A_D,
        tau_f,
        tau_d,
        tau_s,
        PSC_S,
        PSC_M,
        PSC_D,
        PSC_TR,
        PSC_CI,
        PSC_D_TC,
        W_M,
        W_S,
        W_D,
        W_TR,
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
      
     Ise = np.zeros((1,n_neurons))
     Ise_F = np.zeros((1,n_neurons))
     Ise_D = np.zeros((1,n_neurons))

     for k in range(0, n_neurons):   
         AP_aux = 0
         v_aux = v[k][t - 1]
         u_aux = u[k][t - 1]
         I_aux = I_dc[k]
         white_gausian_aux = a_wg_noise[k][t - 1]
         
         if (n_affected == 0):
             I_dbss = 0
         else:
             if (k >= 0 and k <= n_affected):
                 I_dbss = I_dbs[0][t - 1]
             else:
                 I_dbss = I_dbs[1][t - 1]
             
         neuron_contribution = izhikevich_dvdt(v = v_aux, u = u_aux, I = I_aux)
         self_feedback = W_D[k][0]*PSC_D/n_neurons
         layer_S = W_S[k][0]*PSC_S/n_neurons
         layer_M = W_M[k][0]*PSC_M/n_neurons
         layer_TC = W_TC[k][0]*PSC_D_TC/n_neurons
         layer_TR = W_TR[k][0]*PSC_TR/n_neurons
         layer_CI = W_CI[k][0]*PSC_CI/n_neurons
         noise = t_wg_noise[k][t - 1] + poisson_background_E - poisson_background_I
         
         v[k][t] = v_aux + dt*(
             neuron_contribution + 
             self_feedback + 
             layer_TC + layer_S + layer_M + layer_TR + layer_CI + 
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
         
         rr = r; xx = x; Iss = I_syn;
         # Pseudo Linear
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
         r = tm_syn_inst['r']; x = tm_syn_inst['x']; I_syn = tm_syn_inst['Is'];
         Ise[0][k] = tm_syn_inst['Ipost'];
         
         # Facilitating 
         tm_syn_inst_fac = tm_synapse_eq(r = rr, 
                                        x = xx, 
                                        Is = Iss, 
                                        AP = AP_aux, 
                                        tau_f = tau_f, 
                                        tau_d = tau_d, 
                                        tau_s = tau_s, 
                                        U = U, 
                                        A = A_F,
                                        dt = dt)
         rf = tm_syn_inst_fac['r']; xf = tm_syn_inst_fac['x']; Isf = tm_syn_inst_fac['Is'];
         Ise_F[0][k] = tm_syn_inst_fac['Ipost']
         
         # Depressing
         tm_syn_inst_dep = tm_synapse_eq(r = rr, 
                                        x = xx, 
                                        Is = Iss, 
                                        AP = AP_aux, 
                                        tau_f = tau_f, 
                                        tau_d = tau_d, 
                                        tau_s = tau_s, 
                                        U = U, 
                                        A = A_D,
                                        dt = dt)
         Ise_D[0][k] = tm_syn_inst_dep['Ipost']
         
     d_neurons = dict()
     
     d_neurons['r'] = r
     d_neurons['x'] = x
     d_neurons['I_syn'] = I_syn
     d_neurons['PSC_D'] = np.sum(Ise[0])
     d_neurons['v'] = v
     d_neurons['u'] = u
     d_neurons['r_F'] = rf
     d_neurons['x_F'] = xf
     d_neurons['I_syn_F'] = Isf
     d_neurons['PSC_D_T'] = np.sum(Ise_F[0])
     d_neurons['PSC_T_D'] = np.sum(Ise_D[0])
     
     return d_neurons
