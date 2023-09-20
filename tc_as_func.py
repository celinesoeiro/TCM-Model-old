"""
@author: Celine Soeiro

@description: Thalamo-cortical Relay Nucleus (TC) function
    
-- OVERVIEW
Receive inhibitory stimulus from:
    - Thalamic Reticular Nucleus (TR) 

Receive excitatory stimulus from:
    - Layer D

Send inhibitory stimulus to:
    - None
    
Send excitatory stimulus to:
    - Layer D
    - Thalamic Reticular Nucleus (TR) 

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
    r_D: available neurotransmitter resources ready to be used from Thalamus to Layer D
    x_D: neurotransmitter resources that remain available after synaptic transmission from Thalamus to Layer D
    I_syn_D: post-synaptic current from Thalamus to Layer D
    tau_f_D:
    tau_d_D:
    tau_s_D:
    U_D: 
    A_D: Distribution from Thalamus to Layer D
    ------------------------ TCM Model
    PSC_S: Post Synaptic Current from layer S
    PSC_M: Post Synaptic Current from Layer M
    PSC_D: Post Synaptic Current from layer D
    PSC_TC: Post Synaptic Current from TC (Self)
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
    Is: post-synaptic current
    PSC_TC: Post Synaptic Current from TC (Self)
    v: membrane voltage 
    u: membrane recovery variable  
"""
import numpy as np

from model_functions import izhikevich_dvdt, izhikevich_dudt, tm_synapse_eq

def tc_cells(        
       t,
       dt,
       n_neurons, 
       sim_steps,
       v,
       u,
       I_dc, 
       vp,
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
       r_D,
       x_D,
       I_syn_D,
       A_D,
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
       zeta_noise,
       kisi_noise,
       poisson_background_E,
       poisson_background_I,
       n_affected,
       I_dbs,
       spikes,
    ):
     
    Isi = np.zeros((1, n_neurons))
    Isi_D = np.zeros((1, n_neurons))

    for k in range(0, n_neurons):   
        AP_aux = 0
        v_aux = 1*v[k][t - 1]
        u_aux = 1*u[k][t - 1]
        I_aux = I_dc[k]
        white_gausian_aux = zeta_noise[k][t - 1]
        
        if (k >= 1 and k <= (n_affected - 1)):
            I_dbss = I_dbs
        else:
            I_dbss = 0
            
        neuron_contribution = izhikevich_dvdt(v = v_aux, u = u_aux, I = I_aux)
        self_feedback = W_TC[k][0]*PSC_TC/n_neurons
        layer_S = W_S[k][0]*PSC_S/n_neurons
        layer_M = W_M[k][0]*PSC_M/n_neurons
        layer_D = W_D[k][0]*PSC_D/n_neurons
        layer_TR = W_TR[k][0]*PSC_TR/n_neurons
        layer_CI = W_CI[k][0]*PSC_CI/n_neurons
        noise = kisi_noise[k][t - 1] + poisson_background_E - poisson_background_I
        
        v[k][t] = v_aux + dt*(
            neuron_contribution + 
            self_feedback + 
            layer_S + layer_M + layer_D + layer_TR + layer_CI + 
            noise +
            I_dbss
            )
        u[k][t] = u_aux + dt*izhikevich_dudt(v = v_aux, u = u_aux, a = a[0][k], b = b[0][k])
        
        if (v[k][t] >= (vp + white_gausian_aux)):
            AP_aux = 1
            v_aux = vp + white_gausian_aux
            v[k][t] = c[0][k]
            u[k][t] = u_aux + d[0][k]
            spikes[k][t] = t
        
        rr = 1*r;     xx = 1*x;     Iss = 1*I_syn;

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
        r_D = 1*tm_syn_inst_dep['r']; 
        x_D = 1*tm_syn_inst_dep['x']; 
        I_syn_D = 1*tm_syn_inst_dep['Is'];
        Isi_D[0][k] = 1*tm_syn_inst_dep['Ipost'];
        
    tc_neurons = dict()
    
    tc_neurons['r'] = r
    tc_neurons['x'] = x
    tc_neurons['I_syn'] = I_syn
    tc_neurons['PSC_TC'] = np.sum(Isi[0])
    tc_neurons['v'] = v
    tc_neurons['u'] = u
    tc_neurons['r_D'] = r_D
    tc_neurons['x_D'] = x_D
    tc_neurons['I_syn_D'] = I_syn_D
    tc_neurons['PSC_D'] = np.sum(Isi_D[0])

    return tc_neurons
    