# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 21:24:07 2023

@author: Avell
"""
import numpy as np

from model_functions import izhikevich_dvdt, izhikevich_dudt, tm_synapse_eq

def tc_cells(
       t,
       n_neurons, 
       sim_steps,
       voltage,
       u,
       current, 
       a_wg_noise,
       t_wg_noise,
       poisson_background_E,
       poisson_background_I,
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
       r_D,
       x_D,
       I_syn_D,
       tau_f_D,
       tau_d_D,
       tau_s_D,
       U_D,
       A_D,
       fired,
    ):
     
    Isi = np.zeros((1, n_neurons))
    Isi_D = np.zeros((1, n_neurons))

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
        self_feedback = W_TC[k][0]*PSC_TC[0][t - td_wl - td_syn]/n_neurons
        layer_S = W_S[k][0]*PSC_S[0][t - td_ct - td_syn]/n_neurons
        layer_M = W_M[k][0]*PSC_M[0][t - td_ct - td_syn]/n_neurons
        layer_D = W_D[k][0]*PSC_D[0][t - td_ct - td_syn]/n_neurons
        layer_TR = W_TR[k][0]*PSC_TR[0][t - td_wl - td_syn]/n_neurons
        layer_CI = W_CI[k][0]*PSC_CI[0][t - td_ct - td_syn]/n_neurons
        noise = I_dbss + t_wg_noise[k][t - 1] + poisson_background_E[t - td_wl - td_syn] - poisson_background_I[t - td_wl - td_syn]
        
        voltage[k][t] = v_aux + dt*(
            neuron_contribution + 
            self_feedback + 
            layer_S + layer_M + layer_D + layer_TR + layer_CI + 
            noise
            )
        u[k][t] = u_aux + dt*izhikevich_dudt(v = v_aux, u = u_aux, a = a[0][k], b = b[0][k])
        
        if (v_aux >= (vp + white_gausian_aux)):
            AP_aux = 1
            v_aux = vp + white_gausian_aux
            voltage[k][t] = c[0][k]
            u[k][t] = u_aux + d[0][k]
        
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
        
        [rsD, xsD, IsynD, IpostD] = tm_synapse_eq(r = r_D, 
                                                  x = x_D, 
                                                  Is = I_syn_D, 
                                                  AP = AP_aux, 
                                                  tau_f = tau_f_D, 
                                                  tau_d = tau_d_D, 
                                                  tau_s = tau_s_D, 
                                                  U = U_D, 
                                                  A = A_D,
                                                  dt = dt)
        r_D = rsD
        x_D = xsD
        I_syn_D = IsynD
        
        Isi[0][k] = Ipost 
        Isi_D[0][k] = IpostD 
        
        fired[k][t] = AP_aux
    
    PSC_TC[0][t] = np.sum(Ipost)
    PSC_D[0][t] = np.sum(IpostD)

    return r, x, Is, PSC_TC, voltage, u, fired, r_D, x_D, I_syn_D, PSC_D
    