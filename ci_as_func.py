"""
Cortical Interneurons (CI)

@author: Celine Soeiro

Valores de saida
    Inh_AP = vRet(:,i+1)
    Inh_Aux = uRet(:,i+1)
    r = rI
    x = xI
    Is =IsI
    IPSC = IPSC_ret(i+1)

Valores de entrada
    a = aIret
    b = bIret
    c = cIret
    d = dIret
    n = nIret
    v = vRet(:,i)
    u = uRet(:,i)
    r = rIret
    x = xIret
    Is = IsIret
    IPSC = IPSC_ret(i-td_wL-td_syn)
    EPSCs = EPSCs(i-td_CT-td_syn)
    EPSCm = EPSCm(i-td_CT-td_syn)
    EPSCd = EPSCdF(i-td_CT-td_syn)
    IPSC_in = IPSC_INs(i-td_CT-td_syn)
    EPSC_rel = EPSC_rel(i-td_L-td_syn)
    W_II = W_IIret
    W_IErs = W_IE_Ret_s
    W_IErm = W_IE_Ret_m
    W_IErd = W_IE_Ret_d
    W_II_IN = W_II_Ret_INs
    W_IE_rel = W_IE_Ret_Rel
    I_psE = 0*I_ps(5,1,i-td_wL-td_syn)
    I_psI = 0*I_ps(5,2,i-td_wL-td_syn)
    kisi = kisiIret(:,i)+pnIret(:,i)
    zeta = zetaIret(:,i)
    Idc = Idc_Ret
    Idbs = fidN*I_dbs(2,i)
    n_affected = n_conn_N
    dt = dt

------------ OVERVIEW

Receive inhibitory stimulus from:
    - Self

Receive excitatory stimulus from:
    - Supraganular Layer (S)
    - Granular Layer (M)
    - Infragranular Layer (D)
    - Thalamic Reticular Nucleus (TRN)

Send inhibitory stimulus to:
    - Supraganular Layer (S)
    - Granular Layer (M)
    - Infragranular Layer (D)
    
Send excitatory stimulus to:
    - None
"""

import numpy as np

from model_functions import izhikevich_dvdt, izhikevich_dudt, tm_synapse_eq

def ci_cells(
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
        td_tc,
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
        fired,
        spikes,
     ):
      
     Isi = np.zeros((1,n_neurons))

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
         self_feedback = W_CI[k][0]*PSC_CI[0][t - td_wl - td_syn]/n_neurons
         layer_S = W_S[k][0]*PSC_S[0][t - td_wl - td_syn]/n_neurons
         layer_M = W_M[k][0]*PSC_M[0][t - td_wl - td_syn]/n_neurons
         layer_D = W_D[k][0]*PSC_D[0][t - td_wl - td_syn]/n_neurons
         layer_TR = W_TR[k][0]*PSC_TR[0][t - td_tc - td_syn]/n_neurons
         layer_TC = W_TC[k][0]*PSC_TC[0][t - td_tc - td_syn]/n_neurons
         noise = I_dbss + t_wg_noise[k][t - 1] + poisson_background_E[t - td_wl - td_syn] - poisson_background_I[t - td_wl - td_syn]
         
         voltage[k][t] = v_aux + dt*(
             neuron_contribution + 
             self_feedback + 
             layer_S + layer_M + layer_D + layer_TR + layer_TC + 
             noise
             )
         u[k][t] = u_aux + dt*izhikevich_dudt(v = v_aux, u = u_aux, a = a[0][k], b = b[0][k])
         
         if (v_aux >= (vp + white_gausian_aux)):
             AP_aux = 1
             v_aux = vp + white_gausian_aux
             voltage[k][t] = c[0][k]
             u[k][t] = u_aux + d[0][k]
             spikes[k][t] = t
         
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
         
         fired[k][t] = AP_aux
         
         
     PSC_CI[0][t] = np.sum(Ipost)

     return r, x, Is, PSC_CI, voltage, u, fired
