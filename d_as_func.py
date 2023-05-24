"""
Cortical Layer D

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
    - Cortical Interneurons (CI)

Receive excitatory stimulus from:
    - Supraganular Layer (S)
    - Thalamic Reticular Nucleus (TRN)

Send inhibitory stimulus to:
    - None
    
Send excitatory stimulus to:
    - Cortical Interneurons (CI)
    - Supraganular Layer (S)
    - Thalamic Reticular Nucleus (TRN)
    - Basal Ganglia Nucleus -> LATER
"""
import numpy as np

from model_functions import izhikevich_dvdt, izhikevich_dudt, tm_synapse_eq

def d_cells(
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
        W_M,
        W_S,
        W_D,
        W_TR,
        W_TC,
        W_CI,
        PSC_S,
        PSC_M,
        PSC_D,
        PSC_TR,
        PSC_CI,
        PSC_D_TC,
        PSC_D_D,
        PSC_D_F,
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
        r_F,
        x_F,
        Is_F,
        tau_f,
        tau_d,
        tau_s,
        U,
        A,
        A_F,
        A_D,
        vr, 
        vp,
        dt,
        fired,
     ):
      
     Ise = np.zeros((1,n_neurons))
     Ise_F = np.zeros((1,n_neurons))
     Ise_D = np.zeros((1,n_neurons))

     for k in range(0, n_neurons):   
         AP_aux = 0
         v_aux = voltage[k][t - 1]
         u_aux = u[k][t - 1]
         I_aux = current[k]
         white_gausian_aux = a_wg_noise[k][t - 1]
         
         if (n_affected == 0):
             I_dbss = 0
         else:
             if (k >= 1 and k <= n_affected):
                 I_dbss = I_dbs[0][t - 1]
             else:
                 I_dbss = I_dbs[1][t - 1]
             
         neuron_contribution = izhikevich_dvdt(v = v_aux, u = u_aux, I = I_aux)
         self_feedback = W_D[k][0]*PSC_D[0][t - td_wl - td_syn]/n_neurons
         layer_S = W_S[k][0]*PSC_S[0][t - td_bl - td_syn]/n_neurons
         layer_M = W_M[k][0]*PSC_M[0][t - td_bl - td_syn]/n_neurons
         layer_TC = W_TC[k][0]*PSC_D_TC[0][t - td_tc - td_syn]/n_neurons
         layer_TR = W_TR[k][0]*PSC_TR[0][t - td_tc - td_syn]/n_neurons
         layer_CI = W_CI[k][0]*PSC_CI[0][t - td_wl - td_syn]/n_neurons
         noise = I_dbss + t_wg_noise[k][t - 1] + poisson_background_E[t - td_wl - td_syn] - poisson_background_I[t - td_wl - td_syn]
         
         voltage[k][t] = v_aux + dt*(
             neuron_contribution + 
             self_feedback + 
             layer_TC + layer_S + layer_M + layer_TR + layer_CI + 
             noise
             )
         u[k][t] = u_aux + dt*izhikevich_dudt(v = v_aux, u = u_aux, a = a[0][k], b = b[0][k])
         
         if (v_aux >= (vp + white_gausian_aux)):
             AP_aux = 1
             v_aux = vp + white_gausian_aux
             voltage[k][t] = c[0][k]
             u[k][t] = u_aux + d[0][k]
         
         rr = r; xx = x; Iss = Is;
         # Pseudo Linear
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
         Ise[0][k] = Ipost
         
         # Facilitating 
         [rsf, xsf, Isynf, Ipostf] = tm_synapse_eq(r = rr, 
                                                   x = xx, 
                                                   Is = Iss, 
                                                   AP = AP_aux, 
                                                   tau_f = tau_f, 
                                                   tau_d = tau_d, 
                                                   tau_s = tau_s, 
                                                   U = U, 
                                                   A = A_F,
                                                   dt = dt)
         rf = rsf
         xf = xsf
         Isf = Isynf
         Ise_F[0][k] = Ipostf
         
         # Depressing
         [rsd, xsd, Isynd, Ipostd] = tm_synapse_eq(r = r, 
                                                   x = x, 
                                                   Is = Is, 
                                                   AP = AP_aux, 
                                                   tau_f = tau_f, 
                                                   tau_d = tau_d, 
                                                   tau_s = tau_s, 
                                                   U = U, 
                                                   A = A_D,
                                                   dt = dt)
         rf = rsd
         xf = xsd
         Isf = Isynd
         Ise_D[0][k] = Ipostd
         
         fired[k][t] = AP_aux
         
     
     PSC_D[0][t] = np.sum(Ipost)
     PSC_D_D[0][t] = np.sum(Ipostd)
     PSC_D_F[0][t] = np.sum(Ipostf)
      
     return r, x, Is, rf, xf, Isf, voltage, u, fired, PSC_D, PSC_D_F, PSC_D_D
