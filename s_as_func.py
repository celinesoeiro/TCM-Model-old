"""
Cortical Layer S

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
    - Deep Layer (D)
    - Granular Layer (M)

Send inhibitory stimulus to:
    - None
    
Send excitatory stimulus to:
    - Deep Layer (D)
    - Cortical Interneurons (CI)
    - Granular Layer (M)
"""

import numpy as np

def s_cells(
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
    
    SW_self = coupling_matrix['W_EE_s']
    SW_M = coupling_matrix['W_EE_s_m']
    SW_D = coupling_matrix['W_EE_s_d']
    SW_CI = coupling_matrix['W_EI_s_ci']
    SW_TC = coupling_matrix['W_EE_s_tc']
    SW_TR = coupling_matrix['W_EI_s_tr']
 
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
                self_feedback = SW_self[k][0]*PSC_S[0][t]/number_neurons
                layer_CI = SW_CI[k][0]*PSC_CI[0][t]/number_neurons
                layer_M = SW_M[k][0]*PSC_M[0][t]/number_neurons
                layer_D = SW_D[k][0]*PSC_D[0][t]/number_neurons
                layer_TC = SW_TC[k][0]*PSC_TC[0][t]/number_neurons
                layer_TR = SW_TR[k][0]*PSC_TR[0][t]/number_neurons
                noise = 0
                
                v[k][t] = v_aux + dt*(neuron_contribution + self_feedback + layer_CI + layer_M + layer_D + layer_TC + layer_TR + noise)
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

    PSC_S = I_post_synaptic
    
    return PSC_S, AP, v, u, r, x