"""
Cortical Layer M

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

Send inhibitory stimulus to:
    - None
    
Send excitatory stimulus to:
    - Cortical Interneurons (CI)
    - Supraganular Layer (S)
"""

import numpy as np

def m_cells(
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
    tm_synapse_eq,
    synapse_parameters,
    PSC_S,
    PSC_M,
    PSC_D,
    PSC_TR,
    PSC_TC,
    PSC_CI,         
    neuron_type,
    random_factor,
    ):
    
    v = vr*np.ones((number_neurons,simulation_steps))
    u = 0*v
    r = np.zeros((3,len(time_vector)))
    x = np.ones((3,len(time_vector)))
    Is = np.zeros((3,len(time_vector)))
    
    SW_self = coupling_matrix['W_EE_m']
    SW_S = coupling_matrix['W_EE_m_s']
    SW_D = coupling_matrix['W_EE_m_d']
    SW_CI = coupling_matrix['W_EI_m_ci']
    SW_TC = coupling_matrix['W_EE_m_tc']
    SW_TR = coupling_matrix['W_EI_m_tr']
 
    Isi = []
    Ib = current + Idc*np.ones(number_neurons)
    
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
                self_feedback = SW_self[0][k]*PSC_S[0][t]/number_neurons
                layer_CI = SW_CI[0][k]*PSC_CI[0][t]/number_neurons
                layer_S = SW_S[0][k]*PSC_S[0][t]/number_neurons
                layer_D = SW_D[0][k]*PSC_D[0][t]/number_neurons
                layer_TC = SW_TC[0][k]*PSC_TC[0][t]/number_neurons
                layer_TR = SW_TR[0][k]*PSC_TR[0][t]/number_neurons
                noise = 0
                
                v[k][t] = v_aux + dt*(neuron_contribution + self_feedback + layer_CI + layer_S + layer_D + layer_TC + layer_TR + noise)
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
            
        PSC_M = np.sum(Isi, axis=0).reshape(1,len(time_vector))
    
    return PSC_M, Is, AP, v, u, r, x
