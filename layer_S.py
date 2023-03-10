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
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

sns.set()

from model_parameters import TCM_model_parameters, coupling_matrix_normal

# =============================================================================
# INITIAL VALUES
# =============================================================================
global_parameters = TCM_model_parameters()['model_global_parameters']
neuron_quantities = TCM_model_parameters()['neuron_quantities']
neuron_params = TCM_model_parameters()['neuron_paramaters']
currents = TCM_model_parameters()['currents_per_structure']

facilitating_factor_N = global_parameters['connectivity_factor_normal_condition']
facilitating_factor_PD = global_parameters['connectivity_factor_PD_condition']

dt = global_parameters['dt']
sim_steps = global_parameters['simulation_steps']
time = global_parameters['time_vector']
vr = global_parameters['vr']
vp = global_parameters['vp']
chop_till = global_parameters['chop_till']
Idc = global_parameters['Idc']

n = neuron_quantities['S']
n_m = neuron_quantities['M']
n_d = neuron_quantities['D']
n_ci = neuron_quantities['CI']
n_tn = neuron_quantities['TC']
n_tr = neuron_quantities['TR']

neuron_params = neuron_params['S1']

v = vr*np.ones((n,sim_steps))
u = 0*v
r = np.zeros((3,len(time)))
x = np.ones((3,len(time)))
I = np.zeros((3,len(time)))
PSC_self = np.zeros((1,sim_steps))
PSC_M = np.zeros((1,sim_steps))
PSC_D = np.zeros((1,sim_steps))
PSC_TR = np.zeros((1,sim_steps))
PSC_TN = np.zeros((1,sim_steps))
PSC_CI = np.zeros((1,sim_steps))

W_N = coupling_matrix_normal(    
    facilitating_factor = facilitating_factor_N, 
    n_s = n, 
    n_m = n, 
    n_d = n_d, 
    n_ci = n_ci, 
    n_tn = n_tn, 
    n_tr = n_tr)['weights']

SW_self = W_N['W_EE_s']
SW_M = W_N['W_EE_s_m']
SW_D = W_N['W_EE_s_d']
SW_CI = W_N['W_EI_s_ci']
SW_TR = W_N['W_EI_s_ret']
SW_TN = W_N['W_EE_s_rel']

Ib = currents['I_S_1'] + Idc*np.ones(n)

# =============================================================================
# CALCULATING THE NEW VALUE
# =============================================================================

# Izhikevich neuron equations
def dvdt(v, u, I):
    return 0.04*v**2 + 5*v + 140 - u + I

def dudt(v, u, a, b):
    return a*(b*v - u)

# TM synapse
def r_eq(r, t_f, U, fired):
    # fraction of available neurotransmitter resources ready to be used
    return -(r/t_f) + U*(1 - r)*fired

def x_eq(x, t_d, r, U, fired):
    # fraction of the neurotransmitter resources that remain available after synaptic transmission
    return (1 - x)/t_d - (r + U*(1 - r))*x*fired
    
def I_eq(I, t_s, A, U, x, r, fired):
    # post-synaptic current
    return -(I/t_s) + A*(r + U*(1 - r))*x*fired

def getParamaters(synapse_type: str):
    if (synapse_type == 'excitatory'):
        return {
            # [Facilitating, Depressing, Pseudo-linear]
            't_f': [670, 17, 326],
            't_d': [138, 671, 329],
            'U': [0.09, 0.5, 0.29],
            'distribution': [0.2, 0.63, 0.17],
            't_s': 3,
        };
    elif (synapse_type == 'inhibitory'):
        return {
            # [Facilitating, Depressing, Pseudo-linear]
            't_f': [376, 21, 62],
            't_d': [45, 706, 144],
            'U': [0.016, 0.25, 0.32],
            'distribution': [0.08, 0.75, 0.17],
            't_s': 3,
        };
    
    else:
        return 'Invalid synapse_type. Synapse_type must be excitatory or inhibitory.'
    

AP = np.zeros((1,len(time)))

for t in range(1, len(time)):
    AP_aux = AP[0][t]
    for k in range(1, n):        
        v_aux = v[k - 1][t - 1]
        u_aux = u[k - 1][t - 1]
        
        if (v_aux >= vp):
            AP_aux = 1
            v[k][t] = vp
            v[k][t] = neuron_params['c']
            u[k][t] = u[k][t] + neuron_params['d']
        else:
            neuron_contribution = dvdt(v_aux, u_aux, Ib[k])
            self_feedback = SW_self[k][0]*PSC_self[0][t]/n
            layer_M = SW_M[k][0]*PSC_M[0][t]/n
            layer_D = SW_D[k][0]*PSC_D[0][t]/n
            layer_TR = SW_TR[k][0]*PSC_TR[0][t]/n
            layer_TN = SW_TN[k][0]*PSC_TN[0][t]/n
            layer_CI = SW_CI[k][0]*PSC_CI[0][t]/n
            noise = 0
            
            v[k][t] = v_aux + dt*(neuron_contribution + self_feedback + layer_M + layer_D + layer_TR + layer_TN + layer_CI + noise)
            u[k][t] = u_aux + dt*dudt(v_aux, u_aux, neuron_params['a'], neuron_params['b'])
            
        # TM parameters
        tau_f = getParamaters('excitatory')['t_f']
        tau_d = getParamaters('excitatory')['t_d']
        U = getParamaters('excitatory')['U']
        A = getParamaters('excitatory')['distribution']
        tau_s = getParamaters('excitatory')['t_s']
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

PSC_self = I_post_synaptic
    
# Plotting
# indexes = np.arange(0,40, dtype=object)
# for k in range(n):
#     indexes[k] = "neuron " + str(k)
        
# v_RT = pd.DataFrame(v.transpose())
    
