"""
Thalamic Reticular Nucleus (TRN) cells

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
    
    
Methods
- v = self feedback 
    + excitatory inputs from S, M and D layers plus relay cells 
    + inhibitory inputs from INs 
- u : izhikevich neuron logic
- 

"""


import numpy as np
from random import seed, random

# from izhikevich_neuron import izhikevich_neuron
from model_parameters import TCM_model_parameters, coupling_matrix_normal, coupling_matrix_PD

neuron_quantities = TCM_model_parameters()['neuron_quantities']
TCM_model = TCM_model_parameters()['model_global_parameters']
bias_current = TCM_model_parameters()['currents_per_structure']

dt = TCM_model['dt']      
n_sim = TCM_model['simulation_steps']   
n_tr = neuron_quantities['qnt_neurons_tr']
I_Ret = bias_current['I_Ret']

vr = -65

neuron_params = {
    'a': 0.02,
    'b': 0.25,
    'c': -65,
    'd':2.05,
    }

n_neurons = 40 # number of neurons

v = vr*np.ones((n_neurons, n_sim))
u = 0*v

# Synapse initial values
r = np.zeros((3,1))
x = np.zeros((3,1))
i = np.zeros((3,1))

IPSC = np.zeros((1, n_sim)) 
IPSC_INs = np.zeros((1, n_sim)) 

EPSC_s = np.zeros((1, n_sim))
EPSC_m = np.zeros((1, n_sim))
EPSC_df = np.zeros((1, n_sim))
EPSC_rel = np.zeros((1, n_sim))

def izhikevich_neuron(
        params: dict,
        neuron_type: str,
        voltage_pick: float,
        simulation_time: int,
        time_step: float,
        current: np.array,
        initial_voltage = -65,
        ):
    
    # Time grid
    time = np.arange(0, simulation_time + time_step, time_step)
    
    # Check if paramaters exists, if dont display error msg
    if (not params.get('a') 
        or not params.get('b') 
        or not params.get('c') 
        or not params.get('d')
        ): 
        return 'Parameters must be a, b, c and d' 
    
    # Parameters according Izhikevich article 
    seed(1)
    random_factor = random()
    
    if (neuron_type == 'excitatory' or 'excit'):
        a = params['a']
        b = params['b']
        c = params['c'] + 15*random_factor**2
        d = params['d'] - 6*random_factor**2
    elif (neuron_type == 'inhibitory' or 'inhib'):
        a = params['a'] + 0.08*random_factor
        b = params['b'] - 0.05*random_factor
        c = params['c']
        d = params['d']
    else:
        return 'Neuron type must be excitatory or inhibitory'
    
    # Current vector input
    I = current
    
    # membrane potential vector
    v = np.zeros(len(time))    
    v[0] = initial_voltage
    
    # membrane recovery variable vector
    u = np.zeros(len(time))    
    u[0] = b*v[0]
    
    # Izhikevich neuron equations
    def dvdt(v, u, I):
        return 0.04*v**2 + 5*v + 140 - u + I
    
    def dudt(v,u):
        return a*(b*v - u)
    
    # when the neuron fired vector
    fired = []
    
    for t in range(1, len(time)):     
        v_aux = v[t - 1]
        u_aux = u[t - 1]
        I_aux = I[t - 1]
        
        if (v_aux >= voltage_pick):
            v_aux = v[t]
            v[t] = c
            u[t] = u_aux + d
            fired.append(t)
           
        else:            
            # solve using Euler
            dv = dvdt(v_aux, u_aux, I_aux)
            du = dudt(v_aux, u_aux)
            v[t] = v_aux + dv*time_step
            u[t] = u_aux + du*time_step
            
    return v, I

tc_neuron_reticular, I_TR = izhikevich_neuron(
    params = {'a': 0.02, 'b': 0.25, 'c': -65, 'd': 2.05},
    current_value = I_Ret[0],
    current_start = 0,
    current_finish = -1, # go from 0 to max length
    voltage_pick = 30,
    simulation_time = n_tr,
    time_step = dt,
    neuron_type = 'inhibitory',
    )
