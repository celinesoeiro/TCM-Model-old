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


------------ OVERVIEW

Receive inhibitory stimulus from:
    - Self 

Receive excitatory stimulus from:
    - D layer
    - Thalamo-cortical relay nucleus (TCR)

Send inhibitory stimulus to:
    - Thalamo-cortical relay nucleus (TCR)
    
Send excitatory stimulus to:
    
------------ EQUATION (following Izhikevich neuron + Tsodyks & Markram synaptic model)

v/u = inhibitory self feedback 
    + excitatory inputs from D layer 
    + excitatory inputs from TCR


"""


import numpy as np

from utils import poissonSpikeGen
from tsodyks_markram_synapse import TM_Synapse

from izhikevich_neuron_instantaneous import izhikevich_neuron_instaneous
from model_parameters import TCM_model_parameters, coupling_matrix_normal

neuron_quantities = TCM_model_parameters()['neuron_quantities']
TCM_model = TCM_model_parameters()['model_global_parameters']
bias_current = TCM_model_parameters()['currents_per_structure']
noise = TCM_model_parameters()['noise']
random_factor = TCM_model_parameters()['random_factor']

T = TCM_model['simulation_time_ms']                 # simulation time in ms
dt = TCM_model['dt']                                # step
n_sim = TCM_model['simulation_steps']               # Number of simulation steps
td_syn = TCM_model['transmission_delay_synapse']    # Synaptic transmission delay 

n_neurons = neuron_quantities['qnt_neurons_tr']     # Number of neurons

# =============================================================================
# Neuron parameters for Izhikevich neuron model
# =============================================================================
neuron_params = {
    'a': 0.02,
    'b': 0.25,
    'c': -65,
    'd':2.05,
    }

vp = 30             # peak voltage
vr = -65            # initial voltage
v = vr*np.ones((n_neurons, n_sim))
u = 0*v
I_Ret = bias_current['I_Ret']

# =============================================================================
# Synapse parameters for Tsodyks and Markram model
# =============================================================================
r = np.zeros(3)
x = np.zeros(3)
i = np.zeros(3)

# =============================================================================
# Coupling matrix
# =============================================================================
facilitating_factor_N = TCM_model['connectivity_factor_normal_condition']
n_s = neuron_quantities['qnt_neurons_s']
n_m = neuron_quantities['qnt_neurons_m']
n_d = neuron_quantities['qnt_neurons_d']
n_ci = neuron_quantities['qnt_neurons_ci']
n_tc = neuron_quantities['qnt_neurons_tc']
W = coupling_matrix_normal(facilitating_factor_N, n_s, n_m, n_d, n_ci, n_tc, n_neurons)['weights']

# =============================================================================
# Inhibitory self feedback
# =============================================================================
W_II = W['W_II_ret']            # Weight from inhibitory to inhibitory
IPSC = np.zeros((1, n_sim))     # Inhibitory Post Synaptic Current 

# =============================================================================
# excitatory inputs from layers S, M and D
# =============================================================================
W_IE_ret_s = W['W_IE_ret_s']    # Weight from inhibitory to excitatory - reticular to layer S
W_IE_ret_m = W['W_IE_ret_m']    # Weight from inhibitory to excitatory - reticular to layer M
W_IE_ret_d = W['W_IE_ret_d']    # Weight from inhibitory to excitatory - reticular to layer D
EPSC_s = np.zeros((1, n_sim))   # Excitatory Post Synaptic Current - layer S
EPSC_m = np.zeros((1, n_sim))   # Excitatory Post Synaptic Current - layer M
EPSC_d = np.zeros((1, n_sim))   # Excitatory Post Synaptic Current - layer D

# =============================================================================
# excitatory inputs from TCR
# =============================================================================
W_IE_ret_rel = W['W_IE_ret_rel']  # Weight from inhibitory to excitatory - reticular to relay
EPSC_rel = np.zeros((1, n_sim))   # Excitatory Post Synaptic Current - Relay cells

# =============================================================================
# Inhibitory inputs from Cortical Inter Neurons
# =============================================================================
W_II_ret_ci = W['W_II_ret_ci']  # Weight from inhibitory to inhibitory - reticular to cortical interneurons
IPSC_CI = np.zeros((1, n_sim))  # Inhibitory Post Synaptic Current - Cortical Interneurons

# =============================================================================
# POISSONIAN background activity (assuming they invade primary motor cortex from premotor and supplimentary motor areas
# =============================================================================
# Poissonian postsynaptic input to the E and I neurons for all layers

fr = 20 + 2*random_factor   # Poissonian firing frequency from other parts of the brain

[spikess, t_sp] = poissonSpikeGen(fr, T/1000, 1, dt/1000)
tps = np.argwhere(spikess == 1)[:,1]

[t_syn_E, I_PS_E] = TM_Synapse(
    t_event = tps, 
    n_sim = n_sim, 
    t_delay = td_syn, 
    dt = dt, 
    synapse_type = 'excitatory')

[t_syn_I, I_PS_I] = TM_Synapse(
    t_event = tps, 
    n_sim = n_sim, 
    t_delay = td_syn, 
    dt = dt, 
    synapse_type = 'inhibitory')

# =============================================================================
# Noise terms 
# =============================================================================
kisi_Ret_I = noise['kisi_Ret_I']
pn_Ret_I = noise['pn_Ret_I']
zeta_Ret_I = noise['zeta_Ret_I']

# =============================================================================
# Deep Brain Stimulation inputs (DBS)
# =============================================================================

# =============================================================================
# TSODYKS AND MARKRAM SYNAPSE
# =============================================================================

def x_eq(x, t_d, u, delta):
    # fraction of the neurotransmitter resources that remain available after synaptic transmission
    return -(1-x)/t_d - u*x*delta

def u_eq(u, t_f, U, delta):
    # fraction of available neurotransmitter resources ready to be used
    return -(u/t_f) + U*(1 - u)*delta
    
def I_eq(I, t_s, A, u, x, delta):
    # post-synaptic current
    return -(I/t_s) + A*u*x*delta

def TM_synapse_instantaneous(r, x, Is, dt, sp_event, synapse_type):
    if (synapse_type == 'excitatory'):
        # [Facilitating, Depressing, Pseudo-linear]
        t_f = [670, 17, 326]
        t_d = [138, 671, 329]
        U = [0.09, 0.5, 0.29]
        A = [0.2, 0.63, 0.17]
        tau_s = 3
    
    elif (synapse_type == 'inhibitory'):
        # [Facilitating, Depressing, Pseudo-linear]
        t_f = [376, 21, 62]
        t_d = [45, 706, 144]
        U = [0.016, 0.25, 0.32]
        A = [0.08, 0.75, 0.17]
        tau_s = 11
    
    else:
        return 'Invalid synapse_type. Synapse_type must be excitatory or inhibitory.'
    
    # Initial values
    u = np.zeros(3)
    x = np.ones(3)
    I = np.zeros(3)
    
    # Loop trhough the parameters
    for p in range(2):
        # Solve EDOs using Euler method 
        u[p + 1] = u[p] + dt*u_eq(u[p], t_f[p], U[p], sp_event[p])
        x[p + 1] = x[p] + dt*x_eq(x[p], t_d[p], u[p], sp_event[p])
        I[p + 1] = I[p] + dt*I_eq(I[p], tau_s, A[p], u[p], x[p], sp_event[p])
        
    # Concatenate the final current
    
    I_post_synaptic = np.sum(I)
    
    return u, x, I, I_post_synaptic
        
    
 
# =============================================================================
# SIMULATION
# =============================================================================

v = np.zeros(n_neurons)
u = np.zeros(n_neurons)
sp = np.zeros(n_neurons)
Isi= []

for i in range(n_neurons):
    # Self feedback
    
    izhikevich_i = I_Ret[0][i][0] + W_II[i][0]*IPSC[0][i]/n_neurons
    
    [izhikevich_v, izhikevich_u, c, d] = izhikevich_neuron_instaneous(
        params = neuron_params, 
        neuron_type = 'inhibitory',
        voltage_pick = vp, 
        time_step = dt, 
        current_value = izhikevich_i, 
        random_factor = random_factor
        )
    
    # Excitatory inputs
    excitatory_v = (W_IE_ret_s[i][0]*EPSC_s[0][i]/n_neurons) 
    + (W_IE_ret_m[i][0]*EPSC_m[0][i]/n_neurons) 
    + (W_IE_ret_d[i][0]*EPSC_d[0][i]/n_neurons) 
    + (W_IE_ret_rel[i][0]*EPSC_rel[0][i]/n_neurons)
    
    # Inhibitory inputs
    inhibitory_v = W_II_ret_ci[i][0]*IPSC_CI[0][i]/n_neurons
    
    # Background activity and noise
    background_v = I_PS_E[i] - I_PS_I[i] + kisi_Ret_I[0][i][0]
    
    # voltages
    v_aux = izhikevich_v + excitatory_v + inhibitory_v + background_v
    
    u_aux = izhikevich_u 
    
    zeta = zeta_Ret_I[0][i][0]
    
    print(zeta, v_aux, u_aux)
    
    if (v_aux >= (vp + zeta)):
        v[i] = vp + zeta
        v[i] = c
        u[i] = u_aux + d
        sp[i] = 1
    
    spikeI = sp
    [rs, xs, Isyn, Ipost] = TM_synapse_instantaneous(
        r = r, 
        x = x, 
        Is = i, 
        dt = dt, 
        sp_event = spikeI, 
        synapse_type = 'inhibitory'
    )
    r = rs
    x = xs
    Is = Isyn
    Isi.append(Ipost)
    sp[i] = 0
    
    
    