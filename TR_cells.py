"""
Created on Mon Feb  6 20:31:49 2023

@author: Celine Soeiro

@description: Thalamic Reticular Nucleus (TR) cells

# Abreviations:
    PSC: Post Synaptic Current
    IC: Intercortical Neurons
    SW: Synaptic Weight
    S: Surface (Supragranular) layer
    M: Middle (Granular) layer
    D: Deep (Infragranular) layer
    CI: Cortical Interneurons
    TR: Thalamic Reticular Nucleus
    TC: Thalamo-Cortical Relay nucleus
    PD: Parkinsonian Desease

Inputs:
    time step: dt
    peak voltage: vp
    rest voltage: vr
    simulation steps: sim_steps
    number of neurons: n
    neuron_params: a,b,c,d                                                          -> Izhikevich
    membrane recovery variable: v                                                   -> Izhikevich
    membrane potential of the neuron: u                                             -> Izhikevich
    available neurotransmitter resources ready to be used: r                        -> TM model (u in original article)
    neurotransmitter resources that remain available after synaptic transmission: x -> TM model
    post-synaptic current: I                                                        -> TM model
    PSC self contribution: PSC_self
    PSC layer S: PSC_S
    PSC layer M: PSC_M
    PSC layer D: PSC_D
    PSC layer TC: PSC_TC
    PSC CI: PSC_CI
    SW from self: SW_self
    SW from S: SW_S
    SW from M: SW_M
    SW from D: SW_D
    SW from CI: SW_CI
    SW from TC: SW_TC 
    bias current: Ib
"""

import numpy as np

from model_parameters import TCM_model_parameters, coupling_matrix_normal

# =============================================================================
# INITIAL VALUES
# =============================================================================
global_parameters = TCM_model_parameters()['model_global_parameters']
neuron_quantities = TCM_model_parameters()['neuron_quantities']
neuron_params = TCM_model_parameters()['neuron_paramaters']

facilitating_factor_N = global_parameters['connectivity_factor_normal_condition']
facilitating_factor_PD = global_parameters['connectivity_factor_PD_condition']

dt = global_parameters['dt']
sim_steps = global_parameters['simulation_steps']
vr = global_parameters['vr']
vp = global_parameters['vp']
n = neuron_quantities['TR']
neuron_params = neuron_params['TR1']
v = vr*np.ones((n,sim_steps))
u = 0*v
r = np.zeros((3,1))
x = np.ones((3,1))
I = np.zeros((3,1))
PSC_self = np.zeros((1,sim_steps))
PSC_S = np.zeros((1,sim_steps))
PSC_M = np.zeros((1,sim_steps))
PSC_D = np.zeros((1,sim_steps))
PSC_TC = np.zeros((1,sim_steps))
PSC_CI = np.zeros((1,sim_steps))

n_s = neuron_quantities['S']
n_m = neuron_quantities['M']
n_d = neuron_quantities['D']
n_ci = neuron_quantities['CI']
n_tc = neuron_quantities['TC']
W_N = coupling_matrix_normal(facilitating_factor_N, n_s, n_m, n_d, n_ci, n_tc, n)['weights']

SW_self = W_N['W_II_tr']
SW_S = W_N['W_IE_tr_s']
SW_M = W_N['W_IE_tr_m']
SW_D = W_N['W_IE_tr_d']
SW_CI = W_N['W_II_tr_ci']
SW_TC = W_N['W_IE_tr_tc']

Ib = 0.6 + 0.1*np.ones(8)

# =============================================================================
# CALCULATING THE NEW VALUE
# =============================================================================

# Izhikevich neuron equations
def dvdt(v, u, I):
    return 0.04*v**2 + 5*v + 140 - u + I

def dudt(v,u,a,b):
    return a*(b*v - u)

for k in range(n):
    v[k] = v[k] + dt*dvdt(v[k], u[k], Ib[k])