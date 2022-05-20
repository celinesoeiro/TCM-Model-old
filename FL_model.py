"""
@author: Celine Soeiro

@description: Replicate the FL model

Base code: https://senselab.med.yale.edu/ModelDB/showmodel.cshtml?model=266941&file=%2fTCM_ModelDB%2fThalamoCortical_Microcircuit_PD_DBS_ModelDB.m#tabs-2

Abbreviations:
    N -> normal condition
    PD -> Parkinsonian desease condition
"""

import numpy as np
from matplotlib import pyplot as plt
from random import seed, random

from utils import poissonSpikeGen
from tsodyks_markram_synapse import TM_Synapse
from dbs import DBS, dbsDelta

seed(1)
random_factor = random()

# =============================================================================
# IMPORT PARAMETERS
# =============================================================================
from model_parameters import TCM_model_parameters, coupling_matrix_normal, coupling_matrix_PD

TCM_model = TCM_model_parameters()['model_global_parameters']
neurons_per_structure = TCM_model_parameters()['neuron_per_structure']
synaptic_fidelity_per_structure = TCM_model_parameters()['synaptic_fidelity_per_structure']
neurons_connected_with_hyperdirect_neurons = TCM_model_parameters()['neurons_connected_with_hyperdirect_neurons']
bias_current = TCM_model_parameters()['currents_per_structure']
neuron_quantities = TCM_model_parameters()['neuron_quantities']
noise = TCM_model_parameters()['noise']

n_s = neuron_quantities['qnt_neurons_s']
n_m = neuron_quantities['qnt_neurons_m']
n_d = neuron_quantities['qnt_neurons_d']
n_ci = neuron_quantities['qnt_neurons_ci']
n_tc = neuron_quantities['qnt_neurons_tc']
n_tr = neuron_quantities['qnt_neurons_tr']

T = TCM_model['simulation_time_ms']                 # simulation time in ms
dt = TCM_model['dt']                                # step
td_syn = TCM_model['transmission_delay_synapse']    # Synaptic transmission delay 
n_sim = TCM_model['simulation_steps']               # Number of simulation steps
synaptic_fidelity = TCM_model['synaptic_fidelity']
Fs = TCM_model['sampling_frequency']
chop_till = TCM_model['chop_till']

# =============================================================================
# COUPLING MATRIXES
# =============================================================================

# Coupling matrix - Normal condition
facilitating_factor_N = TCM_model['connectivity_factor_normal_condition']

Z_N = coupling_matrix_normal(facilitating_factor_N, n_s, n_m, n_d, n_ci, n_tc, n_tr)['matrix']
# normalizing
# Z_N_norm = Z_N/np.linalg.norm(Z_N)

# Coupling matrix - Parkinsonian Desease condition
facilitating_factor_PD = TCM_model['connectivity_factor_PD_condition']

Z_PD = coupling_matrix_PD(facilitating_factor_PD, n_s, n_m, n_d, n_ci, n_tc, n_tr)['matrix']
# normalizing 
# Z_PD_norm = Z_PD/np.linalg.norm(Z_PD)

# =============================================================================
# Graphs
# =============================================================================
# fig, (ax1, ax2, cax) = plt.subplots(ncols=3,figsize=(10,5), 
#                   gridspec_kw={"width_ratios":[1,1, 0.05]})
# 
# fig.subplots_adjust(wspace=0.3)
# 
# im1 = ax1.imshow(Z_N_norm, 
#                  extent=[-1,1,-1,1], 
#                  vmin = -1, vmax = 1,
#                  cmap=plt.cm.seismic
#                  )
# im2 = ax2.imshow(Z_PD_norm, 
#                  extent=[-1,1,-1,1], 
#                  vmin = -1, vmax = 1,
#                  cmap=plt.cm.seismic
#                  )
# 
# # Major ticks every 1, minor ticks every 1
# major_ticks = np.arange(-1, 1.05, 2/6)
# minor_ticks = np.arange(-1, 1.05, 2/6)
# 
# ax1.set_xticks(major_ticks)
# ax1.set_xticks(minor_ticks, minor=True)
# ax1.set_yticks(major_ticks)
# ax1.set_yticks(minor_ticks, minor=True)
# ax1.grid(True)
# ax2.set_xticks(major_ticks)
# ax2.set_xticks(minor_ticks, minor=True)
# ax2.set_yticks(major_ticks)
# ax2.set_yticks(minor_ticks, minor=True)
# ax2.grid(True)
# 
# ax1.set_title('Normal condition')
# ax2.set_title('PD condition')
# 
# fig.colorbar(im1, cax=cax)
# plt.show()
# =============================================================================

# =============================================================================
# NOISE TERMS
# =============================================================================
kisi_S_E = noise['kisi_S_E']
kisi_M_E = noise['kisi_M_E']
kisi_D_E = noise['kisi_D_E']
kisi_CI_I = noise['kisi_CI_I']
kisi_Rel_E = noise['kisi_Rel_E']
kisi_Ret_I = noise['kisi_Ret_I']

zeta_S_E = noise['zeta_S_E']
zeta_M_E = noise['zeta_M_E']
zeta_D_E = noise['zeta_D_E']
zeta_CI_I = noise['zeta_CI_I']
zeta_Rel_E = noise['zeta_Rel_E']
zeta_Ret_I = noise['zeta_Ret_I']

pn_S_E = noise['pn_S_E']
pn_M_E = noise['pn_M_E']
pn_D_E = noise['pn_D_E']
pn_CI_I = noise['pn_CI_I']
pn_Rel_E = noise['pn_Rel_E']
pn_Ret_I = noise['pn_Ret_I']

# =============================================================================
# BIAS CURRENTS
# =============================================================================
I_S = bias_current['I_S']
I_M = bias_current['I_M']
I_D = bias_current['I_D']
I_CI = bias_current['I_CI']
I_Ret = bias_current['I_Ret']
I_Rel = bias_current['I_Rel']

# =============================================================================
# POISSONIAN background activity (assuming they invade primary motor cortex from premotor and supplimentary motor areas
# =============================================================================
# Poissonian postsynaptic input to the E and I neurons for all layers

fr = 20 + 2*random_factor   # Poissonian firing frequency from other parts of the brain

[spikess, t_sp] = poissonSpikeGen(fr, T/1000, 1, dt/1000)
tps = np.argwhere(spikess==1)[:,1]

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
# DBS
# =============================================================================
I_dbs = DBS(n_sim, synaptic_fidelity, Fs, chop_till)['I_dbs']
I_dbs_pre = DBS(n_sim, synaptic_fidelity, Fs, chop_till)['I_dbs_pre']




