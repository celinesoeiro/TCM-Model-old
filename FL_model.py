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

# =============================================================================
# IMPORT PARAMETERS
# =============================================================================
from model_parameters import TCM_model_parameters, coupling_matrix_normal, coupling_matrix_PD

TCM_model = TCM_model_parameters()['model_global_parameters']
neurons_per_structure = TCM_model_parameters()['neuron_per_structure']
synaptic_fidelity_per_structure = TCM_model_parameters()['synaptic_fidelity']
neurons_connected_with_hyperdirect_neurons = TCM_model_parameters()['neurons_connected_with_hyperdirect_neurons']
bias_current = TCM_model_parameters()['bias_current']
neuron_quantities = TCM_model_parameters()['neuron_quantities']

n_s = neuron_quantities['qnt_neurons_s']
n_m = neuron_quantities['qnt_neurons_m']
n_d = neuron_quantities['qnt_neurons_d']
n_ci = neuron_quantities['qnt_neurons_ci']
n_tc = neuron_quantities['qnt_neurons_tc']
n_tr = neuron_quantities['qnt_neurons_tr']

# =============================================================================
# COUPLING MATRIXES
# =============================================================================

# Coupling matrix - Normal condition
facilitating_factor_N = TCM_model['connectivity_factor_normal_condition']

Z_N = coupling_matrix_normal(facilitating_factor_N, n_s, n_m, n_d, n_ci, n_tc, n_tr)['matrix']
Z_N_norm = Z_N/np.linalg.norm(Z_N)

# Coupling matrix - Parkinsonian Desease condition
facilitating_factor_PD = TCM_model['connectivity_factor_PD_condition']

Z_PD = coupling_matrix_PD(facilitating_factor_PD, n_s, n_m, n_d, n_ci, n_tc, n_tr)['matrix']
Z_PD_norm = Z_PD/np.linalg.norm(Z_PD)

# Graphs
fig, (ax1, ax2, cax) = plt.subplots(ncols=3,figsize=(10,5), 
                  gridspec_kw={"width_ratios":[1,1, 0.05]})

fig.subplots_adjust(wspace=0.3)

im1 = ax1.imshow(Z_N_norm, 
                 extent=[-1,1,-1,1], 
                 vmin = -1, vmax = 1,
                 cmap=plt.cm.seismic
                 )
im2 = ax2.imshow(Z_PD_norm, 
                 extent=[-1,1,-1,1], 
                 vmin = -1, vmax = 1,
                 cmap=plt.cm.seismic
                 )

# Major ticks every 1, minor ticks every 1
major_ticks = np.arange(-1, 1.05, 2/6)
minor_ticks = np.arange(-1, 1.05, 2/6)

ax1.set_xticks(major_ticks)
ax1.set_xticks(minor_ticks, minor=True)
ax1.set_yticks(major_ticks)
ax1.set_yticks(minor_ticks, minor=True)
ax1.grid(True)
ax2.set_xticks(major_ticks)
ax2.set_xticks(minor_ticks, minor=True)
ax2.set_yticks(major_ticks)
ax2.set_yticks(minor_ticks, minor=True)
ax2.grid(True)

ax1.set_title('Normal condition')
ax2.set_title('PD condition')

fig.colorbar(im1, cax=cax)
plt.show()
