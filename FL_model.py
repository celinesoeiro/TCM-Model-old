"""
@author: Celine Soeiro

@description: Replicate the FL model

Base code: https://senselab.med.yale.edu/ModelDB/showmodel.cshtml?model=266941&file=%2fTCM_ModelDB%2fThalamoCortical_Microcircuit_PD_DBS_ModelDB.m#tabs-2

Abbreviations:
    N -> normal condition
    PD -> Parkinsonian desease condition
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

sns.set()

from utils import poissonSpikeGen
from tsodyks_markram_synapse import TM_Synapse
from izhikevich_neuron import izhikevich_neuron
from dbs import DBS

import TR_cells as tr_nucleos
 

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
random_factor = TCM_model_parameters()['random_factor']

n_s = neuron_quantities['S']
n_m = neuron_quantities['M']
n_d = neuron_quantities['D']
n_ci = neuron_quantities['CI']
n_tc = neuron_quantities['TC']
n_tr = neuron_quantities['TR']

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
Z_N_norm = Z_N/np.linalg.norm(Z_N)

# Coupling matrix - Parkinsonian Desease condition
facilitating_factor_PD = TCM_model['connectivity_factor_PD_condition']

Z_PD = coupling_matrix_PD(facilitating_factor_PD, n_s, n_m, n_d, n_ci, n_tc, n_tr)['matrix']
# normalizing 
Z_PD_norm = Z_PD/np.linalg.norm(Z_PD)

# =============================================================================
# Graphs
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(17,7))

fig.subplots_adjust(wspace=0.3)
fig.suptitle('Matriz de conexão')

CM_Normal = pd.DataFrame(Z_N_norm, columns=['S', 'M', 'P', 'IC', 'RTC', 'RT'])
CM_PD = pd.DataFrame(Z_PD_norm, columns=['S', 'M', 'P', 'IC', 'RTC', 'RT'])

sns.heatmap(CM_Normal, 
            vmin=-1, vmax=1, 
            yticklabels=['S', 'M', 'P', 'IC', 'RTC', 'RT'], 
            annot=True, 
            fmt=".3f", 
            linewidth=.75,
            cmap=sns.color_palette("coolwarm", as_cmap=True),
            ax=ax1,
            )
ax1.set(xlabel="", ylabel="")
ax1.xaxis.tick_top()
ax1.set_title('Condição normal')

sns.heatmap(CM_PD, 
            vmin=-1, vmax=1, 
            yticklabels=['S', 'M', 'P', 'IC', 'RTC', 'RT'], 
            annot=True, 
            fmt=".3f", 
            linewidth=.75,
            cmap=sns.color_palette("coolwarm", as_cmap=True),
            ax=ax2,
            )
ax2.set(xlabel="", ylabel="")
ax2.xaxis.tick_top()
ax2.set_title('Condição parkinsoniana')

plt.show()


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
[I_dbs, I_dbs_pre, dev] = DBS(n_sim, synaptic_fidelity, Fs, chop_till)

# Postsynaptic DBS pulses (intra-axonal)

[t_dbs_post, I_dbs_post] = TM_Synapse(
    t_event = I_dbs_pre, 
    n_sim = n_sim,
    t_delay = td_syn,
    dt = dt, 
    dbs = True, 
    synapse_type = 'excitatory'
    )

I_dbs[0,:] = I_dbs_pre  
I_dbs[1,:] = I_dbs_post

# Graphs
x0 = 105000
xf = 110050

fig1, (ax1, ax2) = plt.subplots(ncols=2,figsize=(10,5))
ax1.set_title('Pre Synaptic Stimulus')
ax1.plot(I_dbs[0])
ax1.grid()

zoomed_1 = I_dbs[0]
ax2.set_title('Pre Synaptic Stimulus - Zoom')
ax2.plot(I_dbs[0])
ax2.set_xlim(x0, xf)
ax1.grid()

fig2, (ax1, ax2) = plt.subplots(ncols=2,figsize=(10,5))
ax1.set_title('Post Synaptic Stimulus')
ax1.plot(I_dbs[1])
ax1.grid()

ax2.set_title('Post Synaptic Stimulus - Zoom')
ax2.plot(I_dbs[1])
ax2.set_xlim(x0, xf)
ax1.grid()

# =============================================================================
# SETTING INITIAL SIMULATION VALUES
# =============================================================================


# =============================================================================
# RUN THE SIMULATION
# =============================================================================

# Thalamic Reticular Nucleus (TRN) cells



# =============================================================================



