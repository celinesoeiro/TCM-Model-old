"""
@author: Celine Soeiro

@description: Tsodyks and Markram Synapse Model according FL

For an individual synapse the model reproduces the postsynaptic responses enerated 
by any presynaptice spike train t_sp for interpyramidal synapses in layer V
    
* PHENOMENOLOGICAL MODEL:
    
    dudt = -(u/t_f) + U*(1 - u)*delta(t - t_sp - delta)
    dxdt = -((1 - x)/t_d) - u*x*delta(t - t_sp - delta)    
    dIdt = -(I/t_sp) + A*u*x*delta(t - t_sp - delta)
    
Where:    
    t_sp -> time in which the presynaptic spike arrives
    delta -> Dirac delta function
    U -> increment of u produced by an incoming spike
    t_f -> decay time constant of variable u
    t_d -> recovery time constant of variable x
    t_s -> decay time constant of variable I
    A -> synaptic response amplitude that would be produced with the release of all neurotransmitters (absolute synaptic response)

The two major parameters of the model are A and U
    A -> Can be exhibited only by activating all of the resources
    U -> Determines the dynamics of the synaptic response
    

"""

import numpy as np
import matplotlib.pyplot as plt
from random import seed, random

from Izhikevih_neuron.neuron_population import neuron_population
from Izhikevih_neuron.neuron_params import regular_spiking


# =============================================================================
# Parameters
# =============================================================================
seed(1)
random_factor = random()

def getParamaters(synapse_type: str):
    if (synapse_type == 'excitatory'):
        return {
            # [Facilitating, Depressing, Pseudo-linear]
            't_f': [670, 17, 326],
            't_d': [138, 671, 329],
            'U': [0.09, 0.5, 0.29],
            'distribution': [0.2, 0.63, 0.17],
        };
    elif (synapse_type == 'inhibitory'):
        return {
            # [Facilitating, Depressing, Pseudo-linear]
            't_f': [376, 21, 62],
            't_d': [45, 706, 144],
            'U': [0.016, 0.25, 0.32],
            'distribution': [0.08, 0.75, 0.17],
        };
    else:
        return 'Invalid synapse_type. Synapse_type must be excitatory or inhibitory.'
    
# =============================================================================
# Equations
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
    
def TM_Synapse(t_event, n_sim, t_delay, dt, synapse_type):
    t = np.arange(t_delay + 1, n_sim - 1)
    n_sim = int(n_sim)
    
    # parameters
    t_f = getParamaters(synapse_type)['t_f']
    t_d = getParamaters(synapse_type)['t_d']
    U = getParamaters(synapse_type)['U']
    A = getParamaters(synapse_type)['distribution']
    parameters_length = len(t_f)
    
    # IDK what this variable means
    t_s = 11

    # Initial values
    u = np.zeros((3, n_sim))
    x = np.ones((3, n_sim))
    I = np.zeros((3, n_sim))
    
    # poissonian contribution to the spike
    spd = np.zeros((1,n_sim))

    # Loop trhough the parameters
    for p in range(parameters_length):
        # Update the variable
        spd[0][t_event] = 1/dt
        # Solve ODE using Euler method
        for i in range(t_delay + 1, n_sim - 1):
            delta = spd[0][i - t_delay] # marks when the spike occurs
            
            u[p][i + 1] = u[p][i] + dt*u_eq(u[p][i], t_f[p], U[p], delta)
            x[p][i + 1] = x[p][i] + dt*x_eq(x[p][i], t_d[p], u[p][i], delta)
            I[p][i + 1] = I[p][i] + dt*I_eq(I[p][i], t_s, A[p], u[p][i], x[p][i], delta)

    I_post_synaptic = np.concatenate(I, axis=None)
    
    return t, t_f, t_d, U, A, I_post_synaptic


# =============================================================================
# Utils
# =============================================================================

def poissonSpikeGen(fr, tSim, nTrials, dt):    
    nBins = int(np.floor(tSim/dt))
    spikeMat = np.random.rand(nTrials, nBins) < fr*dt
    tVec = np.arange(0,tSim - dt, dt)
    
    return spikeMat, tVec

# =============================================================================
# Usage
# =============================================================================

sim_time = 10               # Simulation time in seconds (must be a multiplacative of 3 under PD+DBS condition)
T = (sim_time + 1)*1000     # Simulation time in ms with 1 extra second to reach the steady state and trash later
dt = .1                     # Time span and step (of ms)
td_syn = 1                  # Synaptic transmission delay (fixed for all synapses in the TCM)
n_sim = np.round(T/dt)      # Number of simulation steps
synapse_type = 'inhibitory' # Synapse type
fr = 20 + 2*random_factor   # Poissonian firing frequency from other parts of the brain

[spikess, tsp] = poissonSpikeGen(fr, T/1000, 1, dt/1000)
tps = np.argwhere(spikess==1)[:,1]

TM_Synapse(tps, n_sim, td_syn, dt, synapse_type)


rs_neuron_params = regular_spiking()

RS_population_voltage, RS_population_current = neuron_population(1, rs_neuron_params)

first_neuron = RS_population_voltage[0]
plt.plot(first_neuron)
plt.show()




