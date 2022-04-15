# -*- coding: utf-8 -*-
"""
@author: Celine Soeiro

@description: Tsodyks and Markram Synapse Model


For an individual synapse the model reproduces the postsynaptic responses enerated 
by any presynaptice spike train t_sp for interpyramidal synapses in layer V
    
* PHENOMENOLOGICAL MODEL:
        
    dxdt = z/t_rec - U_se*x*(t_sp - 0)*delta*(t - t_sp)
    dy/dt = -y/t_in + U_se*x*(t_sp - 0)*delta*(t - t_sp)
    dz/dt = y/t_in - z/t_rec
    
Where:    
    t_sp -> time in which the presynaptic spike arrives
    U_se -> Utilization of the synaptic efficacy
    t_in -> time constant in which the presynaptic spike inactivates
    t_rec -> time constant in which the presynaptic spike recovers
    x -> fraction of resources in the recovered state
    y -> fraction of resources in the active state
    z -> fraction of resources in the inactive state

The postsynaptice current is taken to be proportional to the fraction of resources in the active state
    I_s(t) = A_se*y(t)

Where:
    A_se -> The absolute synaptice strenth

The two major parameters of the model are A_se and U_se
    A_Se -> Can be exhibited only by activating all of the resources
    U_se -> Determines the dynamics of the synaptic response
    

* POPULATION SIGNAL:
    
    dx/dt = (1-x)/t_rec - U_se_1*x*r(t)
    dy/dt = -y/t_in + U_se_1*x*r(t)
    dU_se_avg/dt = -U_se_avg/t_facil + U_se*(1-U_se_avg)*r(t)
    U_se_1 = U_se_avg*(1 - U_se) + U_se

Where:
    x -> fraction of resources in the recovered state
    r(t) -> rate of a Poisson train for the neuron at time t
    U_se_avg -> the average value of U_se immediately before the spike
    
The postsynaptice current is taken to be proportional to the fraction of resources in the active state
    I_s(t) = A_se*y(t)

Where:
    A_se -> The absolute synaptice strenth
    
Depressing synapses are described by the first of these equations with the fixed value of U_se_1.


"""

import matplotlib.pyplot as plt

from neuron_population import neuron_population
from neuron_params import regular_spiking

neuron_params = regular_spiking()

RS_population_voltage, RS_population_current = neuron_population(100, neuron_params)

first_neuron = RS_population_voltage[0]
plt.plot(first_neuron)
plt.show()

