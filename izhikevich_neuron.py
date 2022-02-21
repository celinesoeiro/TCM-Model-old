"""

@author: Celine Soeiro

@description: Izhikevich Model

The Izhikevich model can be represented through an 2-D system of differential 
equations:
    
    dvdt = 0.04*v*v + 5*v + 140 - u + I
    dudt = a*(b*v - u)
    
    with conditions: 
        if v >= 30mV, then v = c and u = u + d

"""

# =============================================================================
# CONSTANTS DEFINITIONS
# 
# u: Equation variable - Represents membrane recovery variable
# v: Equation variable - Represents membrane potential of the neuron
# a: Equation parameter - Time scale of the recovery variable u
# b: Equation parameter - Sensitivity of u to the fluctuations in v
# c: Equation parameter - After-spike reset value of v
# d: Equation parameter - After-spike reset value of u
#
# excitatory_neurons: Number of excitatory neurons
# inhibitory_neurons: Number of inhibitory neurons
# excitatory_vector: Column vector of excitatory neurons
# inhibitory_vector: Column vector of inhibitory neurons
# 
# =============================================================================

import numpy as np
import pylab as plt

time_steps = 1000
excitatory_neurons = 800
inhibitory_neurons = 200
neurons = excitatory_neurons + inhibitory_neurons
excitatory_vector = np.random.rand(excitatory_neurons, 1)
inhibitory_vector = np.random.rand(inhibitory_neurons, 1)

a = np.concatenate([
    0.02*np.ones((excitatory_neurons, 1)),  # excitatory contribution
    0.02 + 0.08*inhibitory_vector           # inhibitory contribution
    ])
b = np.concatenate([
    0.2*np.ones((excitatory_neurons, 1)),   # excitatory contribution
    0.25 - 0.05*inhibitory_vector           # inhibitory contribution
    ])
c = np.concatenate([
    -65 + 15*excitatory_vector**2,          # excitatory contribution
    -65*np.ones((inhibitory_neurons, 1))    # inhibitory contribution
    ])
d = np.concatenate([
    8 - 6*excitatory_vector**2,             # excitatory contribution
    2*np.ones((inhibitory_neurons, 1))      # inhibitory contribution
    ])
S = np.concatenate([
    0.5*np.random.rand(neurons, excitatory_neurons),
    -np.random.rand(neurons,inhibitory_neurons)
    ], axis = 1)


v = -65*np.ones((neurons, 1))
u = b*-65
neurons_that_fired_across_time = []
voltage_across_time = []

for t in range(1, time_steps + 1):
    # The input voltage
    I = np.concatenate([
        5*np.random.rand(excitatory_neurons, 1),
        2*np.random.rand(inhibitory_neurons, 1)
        ])

    neurons_that_fired = np.where(v > 30)
    voltage_across_time.append(float(v[10]))
    neurons_that_fired_across_time.append([
        t + 0 * neurons_that_fired[0], 
        neurons_that_fired[0]
        ])
    
    for i in neurons_that_fired[0]:
        v[i] = c[i]
        u[i] += d[i]
    
    I += np.expand_dims(np.sum(S[:, neurons_that_fired[0]], axis = 1), axis = 1)
    
    # Incrementing 0.5ms for numerical stability
    v += 0.5*(0.04*v**2 + 5*v + 140 - u + I)
    v += 0.5*(0.04*v**2 + 5*v + 140 - u + I)
    u = u + a*(b*v - u)

voltage_across_time = np.array(voltage_across_time)

# Plots
fig1, axs1 = plt.subplots(2, sharex=True)
fig1.suptitle("Izhikevich model")
axs1[0].plot(voltage_across_time, 'b', label='voltage (mV)')
axs1[1].plot(I, 'k', label='input voltage (mV)')
axs1[0].legend(shadow=True, fancybox=True)
axs1[0].grid()