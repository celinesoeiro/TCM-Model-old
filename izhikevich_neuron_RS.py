# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 22:36:00 2022

@author: celin
"""

import numpy as np
import matplotlib.pyplot as plt

simulation_time = 100   # ms
step = 0.1              # ms
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
    0.2 - 0.05*inhibitory_vector           # inhibitory contribution
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
time = np.arange(0, simulation_time + step, step) 

for t in time:
    # The input voltage
    I = np.concatenate([
        10*np.random.rand(excitatory_neurons, 1),
        10*np.random.rand(inhibitory_neurons, 1)
        ])

    neurons_that_fired = np.where(v > 30)

    voltage_across_time.append(float(v[10]))
    neurons_that_fired_across_time.append([
        t, 
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

plt.figure(1)
plt.suptitle("Izhikevich model - Regular Spiking")
plt.plot(voltage_across_time, 'b', label='voltage (mV)')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.show()

plt.figure(2)
plt.suptitle("Izhikevich model - Regular Spiking")
plt.plot(I, 'k', label='input voltage (mV)')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.show()

