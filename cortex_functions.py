"""
Created on Tue Sep 19 14:26:30 2023

@author: celinesoeiro
"""

import numpy as np

def poissonSpikeGen(time, fr, dt, n_neurons):
    # Calculate the number of spikes expected per neuron in the given time step
    mean_spikes = fr * dt
    spike_times = []
    
    for _ in range(n_neurons):
        num_steps = int(time/dt)    
        spike_train = np.random.poisson(mean_spikes, num_steps)*dt
        spike_times.append(spike_train)
        
    return spike_times


def izhikevich_dvdt(v, u, I):
    return 0.04*v**2 + 5*v + 140 - u + I

def izhikevich_dudt(v, u, a, b):
    return a*(b*v - u)