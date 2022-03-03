"""

@author: Celine Soeiro

@description: Izhikevich Model

The Izhikevich model can be represented through an 2-D system of differential 
equations:
    
    dvdt = 0.04*v*v + 5*v + 140 - u + I
    dudt = a*(b*v - u)
    
    with conditions: 
        if v >= 30mV, then v = c and u = u + d
    
    for excitatory neurons: 
        (ai,bi) = (a,b)
        (ci,di) = (c,d) + (15, -6)r**2
    for inhibitory neurons:
        (ai,bi) = (a,b) + (0.08, -0.05)*r
        (ci,di) = (c,d)

    u: Equation variable - Represents membrane recovery variable
    v: Equation variable - Represents membrane potential of the neuron
    a: Equation parameter - Time scale of the recovery variable u
    b: Equation parameter - Sensitivity of u to the fluctuations in v
    c: Equation parameter - After-spike reset value of v
    d: Equation parameter - After-spike reset value of u
    S: Synaptic matrix - Synaptic connection weights between the neurons
    
    excitatory_neurons: Number of excitatory neurons
    inhibitory_neurons: Number of inhibitory neurons
    excitatory_vector: Column vector of excitatory neurons
    inhibitory_vector: Column vector of inhibitory neurons
"""

import numpy as np
from utils import randn

def izhikevich_neuron(
        params: dict,
        input_current: np.array([5,2], dtype=float),
        simulation_time: int,
        time_step: float,
        excitatory_neurons: int, 
        inhibitory_neurons: int,
        voltage_pick: float,
        ):
    
    # Check if paramaters exists, if dont display error msg
    if (not params.get('a') 
        or not params.get('b') 
        or not params.get('c') 
        or not params.get('d')
        ): 
        return 'Parameters must be a, b, c and d' 
    
    # Parameters according Izhikevich article 
    neurons = excitatory_neurons + inhibitory_neurons
    excitatory_rand_factor = np.random.rand(excitatory_neurons, 1)
    inhibitory_rand_factor = np.random.rand(inhibitory_neurons, 1)
    
    a = np.concatenate([
        params['a']*np.ones((excitatory_neurons, 1)),   # excitatory contribution
        params['a'] + 0.08*inhibitory_rand_factor       # inhibitory contribution
        ])
    b = np.concatenate([
        params['b']*np.ones((excitatory_neurons, 1)),   # excitatory contribution
        params['b'] - 0.05*inhibitory_rand_factor       # inhibitory contribution
        ])
    c = np.concatenate([
        params['c'] + 15*excitatory_rand_factor**2,     # excitatory contribution
        params['c']*np.ones((inhibitory_neurons, 1))    # inhibitory contribution
        ])
    d = np.concatenate([
        params['d'] - 6*excitatory_rand_factor**2,      # excitatory contribution
        params['d']*np.ones((inhibitory_neurons, 1))    # inhibitory contribution
        ])
    
    # Synaptic Matrix
    S = np.concatenate([
        0.5*np.random.rand(neurons, excitatory_neurons),# excitatory contribution
        -np.random.rand(neurons,inhibitory_neurons)     # inhibitory contribution
        ], axis = 1)
    
    v = -65*np.ones((neurons, 1))
    u = b*v
    
    neurons_that_fired_across_time = []
    voltage_across_time = []
    
    time = np.arange(0, simulation_time + time_step, time_step) 
    
    for t in time:
        I = np.concatenate([
        input_current[0]*randn(excitatory_neurons, column=True),# excitatory contribution
        input_current[1]*randn(inhibitory_neurons, column=True) # inhibitory contribution
        ])
        
        # Array contaning neurons that fired
        fired = np.where(v >= voltage_pick)

        voltage_across_time.append(float(v[10]))
        neurons_that_fired_across_time.append([
            t, 
            fired[0]
            ])

        for i in fired[0]:
            v[i] = c[i]
            u[i] += d[i]

        I += np.expand_dims(np.sum(S[:, fired[0]], axis = 1), axis = 1)
        
        v += 0.5*(0.04*v**2 + 5*v + 140 - u + I)
        v += 0.5*(0.04*v**2 + 5*v + 140 - u + I)
        u = u + a*(b*v - u)
    
    voltage_across_time = np.array(voltage_across_time)
    
    return voltage_across_time

