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
from random import seed, random

def izhikevich_neuron(
        params: dict,
        neuron_type: str,
        neurons: int,
        voltage_pick: float,
        simulation_time: int,
        time_step: float,
        current_value: int,
        current_start: int,
        current_finish: int,
        initial_voltage = -65,
        ):
    
    # Time grid
    time = np.arange(0, simulation_time + time_step, time_step)
    
    # Check if paramaters exists, if dont display error msg
    if (not params.get('a') 
        or not params.get('b') 
        or not params.get('c') 
        or not params.get('d')
        ): 
        return 'Parameters must be a, b, c and d' 
    
    # Parameters according Izhikevich article 
    seed(1)
    random_factor = random()
    
    if (neuron_type == 'excitatory' or 'excit'):
        a = params['a']
        b = params['b']
        c = params['c'] + 15*random_factor**2
        d = params['d'] - 6*random_factor**2
    elif (neuron_type == 'inhibitory' or 'inib'):
        a = params['a'] + 0.08*random_factor
        b = params['b'] - 0.05*random_factor
        c = params['c']
        d = params['d']
    else:
        return 'Neuron type must be excitatory or inhibitory'
    
    I = np.zeros(len(time))
    I[current_start:current_finish] = current_value
    
    v = np.zeros(len(time))    
    v[0] = initial_voltage
    
    u = np.zeros(len(time))    
    u[0] = b*v[0]
    
    # Izhikevich neuron equations
    def dvdt(v, u, I):
        return 0.04*v**2 + 5*v + 140 - u + I
    
    def dudt(v,u):
        return a*(b*v - u)
    
    fired = []
    
    for t in range(1, len(time)):     
        vc = v[t - 1]
        uc = u[t - 1]
        Ic = I[t - 1]
        
        if (vc >= voltage_pick):
            vc = v[t]
            v[t] = c
            u[t] = uc + d
            fired.append(t)
        
        else:            
            # solve using euler
            dv = dvdt(vc, uc, Ic)
            du = dudt(vc, uc)
            v[t] = vc + dv*time_step
            u[t] = uc + du*time_step

    return v, I

