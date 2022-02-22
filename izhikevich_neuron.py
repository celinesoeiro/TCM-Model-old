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
import matplotlib.pyplot as plt

def izhikevich_neurons(
        neuron_type, 
        time_steps, 
        excitatory_neurons, 
        inhibitory_neurons):
    
    neurons = excitatory_neurons + inhibitory_neurons
    excitatory_vector = np.random.rand(excitatory_neurons, 1)
    inhibitory_vector = np.random.rand(inhibitory_neurons, 1)
    
    if (neuron_type == 'RS'): 
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
    elif (neuron_type == 'IB'):
        a = np.concatenate([
            0.02*np.ones((excitatory_neurons, 1)),  # excitatory contribution
            0.02 + 0.08*inhibitory_vector           # inhibitory contribution
            ])
        b = np.concatenate([
            0.2*np.ones((excitatory_neurons, 1)),   # excitatory contribution
            0.2 - 0.05*inhibitory_vector            # inhibitory contribution
            ])
        c = np.concatenate([
            -55 + 15*excitatory_vector**2,          # excitatory contribution
            -55*np.ones((inhibitory_neurons, 1))    # inhibitory contribution
            ])
        d = np.concatenate([
            4 - 6*excitatory_vector**2,             # excitatory contribution
            4*np.ones((inhibitory_neurons, 1))      # inhibitory contribution
            ])
        S = np.concatenate([
            0.5*np.random.rand(neurons, excitatory_neurons),
            -np.random.rand(neurons,inhibitory_neurons)
            ], axis = 1)
    elif (neuron_type == 'CH'):
        a = np.concatenate([
            0.02*np.ones((excitatory_neurons, 1)),  # excitatory contribution
            0.02 + 0.08*inhibitory_vector           # inhibitory contribution
            ])
        b = np.concatenate([
            0.2*np.ones((excitatory_neurons, 1)),   # excitatory contribution
            0.2 - 0.05*inhibitory_vector            # inhibitory contribution
            ])
        c = np.concatenate([
            -50 + 15*excitatory_vector**2,          # excitatory contribution
            -50*np.ones((inhibitory_neurons, 1))    # inhibitory contribution
            ])
        d = np.concatenate([
            2 - 6*excitatory_vector**2,             # excitatory contribution
            2*np.ones((inhibitory_neurons, 1))      # inhibitory contribution
            ])
        S = np.concatenate([
            0.5*np.random.rand(neurons, excitatory_neurons),
            -np.random.rand(neurons,inhibitory_neurons)
            ], axis = 1)
    elif (neuron_type == 'FS'):
        a = np.concatenate([
            0.1*np.ones((excitatory_neurons, 1)),  # excitatory contribution
            0.1 + 0.08*inhibitory_vector           # inhibitory contribution
            ])
        b = np.concatenate([
            0.2*np.ones((excitatory_neurons, 1)),  # excitatory contribution
            0.2 - 0.05*inhibitory_vector           # inhibitory contribution
            ])
        c = np.concatenate([
            -65 + 15*excitatory_vector**2,         # excitatory contribution
            -65*np.ones((inhibitory_neurons, 1))   # inhibitory contribution
            ])
        d = np.concatenate([
            2 - 6*excitatory_vector**2,            # excitatory contribution
            2*np.ones((inhibitory_neurons, 1))     # inhibitory contribution
            ])
        S = np.concatenate([
            0.5*np.random.rand(neurons, excitatory_neurons),
            -np.random.rand(neurons,inhibitory_neurons)
            ], axis = 1)
    elif (neuron_type == 'TC'):
        a = np.concatenate([
            0.02*np.ones((excitatory_neurons, 1)),  # excitatory contribution
            0.02 + 0.08*inhibitory_vector           # inhibitory contribution
            ])
        b = np.concatenate([
            0.25*np.ones((excitatory_neurons, 1)),   # excitatory contribution
            0.25 - 0.05*inhibitory_vector           # inhibitory contribution
            ])
        c = np.concatenate([
            -65 + 15*excitatory_vector**2,          # excitatory contribution
            -65*np.ones((inhibitory_neurons, 1))    # inhibitory contribution
            ])
        d = np.concatenate([
            0.05 - 6*excitatory_vector**2,             # excitatory contribution
            0.05*np.ones((inhibitory_neurons, 1))      # inhibitory contribution
            ])
        S = np.concatenate([
            0.5*np.random.rand(neurons, excitatory_neurons),
            -np.random.rand(neurons,inhibitory_neurons)
            ], axis = 1)
    elif (neuron_type == 'RZ'):
        a = np.concatenate([
            0.1*np.ones((excitatory_neurons, 1)),  # excitatory contribution
            0.1 + 0.08*inhibitory_vector           # inhibitory contribution
            ])
        b = np.concatenate([
            0.25*np.ones((excitatory_neurons, 1)),   # excitatory contribution
            0.25 - 0.05*inhibitory_vector           # inhibitory contribution
            ])
        c = np.concatenate([
            -65 + 15*excitatory_vector**2,          # excitatory contribution
            -65*np.ones((inhibitory_neurons, 1))    # inhibitory contribution
            ])
        d = np.concatenate([
            2 - 6*excitatory_vector**2,             # excitatory contribution
            2*np.ones((inhibitory_neurons, 1))      # inhibitory contribution
            ])
        S = np.concatenate([
            0.5*np.random.rand(neurons, excitatory_neurons),
            -np.random.rand(neurons,inhibitory_neurons)
            ], axis = 1)
    elif (neuron_type == 'LTS'):
        a = np.concatenate([
            0.02*np.ones((excitatory_neurons, 1)),  # excitatory contribution
            0.02 + 0.08*inhibitory_vector           # inhibitory contribution
            ])
        b = np.concatenate([
            0.25*np.ones((excitatory_neurons, 1)),   # excitatory contribution
            0.25 - 0.05*inhibitory_vector           # inhibitory contribution
            ])
        c = np.concatenate([
            -65 + 15*excitatory_vector**2,          # excitatory contribution
            -65*np.ones((inhibitory_neurons, 1))    # inhibitory contribution
            ])
        d = np.concatenate([
            2 - 6*excitatory_vector**2,             # excitatory contribution
            2*np.ones((inhibitory_neurons, 1))      # inhibitory contribution
            ])
        S = np.concatenate([
            0.5*np.random.rand(neurons, excitatory_neurons),
            -np.random.rand(neurons,inhibitory_neurons)
            ], axis = 1)
    else:
        return 'Neuron type is incorrect. Try RS, IB, CH, FS, TC, RZ or LTS.'

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
    
    return voltage_across_time

RS_neuron = izhikevich_neurons('RS',1000,800,200)
IB_neuron = izhikevich_neurons('IB',1000,800,200)
CH_neuron = izhikevich_neurons('CH',1000,800,200)
FS_neuron = izhikevich_neurons('FS',1000,800,200)
TC_neuron = izhikevich_neurons('TC',1000,800,200)
RZ_neuron = izhikevich_neurons('RZ',1000,800,200)
LTS_neuron = izhikevich_neurons('LTS',1000,800,200)

# Plots
plt.figure(1)
plt.suptitle("Izhikevich model - Regular Spiking")
plt.plot(RS_neuron, 'b', label='voltage (mV)')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.show()

plt.figure(2)
plt.suptitle("Izhikevich model - Intrinsically Bursting")
plt.plot(IB_neuron, 'b', label='voltage (mV)')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.show()

plt.figure(3)
plt.suptitle("Izhikevich model - Chattering")
plt.plot(CH_neuron, 'b', label='voltage (mV)')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.show()

plt.figure(4)
plt.suptitle("Izhikevich model - Fast Spiking")
plt.plot(FS_neuron, 'b', label='voltage (mV)')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.show()

plt.figure(5)
plt.suptitle("Izhikevich model - Thalamo-Cortical")
plt.plot(TC_neuron, 'b', label='voltage (mV)')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.show()

plt.figure(6)
plt.suptitle("Izhikevich model - Resonator")
plt.plot(RZ_neuron, 'b', label='voltage (mV)')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.show()

plt.figure(7)
plt.suptitle("Izhikevich model - Low-threshold spiking")
plt.plot(LTS_neuron, 'b', label='voltage (mV)')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.show()
