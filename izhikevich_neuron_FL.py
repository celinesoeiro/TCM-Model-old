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
        a,
        b,
        c,
        d,
        simulation_time,
        time_step,
        excitatory_neurons, 
        inhibitory_neurons,
        input_voltage,
        voltage_pick,
        excitatory = True,
        inhibitory = True,
        ):
    
    neurons = excitatory_neurons + inhibitory_neurons
    excitatory_vector = np.random.rand(excitatory_neurons, 1)
    inhibitory_vector = np.random.rand(inhibitory_neurons, 1)
    
    if (excitatory and inhibitory):
        a_arr = np.concatenate([
            a*np.ones((excitatory_neurons, 1)),  # excitatory contribution
            a + 0.08*inhibitory_vector           # inhibitory contribution
            ])
        b_arr = np.concatenate([
            b*np.ones((excitatory_neurons, 1)),   # excitatory contribution
            b - 0.05*inhibitory_vector           # inhibitory contribution
            ])
        c_arr = np.concatenate([
            c + 15*excitatory_vector**2,          # excitatory contribution
            c*np.ones((inhibitory_neurons, 1))    # inhibitory contribution
            ])
        d_arr = np.concatenate([
            d - 6*excitatory_vector**2,             # excitatory contribution
            d*np.ones((inhibitory_neurons, 1))      # inhibitory contribution
            ])
        S = np.concatenate([
            0.5*np.random.rand(neurons, excitatory_neurons),
            -np.random.rand(neurons,excitatory_neurons)
            ], axis = 1)
    elif (excitatory): 
        a_arr = np.concatenate([
            a*np.ones((excitatory_neurons, 1)),  # excitatory contribution
            a*np.ones((inhibitory_neurons, 1))           # inhibitory contribution
            ])
        b_arr = np.concatenate([
            b*np.ones((excitatory_neurons, 1)),   # excitatory contribution
            b*np.ones((inhibitory_neurons, 1))           # inhibitory contribution
            ])
        c_arr = np.concatenate([
            c + 15*excitatory_vector**2,          # excitatory contribution
            c + 15*inhibitory_vector**2    # inhibitory contribution
            ])
        d_arr = np.concatenate([
            d - 6*excitatory_vector**2,             # excitatory contribution
            d - 6*inhibitory_vector**2,      # inhibitory contribution
            ])
        S = np.concatenate([
            0.5*np.random.rand(neurons, excitatory_neurons),
            -np.random.rand(neurons,inhibitory_neurons)
            ], axis = 1)
    elif (inhibitory):
        a_arr = np.concatenate([
            a + 0.08*excitatory_vector,  # excitatory contribution
            a + 0.08*inhibitory_vector           # inhibitory contribution
            ])
        b_arr = np.concatenate([
            b - 0.05*excitatory_vector,   # excitatory contribution
            b - 0.05*inhibitory_vector           # inhibitory contribution
            ])
        c_arr = np.concatenate([
            c*np.ones((excitatory_neurons, 1)),          # excitatory contribution
            c*np.ones((inhibitory_neurons, 1))    # inhibitory contribution
            ])
        d_arr = np.concatenate([
            d*np.ones((excitatory_neurons, 1)),             # excitatory contribution
            d*np.ones((inhibitory_neurons, 1))      # inhibitory contribution
            ])
        S = np.concatenate([
            0.5*np.random.rand(neurons, excitatory_neurons),
            -np.random.rand(neurons,excitatory_neurons)
            ], axis = 1)

    v = -65*np.ones((neurons, 1))
    u = b*-65
    neurons_that_fired_across_time = []
    voltage_across_time = []
    
    time = np.arange(0, simulation_time + time_step, time_step) 
    
    for t in time:
        # The input voltage
        I = input_voltage*np.concatenate([
            np.random.rand(excitatory_neurons, 1),
            np.random.rand(inhibitory_neurons, 1)
            ])
    
        neurons_that_fired = np.where(v > voltage_pick)

        voltage_across_time.append(float(v[10]))
        neurons_that_fired_across_time.append([
            t, 
            neurons_that_fired[0]
            ])

        for i in neurons_that_fired[0]:
            v[i] = c_arr[i]
            u[i] += d_arr[i]
        
        I += np.expand_dims(np.sum(S[:, neurons_that_fired[0]], axis = 1), axis = 1)
        
        # Incrementing 0.5ms for numerical stability
        v += 0.5*(0.04*v**2 + 5*v + 140 - u + I)
        v += 0.5*(0.04*v**2 + 5*v + 140 - u + I)
        u = u + a_arr*(b_arr*v - u)
    
    voltage_across_time = np.array(voltage_across_time)
    
    return voltage_across_time

RS_neuron = izhikevich_neurons(
    a = 0.02,
    b = 0.2,
    c = -65,
    d = 8,
    simulation_time = 100,
    time_step = 0.1,
    excitatory_neurons = 800,
    inhibitory_neurons = 200,
    input_voltage = 5,
    voltage_pick = 30,
    excitatory=True,
    inhibitory=False
    )
IB_neuron = izhikevich_neurons(
    a = 0.02,
    b = 0.2,
    c = -55,
    d = 4,
    simulation_time = 100,
    time_step = 0.1,
    excitatory_neurons = 800,
    inhibitory_neurons = 200,
    input_voltage = 5,
    voltage_pick = 30,
    excitatory=True,
    inhibitory=False
    )
FS_neuron = izhikevich_neurons(
    a = 0.1,
    b = 0.2,
    c = -65,
    d = 2,
    simulation_time = 100,
    time_step = 0.1,
    excitatory_neurons = 800,
    inhibitory_neurons = 200,
    input_voltage = 7,
    voltage_pick = 30,
    excitatory=False,
    inhibitory=True
    )
LTS_neuron = izhikevich_neurons(
    a = 0.02,
    b = 0.25,
    c = -65,
    d = 2,
    simulation_time = 100,
    time_step = 0.1,
    excitatory_neurons = 800,
    inhibitory_neurons = 200,
    input_voltage = 7,
    voltage_pick = 30,
    excitatory=False,
    inhibitory=True
    )
TC_neuron = izhikevich_neurons(
    a = 0.02,
    b = 0.25,
    c = -65,
    d = 0.05,
    simulation_time = 100,
    time_step = 0.1,
    excitatory_neurons = 800,
    inhibitory_neurons = 200,
    input_voltage = 1.1,
    voltage_pick = 30,
    excitatory=True,
    inhibitory=False
    )
TR_neuron = izhikevich_neurons(
    a = 0.02,
    b = 0.25,
    c = -65,
    d = 2.05,
    simulation_time = 100,
    time_step = 0.1,
    excitatory_neurons = 800,
    inhibitory_neurons = 200,
    input_voltage = 1.5,
    voltage_pick = 30,
    excitatory=False,
    inhibitory=True
    )

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
plt.suptitle("Izhikevich model - Fast Spiking")
plt.plot(FS_neuron, 'b', label='voltage (mV)')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.show()

plt.figure(4)
plt.suptitle("Izhikevich model - Low Thresholding Spiking")
plt.plot(LTS_neuron, 'b', label='voltage (mV)')
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
plt.suptitle("Izhikevich model - Thalamic Reticular")
plt.plot(TR_neuron, 'b', label='voltage (mV)')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.show()

