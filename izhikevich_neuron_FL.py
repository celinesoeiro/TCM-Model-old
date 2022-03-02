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


import matplotlib.pyplot as plt

from izhikevich_neuron import izhikevich_neuron


RS_neuron = izhikevich_neuron(
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
IB_neuron = izhikevich_neuron(
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
FS_neuron = izhikevich_neuron(
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
LTS_neuron = izhikevich_neuron(
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
TC_neuron_rest = izhikevich_neuron(
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
TC_neuron_depolarized = izhikevich_neuron(
    a = 0.02,
    b = 0.25,
    c = -65,
    d = 2.05,
    simulation_time = 100,
    time_step = 0.1,
    excitatory_neurons = 800,
    inhibitory_neurons = 200,
    input_voltage = -1.5,
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
plt.plot(TC_neuron_rest, 'b', label='voltage (mV)')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.show()

plt.figure(6)
plt.suptitle("Izhikevich model - Thalamic Reticular")
plt.plot(TC_neuron_depolarized, 'b', label='voltage (mV)')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.show()

