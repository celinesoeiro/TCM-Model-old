"""

@author: Celine Soeiro

@description: Izhikevich neurons parameters used by FL Model

Tonic Spiking:
    a = 0.02 | b = 0.2 | c = -65 | d = 6 | I = [14, 14] | Vp = 30 | V = -70 | tau = 0.25

Phasic Spiking:
    a = 0.02 | b = 0.25 | c = -65 | d = 6 | I = 0.5 | Vp = 30 | V = -64 | tau = 0.25
    
Tonic Bursting:
    a = 0.02 | b = 0.2 | c = -50 | d = 2 | I = 15 | Vp = 30 | V = -70 | tau = 0.25
    
LTS:
    a = 0.02 | b = 0.25 | c = -65 | d = 2 | Idc = 0 | Vp = 30 | V = -70 | tau = 0.25
    
TC:
    a = 0.02 | b = 0.25 | c = -65 | d = 0.05 | Idc = 0 | Vp = 30 | V = -70 | tau = 0.25

TR: 
    a = 0.02 | b = 0.25 | c = -65 | d = 2.05 | Idc = 0 | Vp = 30 | V = -70 | tau = 0.25
    
"""

from izhikevich_neuron import izhikevich_neuron

import matplotlib.pyplot as plt

tonic_spiking_neuron, I_TS = izhikevich_neuron(
    params = {'a': 0.02, 'b': 0.2, 'c': -65, 'd': 6},
    current_value = 14,
    current_start = 0,
    current_finish = 250,
    voltage_pick = 30,
    initial_voltage = -70,
    simulation_time = 100,
    time_step = 0.25,
    neurons = 10,
    neuron_type = 'excitatory'
    )

phasic_spiking_neuron, I_PS = izhikevich_neuron(
    params = {'a': 0.02, 'b': 0.25, 'c': -65, 'd': 6},
    current_value = 0.5,
    current_start = 50,
    current_finish = 200,
    voltage_pick = 30,
    initial_voltage = -64,
    simulation_time = 100,
    time_step = 0.25,
    neurons = 10,
    neuron_type = 'excitatory'
    )

tonic_bursting_neuron, I_TB = izhikevich_neuron(
    params = {'a': 0.02, 'b': 0.2, 'c': -50, 'd': 2},
    current_value = 15,
    current_start = 50,
    current_finish = 200,
    voltage_pick = 30,
    initial_voltage = -70,
    simulation_time = 100,
    time_step = 0.25,
    neurons = 10,
    neuron_type = 'excitatory'
    )

phasic_bursting_neuron, I_PB = izhikevich_neuron(
    params = {'a': 0.02, 'b': 0.25, 'c': -55, 'd': 0.05},
    current_value = 0.6,
    current_start = 50,
    current_finish = 200,
    voltage_pick = 30,
    initial_voltage = -64,
    simulation_time = 100,
    time_step = 0.2,
    neurons = 10,
    neuron_type = 'excitatory'
    )

mixed_mode_neuron, I_MM = izhikevich_neuron(
    params = {'a': 0.02, 'b': 0.2, 'c': -55, 'd': 4},
    current_value = 10,
    current_start = 50,
    current_finish = 200,
    voltage_pick = 30,
    initial_voltage = -70,
    simulation_time = 100,
    time_step = 0.25,
    neurons = 10,
    neuron_type = 'excitatory'
    )

spike_frequency_adapt, I_SF = izhikevich_neuron(
    params = {'a': 0.01, 'b': 0.2, 'c': -65, 'd': 8},
    current_value = 30,
    current_start = 50,
    current_finish = 200,
    voltage_pick = 30,
    initial_voltage = -70,
    simulation_time = 100,
    time_step = 0.25,
    neurons = 10,
    neuron_type = 'excitatory'
    )


# Plots
plt.figure(1)
plt.suptitle("Izhikevich model - Tonic Spiking")
plt.plot(tonic_spiking_neuron, 'b', label='voltage (mV)')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.show()

plt.figure(2)
plt.suptitle("Izhikevich model - Phasic Spiking")
plt.plot(phasic_spiking_neuron, 'b', label='voltage (mV)')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.show()

plt.figure(3)
plt.suptitle("Izhikevich model - Tonic Bursting")
plt.plot(tonic_bursting_neuron, 'b', label='voltage (mV)')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.show()

plt.figure(4)
plt.suptitle("Izhikevich model - Phasic Bursting")
plt.plot(phasic_bursting_neuron, 'b', label='voltage (mV)')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.show()

plt.figure(5)
plt.suptitle("Izhikevich model - Mixed Mode")
plt.plot(mixed_mode_neuron, 'b', label='voltage (mV)')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.show()

plt.figure(6)
plt.suptitle("Izhikevich model - Spike Frequency Adaptation")
plt.plot(spike_frequency_adapt, 'b', label='voltage (mV)')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.show()