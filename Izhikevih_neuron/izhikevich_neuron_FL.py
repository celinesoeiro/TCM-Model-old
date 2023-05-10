"""

@author: Celine Soeiro

@description: Izhikevich neurons parameters used by FL Model

RS:
    a = 0.02 | b = 0.2 | c = -65 | d = 8 | Idc = 2.5 | Vp = 30

IB:
    a = 0.02 | b = 0.2 | c = -55 | d = 4 | Idc = 2.5 | Vp = 30
    
FS:
    a = 0.1 | b = 0.2 | c = -65 | d = 2 | Idc = 3.2 | Vp = 30
    
LTS:
    a = 0.02 | b = 0.25 | c = -65 | d = 2 | Idc = 0 | Vp = 30
    
TC:
    a = 0.02 | b = 0.25 | c = -65 | d = 0.05 | Idc = 0 | Vp = 30

TR: 
    a = 0.02 | b = 0.25 | c = -65 | d = 2.05 | Idc = 0 | Vp = 30

"""


import matplotlib.pyplot as plt

from izhikevich_neuron import izhikevich_neuron

RS_neuron, I_RS = izhikevich_neuron(
    params = {'a': 0.02, 'b': 0.2, 'c': -65, 'd': 8},
    current_value = 2.5,
    current_start = 100,
    current_finish = 900,
    voltage_pick = 30,
    simulation_time = 100,
    time_step = 0.1,
    neuron_type = 'excitatory',
    initial_voltage = -65
    )

IB_neuron, I_IB = izhikevich_neuron(
    params = {'a': 0.02, 'b': 0.2, 'c': -55, 'd': 4},
    current_value = 2.5,
    current_start = 100,
    current_finish = 900,
    voltage_pick = 30,
    simulation_time = 100,
    time_step = 0.1,
    neuron_type = 'excitatory',
    )

FS_neuron, I_FS = izhikevich_neuron(
    params = {'a': 0.1, 'b': 0.2, 'c': -65, 'd': 2},
    current_value = 3.2,
    current_start = 100,
    current_finish = 900,
    voltage_pick = 30,
    simulation_time = 100,
    time_step = 0.1,
    neuron_type = 'inhibitory',
    )

LTS_neuron, I_LTS = izhikevich_neuron(
    params = {'a': 0.02, 'b': 0.25, 'c': -65, 'd': 2},
    current_value = 0,
    current_start = 100,
    current_finish = 900,
    voltage_pick = 30,
    simulation_time = 100,
    time_step = 0.1,
    neuron_type = 'inhibitory',
    )

TC_neuron_relay, I_TC = izhikevich_neuron(
    params = {'a': 0.02, 'b': 0.25, 'c': -65, 'd': 0.05},
    current_value = 0,
    current_start = 100,
    current_finish = 900,
    voltage_pick = 30,
    simulation_time = 100,
    time_step = 0.1,
    neuron_type = 'excitatory',
    )

TC_neuron_reticular, I_TR = izhikevich_neuron(
    params = {'a': 0.02, 'b': 0.25, 'c': -65, 'd': 2.05},
    current_value = 0,
    current_start = 100,
    current_finish = 900,
    voltage_pick = 30,
    simulation_time = 100,
    time_step = 0.1,
    neuron_type = 'inhibitory',
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
plt.plot(TC_neuron_relay, 'b', label='voltage (mV)')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.show()

plt.figure(6)
plt.suptitle("Izhikevich model - Thalamic Reticular")
plt.plot(TC_neuron_reticular, 'b', label='voltage (mV)')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.show()


