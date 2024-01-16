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

# currents = [2.5, 2.5, 3.2, 0 ,0 ,0]; # article
currents = [3.5, 3.6, 3.8, 0.6, 0.5, 0.5] # code


import matplotlib.pyplot as plt

from izhikevich_neuron import izhikevich_neuron

RS_neuron, I_RS = izhikevich_neuron(
    params = {'a': 0.02, 'b': 0.2, 'c': -65, 'd': 8},
    neuron_type = 'excitatory',
    voltage_pick = 30,
    simulation_time = 100,
    time_step = 0.01,
    current_value = currents[0],
    current_start = 500,
    current_finish = 10000,
    )

# Plots
plt.figure(1)
plt.suptitle("Izhikevich model - Regular Spiking")
plt.subplot(2,1,1)
plt.plot(RS_neuron, 'b', label='voltage (mV)')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(I_RS, label='Current')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.show()

IB_neuron, I_IB = izhikevich_neuron(
    params = {'a': 0.02, 'b': 0.2, 'c': -55, 'd': 4},
    current_value = currents[1],
    current_start = 500,
    current_finish = 10000,
    voltage_pick = 30,
    simulation_time = 100,
    time_step = 0.01,
    neuron_type = 'excitatory',
    )

plt.figure(2)
plt.suptitle("Izhikevich model - Intrinically Bursting")
plt.subplot(2,1,1)
plt.plot(IB_neuron, 'b', label='voltage (mV)')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(I_IB, label='Current')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.show()

FS_neuron, I_FS = izhikevich_neuron(
    params = {'a': 0.1, 'b': 0.2, 'c': -65, 'd': 2},
    current_value = currents[2],
    current_start = 500,
    current_finish = 10000,
    voltage_pick = 30,
    simulation_time = 100,
    time_step = 0.01,
    neuron_type = 'inhibitory',
    )

plt.figure(3)
plt.suptitle("Izhikevich model - Fast Spiking")
plt.subplot(2,1,1)
plt.plot(FS_neuron, 'b', label='voltage (mV)')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(I_FS, label='Current')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.show()

LTS_neuron, I_LTS = izhikevich_neuron(
    params = {'a': 0.02, 'b': 0.25, 'c': -65, 'd': 2},
    current_value = currents[3],
    current_start = 500,
    current_finish = 10000,
    voltage_pick = 30,
    simulation_time = 100,
    time_step = 0.01,
    neuron_type = 'inhibitory',
    )

plt.figure(4)
plt.suptitle("Izhikevich model - Low Thresholding Spiking")
plt.subplot(2,1,1)
plt.plot(LTS_neuron, 'b', label='voltage (mV)')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(I_LTS, label='Current')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.show()

TC_neuron_relay, I_TC = izhikevich_neuron(
    params = {'a': 0.02, 'b': 0.25, 'c': -65, 'd': 0.05},
    current_value = currents[4],
    current_start = 500,
    current_finish = 10000,
    voltage_pick = 30,
    simulation_time = 100,
    time_step = 0.01,
    neuron_type = 'excitatory',
    )

plt.figure(5)
plt.suptitle("Izhikevich model - Thalamo-Cortical")
plt.subplot(2,1,1)
plt.plot(TC_neuron_relay, 'b', label='voltage (mV)')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(I_TC, label='Current')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.show()

TC_neuron_reticular, I_TR = izhikevich_neuron(
    params = {'a': 0.02, 'b': 0.25, 'c': -65, 'd': 2.05},
    current_value = currents[5],
    current_start = 500,
    current_finish = 10000,
    voltage_pick = 30,
    simulation_time = 100,
    time_step = 0.01,
    neuron_type = 'inhibitory',
    )

plt.figure(6)
plt.suptitle("Izhikevich model - Thalamic Reticular")
plt.subplot(2,1,1)
plt.plot(TC_neuron_reticular, 'b', label='voltage (mV)')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(I_TR, label='Current')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.show()

