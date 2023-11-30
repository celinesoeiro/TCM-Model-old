"""
Created on Wed Nov 15 21:27:23 2023

@author: celinesoeiro
"""
import matplotlib.pyplot as plt
import numpy as np

from izhikevich_neuron import izhikevich_neuron

ms = 1000
dt = 1/ms
vp = 30

RS_neuron, I_RS = izhikevich_neuron(
    # params = {'a': 0.02, 'b': 0.2, 'c': -65, 'd': 8}, # RS
    params = {'a': 0.02, 'b': 0.2, 'c': -55, 'd': 4},  # IB
    current_value = 4,
    current_start = 0,    
    current_finish = -1,    
    voltage_pick = vp,
    simulation_time = 1*ms, 
    time_step = dt,
    neuron_type = 'excitatory',
    initial_voltage = -65
    )

plt.figure(1)
plt.suptitle("Izhikevich model - Regular Spiking")
plt.plot(RS_neuron, 'b', label='voltage (mV)')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.show()

AP = []

for i in RS_neuron:
    if(i > vp):
        AP.append(1)
    else:
        AP.append(0)

def tm_synapse_eq(r, x, Is, AP, tau_f, tau_d, tau_s, U, A, dt):        
    for p in range(0, 3):
        # Solve EDOs using Euler method
        r[0][p] = r[0][p] + dt*(-r[0][p]/tau_f[p] + U[p]*(1 - r[0][p])*AP)
        x[0][p] = x[0][p] + dt*((1 - x[0][p])/tau_d[p - 1] - (r[0][p] + U[p]*(1 - r[0][p]))*x[0][p]*AP)
        Is[0][p] = Is[0][p] + dt*(-Is[0][p]/tau_s + A[p]*x[0][p]*(r[0][p] + U[p]*(1 - r[0][p]))*AP)
        
    Ipost = np.sum(Is)
    
    tm_syn_inst = dict()
    tm_syn_inst['r'] = r
    tm_syn_inst['x'] = x
    tm_syn_inst['Is'] = Is
    tm_syn_inst['Ipost'] = np.around(Ipost, decimals=6)
        
    return tm_syn_inst

r = np.zeros((1, 3))
x = np.zeros((1, 3))
Is = np.zeros((1, 3))

# Excitatory
tau_f_E = [670, 17, 326]
tau_d_E = [138, 671, 329]
tau_s_E = 3
U_E = [0.09, 0.5, 0.29]
A_E = [0.2, 0.63, 0.17]

tm_E_values = []
for i in RS_neuron:
    if (i > vp):
        tm = tm_synapse_eq(r, x, Is, 1, tau_f_E, tau_d_E, tau_s_E, U_E, A_E, dt)
        tm_E_values.append(tm['Ipost'])
    else:
        tm = tm_synapse_eq(r, x, Is, 0, tau_f_E, tau_d_E, tau_s_E, U_E, A_E, dt)
        tm_E_values.append(tm['Ipost'])

plt.figure(1)
plt.suptitle("TM model - Regular Spiking Post Synaptic Response")
plt.plot(tm_E_values, 'b', label='voltage (mV)')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.show()

# Inhibitory
tau_f_I = [376, 21, 62]
tau_d_I = [45, 706, 144]
tau_s_I = 11
U_I = [0.016, 0.25, 0.32]
A_I = [0.08, 0.75, 0.17]

tm_I_values = []
for i in RS_neuron:
    if (i > vp):
        tm = tm_synapse_eq(r, x, Is, 1, tau_f_I, tau_d_I, tau_s_I, U_I, A_I, dt)
        tm_I_values.append(tm['Ipost'])
    else:
        tm = tm_synapse_eq(r, x, Is, 0, tau_f_I, tau_d_I, tau_s_I, U_I, A_I, dt)
        tm_I_values.append(tm['Ipost'])

plt.figure(1)
plt.suptitle("TM model - Regular Spiking Post Synaptic Response")
plt.plot(tm_I_values, 'b', label='voltage (mV)')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.show()
