#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 22:47:01 2024

@author: celinesoeiro
"""

import matplotlib.pyplot as plt
import numpy as np
from random import seed, random
import seaborn as sns
sns.set()

seed(1)
random_factor = random()

from model_functions import izhikevich_dudt, izhikevich_dvdt

ms = 1000           # 1ms
rate = 5 * 1/ms    # spike rate 
bin_size = 1        # bin size 
tmax = 1 * ms       # the total lenght of the spike train
dt = rate

# Tsodkys and Markram synapse model
t_s_E = 3         # decay time constante of I (PSC current)
t_s_I = 11        # decay time constante of I (PSC current)

# Should sum to 100% (70% + 30%)
## Depression synapses parameters for excitatory synapses
t_f_E_D = 17
t_d_E_D = 671
U_E_D = 0.5
A_E_D = 0.7

# Izhikevich neuron model
vp = 30     # voltage peak
vr = -65    # voltage threshold
rs_params = {'a': 0.02, 'b': 0.2, 'c': -65, 'd': 8}  # Regular Spiking
I_rs = 3.5

a_rs = rs_params['a']
b_rs = rs_params['b']
c_rs = rs_params['c'] + 15*random_factor**2
d_rs = rs_params['d'] - 6*random_factor**2

def homogeneous_poisson(rate, tmax, bin_size): 
    nbins = np.floor(tmax/bin_size).astype(int) 
    prob_of_spike = rate * bin_size 
    spikes = np.random.rand(nbins) < prob_of_spike 
    return spikes * 1

spikes_poisson = homogeneous_poisson(rate, tmax, bin_size) 
time = np.arange(len(spikes_poisson)) * bin_size 

sim_steps = len(time)

R_E_D = np.zeros((1, sim_steps)) # R for Excitatory Depression
u_E_D = np.zeros((1, sim_steps)) # u for Excitatory Depression
I_E_D = np.zeros((1, sim_steps)) # I for Excitatory Depression
R_E_D[0][0] = 1

v = np.zeros((1, sim_steps)); v[0][0] = vr;
u = np.zeros((1, sim_steps)); u[0][0] = vr*rs_params['b'];

W_coupling = 1
td_syn = 1
PSC_E_D = np.zeros((1, sim_steps))

def synapse_utilization(u, tau_f, U, AP, dt):
    return -(dt/tau_f)*u + U*(1 - u)*AP

def synapse_recovery(R, tau_d, u_next, AP, dt):
    return (dt/tau_d)*(1 - R) - u_next*R*AP

def synapse_current(I, tau_s, A, R, u_next, AP, dt):
    return -(dt/tau_s)*I + A*R*u_next*AP

for t in time:
    AP_syn = spikes_poisson[t]
    # Synapse var - Excitatory - Depression
    syn_u_aux_E_D = 1*u_E_D[0][t - 1]
    syn_R_aux_E_D = 1*R_E_D[0][t - 1]
    syn_I_aux_E_D = 1*I_E_D[0][t - 1]
        
    # Synapse - Excitatory - Depression
    syn_du_E_D = synapse_utilization(u = syn_u_aux_E_D, 
                                     tau_f = t_f_E_D, 
                                     U = U_E_D, 
                                     AP = AP_syn, 
                                     dt = dt)
    u_E_D[0][t] = syn_u_aux_E_D + syn_du_E_D
    
    syn_dR_E_D = synapse_recovery(R = syn_R_aux_E_D, 
                              tau_d = t_d_E_D, 
                              u_next = u_E_D[0][t], 
                              AP = AP_syn, 
                              dt = dt)
    R_E_D[0][t] = syn_R_aux_E_D + syn_dR_E_D
    
    syn_dI_E_D = synapse_current(I = syn_I_aux_E_D, 
                             tau_s = t_s_E, 
                             A = A_E_D, 
                             R = syn_R_aux_E_D, 
                             u_next = u_E_D[0][t], 
                             AP = AP_syn, 
                             dt = dt)
    I_E_D[0][t] = syn_I_aux_E_D + syn_dI_E_D
    PSC_E_D[0][t] = I_E_D[0][t]
    
    AP_aux = 0
    v_aux = 1*v[0][t - 1]
    u_aux = 1*u[0][t - 1]
    
    # Neuron - RS - Excitatory
    if (v_aux >= vp):
        AP_aux = 1
        v_aux = v[0][t]
        v[0][t] = c_rs
        u[0][t] = u_aux + d_rs
    else:
        AP_aux = 0
        dv = izhikevich_dvdt(v = v_aux, u = u_aux, I = I_rs*PSC_E_D[0][t - 1])
        du = izhikevich_dudt(v = v_aux, u = u_aux, a = a_rs, b = b_rs)

    v[0][t] = v_aux + dt*dv
    u[0][t] = u_aux + dt*du
    
fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(15, 10))
ax1.plot(time, spikes_poisson)
ax2.plot(time,PSC_E_D[0])
ax3.plot(time,v[0])

ax1.set_title('Trem de pulsos gerado por Poisson')
ax2.set_title('PSC - Excitatoria dominada por depressao')
ax3.set_title('Tensao do neuronio')
