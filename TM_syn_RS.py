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

from model_functions import izhikevich_dudt, izhikevich_dvdt, synapse_current, synapse_recovery, synapse_utilization

ms = 1000           # 1ms
rate = 20 * 1/ms    # spike rate 
bin_size = 1        # bin size 
tmax = 2 * ms       # the total lenght of the spike train
dt = rate

# Tsodkys and Markram synapse model
t_s_E = 3         # decay time constante of I (PSC current)
t_s_I = 11        # decay time constante of I (PSC current)

## Depression synapses parameters for excitatory synapses
t_f_E_D = 17
t_d_E_D = 671
U_E_D = 0.5
A_E_D = 0.63

## Facilitation synapses parameters for excitatory synapses
t_f_E_F = 670
t_d_E_F = 138
U_E_F = 0.09
A_E_F = 0.2

## Depression synapses parameters for inhibitory synapses
t_f_I_D = 21
t_d_I_D = 706
U_I_D = 0.25
A_I_D = 0.75

## Facilitation synapses parameters for inhibitory synapses
t_f_I_F = 376
t_d_I_F = 45
U_I_F = 0.016
A_I_F = 0.08

# Izhikevich neuron model
vp = 30     # voltage peak
vr = -65    # voltage threshold
rs_params = {'a': 0.02, 'b': 0.2, 'c': -65, 'd': 8}  # Regular Spiking
I_rs = 3.5*7

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

W_coupling = 5
td_syn = 1

# =============================================================================
# EXCITATORY - DEPRESSION
# =============================================================================

v = np.zeros((1, sim_steps)); v[0][0] = vr;
u = np.zeros((1, sim_steps)); u[0][0] = vr*rs_params['b'];

R_E_D = np.zeros((1, sim_steps)) # R for Excitatory Depression
u_E_D = np.zeros((1, sim_steps)) # u for Excitatory Depression
G_E_D = np.zeros((1, sim_steps)) # I for Excitatory Depression
R_E_D[0][0] = 1

PSC_E_D = np.zeros((1, sim_steps))
AP_E_D = np.zeros((1, sim_steps))

for t in time:
    AP_syn = spikes_poisson[t]
    
    AP_aux = 0
    v_aux = 1*v[0][t - 1]
    u_aux = 1*u[0][t - 1]
    
    # Neuron - RS - Excitatory
    if (v_aux >= vp):
        AP_aux = 1
        AP_E_D[0][t] = t - 1
        v_aux = v[0][t]
        v[0][t] = c_rs
        u[0][t] = u_aux + d_rs
    else:
        AP_aux = 0
        dv = izhikevich_dvdt(v = v_aux, u = u_aux, I = I_rs)
        du = izhikevich_dudt(v = v_aux, u = u_aux, a = a_rs, b = b_rs)

        v[0][t] = v_aux + dt*(dv + W_coupling*PSC_E_D[0][t - 1])
        u[0][t] = u_aux + dt*du
    
    # Synapse var - Excitatory - Depression
    syn_u_aux_E_D = 1*u_E_D[0][t - 1]
    syn_R_aux_E_D = 1*R_E_D[0][t - 1]
    syn_G_aux_E_D = 1*G_E_D[0][t - 1]
        
    # Synapse - Excitatory - Depression
    syn_du_E_D = synapse_utilization(u = syn_u_aux_E_D, tau_f = t_f_E_D, U = U_E_D, AP = AP_syn, dt = dt)
    u_E_D[0][t] = syn_u_aux_E_D + syn_du_E_D
    
    syn_dR_E_D = synapse_recovery(R = syn_R_aux_E_D, tau_d = t_d_E_D, u_next = u_E_D[0][t], AP = AP_syn, dt = dt)
    R_E_D[0][t] = syn_R_aux_E_D + syn_dR_E_D
    
    syn_dG_E_D = synapse_current(I = syn_G_aux_E_D,tau_s = t_s_E, A = A_E_D, R = syn_R_aux_E_D, u_next = u_E_D[0][t], AP = AP_syn, dt = dt)
    G_E_D[0][t] = syn_G_aux_E_D + syn_dG_E_D
    PSC_E_D[0][t] = 1*G_E_D[0][t]

v_value = []
for ap in AP_E_D[0]:
    if (ap > 1):
        v_value.append(v[0][int(ap)])
        
fig, (ax1,ax2,ax3, ax4) = plt.subplots(4,1,figsize=(15, 15))
ax1.plot(time, spikes_poisson)
ax2.plot(time,PSC_E_D[0])
ax3.plot(time,v[0])
ax4.stem(np.arange(0,len(v_value)), v_value)
ax4.plot(np.arange(0,len(v_value)), v_value,  'o--', color='grey', alpha=0.3)

ax1.set_title('Trem de pulsos gerado por Poisson')
ax2.set_title('PSC - Excitatoria dominada por depressao')
ax3.set_title('Tensao do neuronio - Disparo Regular (RS)')
ax4.set_title('Valor de pico da tensao')

del v, u, t, v_aux, u_aux, AP_aux, v_value

# =============================================================================
# EXCITATORY - FACILITATION
# =============================================================================

v = np.zeros((1, sim_steps)); v[0][0] = vr;
u = np.zeros((1, sim_steps)); u[0][0] = vr*rs_params['b'];

R_E_F = np.zeros((1, sim_steps)) # R for Excitatory Facilitation
u_E_F = np.zeros((1, sim_steps)) # u for Excitatory Facilitation
G_E_F = np.zeros((1, sim_steps)) # I for Excitatory Facilitation
R_E_F[0][0] = 1

PSC_E_F = np.zeros((1, sim_steps))
AP_E_F = np.zeros((1, sim_steps))

for t in time:
    AP_syn = spikes_poisson[t]
    v_aux = 1*v[0][t - 1]
    u_aux = 1*u[0][t - 1]
    
    # Neuron - RS - Excitatory
    if (v_aux >= vp):
        AP_aux = 1
        AP_E_F[0][t] = t - 1
        v_aux = v[0][t]
        v[0][t] = c_rs
        u[0][t] = u_aux + d_rs
    else:
        AP_aux = 0
        dv = izhikevich_dvdt(v = v_aux, u = u_aux, I = I_rs)
        du = izhikevich_dudt(v = v_aux, u = u_aux, a = a_rs, b = b_rs)

        v[0][t] = v_aux + dt*(dv + W_coupling*PSC_E_F[0][t - 1])
        u[0][t] = u_aux + dt*du
    
    # Synapse var - Excitatory - Facilitation
    syn_u_aux_E_F = 1*u_E_F[0][t - 1]
    syn_R_aux_E_F = 1*R_E_F[0][t - 1]
    syn_G_aux_E_F = 1*G_E_F[0][t - 1]
        
    syn_du_E_F = synapse_utilization(u = syn_u_aux_E_F, tau_f = t_f_E_F, U = U_E_F, AP = AP_syn, dt = dt)
    u_E_F[0][t] = syn_u_aux_E_F + syn_du_E_F
    
    syn_dR_E_F = synapse_recovery(R = syn_R_aux_E_F, tau_d = t_d_E_F, u_next = u_E_F[0][t], AP = AP_syn, dt = dt)
    R_E_F[0][t] = syn_R_aux_E_F + syn_dR_E_F
    
    syn_dG_E_F = synapse_current(I = syn_G_aux_E_F, tau_s = t_s_E, A = A_E_F, R = syn_R_aux_E_F, u_next = u_E_F[0][t], AP = AP_syn,dt = dt)
    G_E_F[0][t] = syn_G_aux_E_F + syn_dG_E_F
    PSC_E_F[0][t] = 1*G_E_F[0][t]*v_aux
    
    AP_aux = 0
    v_aux = 1*v[0][t - 1]
    u_aux = 1*u[0][t - 1]
   
v_value = []
for ap in AP_E_F[0]:
    if (ap > 1):
        v_value.append(v[0][int(ap)])

fig, (ax1,ax2,ax3, ax4) = plt.subplots(4,1,figsize=(15, 15))
ax1.plot(time, spikes_poisson)
ax2.plot(time,PSC_E_F[0])
ax3.plot(time,v[0])
ax4.stem(np.arange(0,len(v_value)), v_value)
ax4.plot(np.arange(0,len(v_value)), v_value,  'o--', color='grey', alpha=0.3)

ax1.set_title('Trem de pulsos gerado por Poisson')
ax2.set_title('PSC - Excitatoria dominada por facilitacao')
ax3.set_title('Tensao do neuronio - Disparo Regular (RS)')
ax4.set_title('Valor de pico da tensao')

del v, u, t, v_aux, u_aux, AP_aux, v_value

# =============================================================================
# INHIBITORY - DEPRESSION
# =============================================================================

v = np.zeros((1, sim_steps)); v[0][0] = vr;
u = np.zeros((1, sim_steps)); u[0][0] = vr*rs_params['b'];

R_I_D = np.zeros((1, sim_steps)) # R for Inhibitory Depression
u_I_D = np.zeros((1, sim_steps)) # u for Inhibitory Depression
G_I_D = np.zeros((1, sim_steps)) # I for Inhibitory Depression
R_I_D[0][0] = 1

PSC_I_D = np.zeros((1, sim_steps))
AP_I_D = np.zeros((1, sim_steps))

for t in time:
    AP_aux = 0
    v_aux = 1*v[0][t - 1]
    u_aux = 1*u[0][t - 1]
    AP_syn = spikes_poisson[t]
    
    # Neuron - RS - Excitatory
    if (v_aux >= vp):
        AP_aux = 1
        AP_I_D[0][t] = t - 1
        v_aux = v[0][t]
        v[0][t] = c_rs
        u[0][t] = u_aux + d_rs
    else:
        AP_aux = 0
        dv = izhikevich_dvdt(v = v_aux, u = u_aux, I = I_rs)
        du = izhikevich_dudt(v = v_aux, u = u_aux, a = a_rs, b = b_rs)
 
        v[0][t] = v_aux + dt*(dv + W_coupling*PSC_I_D[0][t - 1])
        u[0][t] = u_aux + dt*du
        
    # Synapse var - Inhibitory - Depression
    syn_u_aux_I_D = 1*u_I_D[0][t - 1]
    syn_R_aux_I_D = 1*R_I_D[0][t - 1]
    syn_G_aux_I_D = 1*G_I_D[0][t - 1]
        
    # Synapse - Inhibitory - Depression
    syn_du_I_D = synapse_utilization(u = syn_u_aux_I_D, tau_f = t_f_I_D, U = U_I_D, AP = AP_syn, dt = dt)
    u_I_D[0][t] = syn_u_aux_I_D + syn_du_I_D
    
    syn_dR_I_D = synapse_recovery(R = syn_R_aux_I_D,tau_d = t_d_I_D, u_next = u_I_D[0][t], AP = AP_syn, dt = dt)
    R_I_D[0][t] = syn_R_aux_I_D + syn_dR_I_D
    
    syn_dG_I_D = synapse_current(I = syn_G_aux_I_D, tau_s = t_s_I, A = A_I_D, R = syn_R_aux_I_D, u_next = u_I_D[0][t], AP = AP_syn,dt = dt)
    G_I_D[0][t] = syn_G_aux_I_D + syn_dG_I_D
    PSC_I_D[0][t] = 1*G_I_D[0][t]*v_aux
    
v_value = []
for ap in AP_E_D[0]:
    if (ap > 1):
        v_value.append(v[0][int(ap)])
     
fig, (ax1,ax2,ax3, ax4) = plt.subplots(4,1,figsize=(15, 15))
ax1.plot(time, spikes_poisson)
ax2.plot(time,PSC_I_D[0])
ax3.plot(time,v[0])
ax4.stem(np.arange(0,len(v_value)), v_value)
ax4.plot(np.arange(0,len(v_value)), v_value,  'o--', color='grey', alpha=0.3)

ax1.set_title('Trem de pulsos gerado por Poisson')
ax2.set_title('PSC - Inibitoria dominada por depressao')
ax3.set_title('Tensao do neuronio - Disparo Regular (RS)')
ax4.set_title('Valor de pico da tensao')

del v, u, t, v_aux, u_aux, AP_aux, v_value

# =============================================================================
# INHIBITORY - FACILITATION
# =============================================================================

v = np.zeros((1, sim_steps)); v[0][0] = vr;
u = np.zeros((1, sim_steps)); u[0][0] = vr*rs_params['b'];

R_I_F = np.zeros((1, sim_steps)) # R for Inhibitory Facilitation
u_I_F = np.zeros((1, sim_steps)) # u for Inhibitory Facilitation
G_I_F = np.zeros((1, sim_steps)) # I for Inhibitory Facilitation
R_I_F[0][0] = 1

PSC_I_F = np.zeros((1, sim_steps))
AP_I_F = np.zeros((1, sim_steps))

for t in time:
    AP_syn = spikes_poisson[t]
    
    AP_aux = 0
    v_aux = 1*v[0][t - 1]
    u_aux = 1*u[0][t - 1]
    
    # Neuron - RS - Excitatory
    if (v_aux >= vp):
        AP_aux = 1
        AP_I_F[0][t] = t - 1
        v_aux = v[0][t]
        v[0][t] = c_rs
        u[0][t] = u_aux + d_rs
    else:
        AP_aux = 0
        dv = izhikevich_dvdt(v = v_aux, u = u_aux, I = I_rs)
        du = izhikevich_dudt(v = v_aux, u = u_aux, a = a_rs, b = b_rs)
    
        v[0][t] = v_aux + dt*(dv + W_coupling*PSC_I_F[0][t - 1])
        u[0][t] = u_aux + dt*du
    
    # Synapse var - Inhibitory - Faciliation
    syn_u_aux_I_F = 1*u_I_F[0][t - 1]
    syn_R_aux_I_F = 1*R_I_F[0][t - 1]
    syn_G_aux_I_F = 1*G_I_F[0][t - 1]   
        
    # Synapse - Inhibitory - Facilitation
    syn_du_I_F = synapse_utilization(u = syn_u_aux_I_F, tau_f = t_f_I_F, U = U_I_F, AP=AP_syn, dt = dt)
    u_I_F[0][t] = syn_u_aux_I_F + syn_du_I_F
    
    syn_dR_I_F = synapse_recovery(R = syn_R_aux_I_F, tau_d = t_d_I_F, u_next = u_I_F[0][t], AP = AP_syn, dt = dt)
    R_I_F[0][t] = syn_R_aux_I_F + syn_dR_I_F
    
    syn_dG_I_F = synapse_current(I = syn_G_aux_I_F, tau_s = t_s_I, A = A_I_F, R = syn_R_aux_I_F,u_next = u_I_F[0][t], AP = AP_syn, dt = dt)
    G_I_F[0][t] = syn_G_aux_I_F + syn_dG_I_F
    PSC_I_F[0][t] = 1*G_I_F[0][t]*v_aux
    
v_value = []
for ap in AP_E_D[0]:
    if (ap > 1):
        v_value.append(v[0][int(ap)])
    
fig, (ax1,ax2,ax3, ax4) = plt.subplots(4,1,figsize=(15, 15))
ax1.plot(time, spikes_poisson)
ax2.plot(time,PSC_I_F[0])
ax3.plot(time,v[0])
ax4.stem(np.arange(0,len(v_value)), v_value)
ax4.plot(np.arange(0,len(v_value)), v_value,  'o--', color='grey', alpha=0.3)

ax1.set_title('Trem de pulsos gerado por Poisson')
ax2.set_title('PSC - Inibitoria dominada por facilitacao')
ax3.set_title('Tensao do neuronio - Disparo Regular (RS)')
ax4.set_title('Valor de pico da tensao')
