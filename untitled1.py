#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 21:56:45 2024

@author: celinesoeiro
"""
import numpy as np

ms = 1000                               # 1 second = 1000 miliseconds
dt = 100/ms                             # time step of 10 ms
simulation_time = 15                     # simulation time in seconds
fs = int(ms/dt)                  # sampling frequency in Hz
T = int((simulation_time)*ms)       # Simulation time in ms with 1 extra second to reach the steady state and trash later
sim_steps = int(T/dt)                   # number of simulation steps
chop_till = 1*fs;                # Cut the first 1 seconds of the simulation

td_syn = 1                          # Synaptic transmission delay (fixed for all synapses in the TCM)
td_thalamus_cortex = 15                 # time delay from thalamus to cortex (ms) (transmission time delay)
td_cortex_thalamus = 20                 # time delay from cortex to thalamus (ms) (transmission time delay)  
td_layers = 8                           # time delay between the layers in cortex and nuclei in thalamus (ms) (PSC delay)
td_within_layers = 1                    # time delay within a structure (ms)

step = int(sim_steps/3) # 1 part is zero, 1 part is dbs and another part is back to zero -> pulse

dbs_freq = 80

t_f_E =  [670, 17, 326]
t_d_E = [138, 671, 329]
U_E =  [0.09, 0.5, 0.29]
A_E = [0.2, 0.63, 0.17]
t_s_E = 3

p = 3

# I_dbs = np.zeros((2, sim_steps))
# f_dbs = dbs_freq

# dbs_duration = step
# dbs_amplitude = 1   # 1mA

# T_dbs = np.round(fs/f_dbs)
# dbs_arr = np.arange(0, dbs_duration, T_dbs)
# I_dbs_full = np.zeros((1, dbs_duration))

# for i in dbs_arr:
#     I_dbs_full[0][int(i)] = dbs_amplitude

# I_dbs_pre = 1*np.concatenate((
#     np.zeros((1, step)), 
#     I_dbs_full, 
#     np.zeros((1, step))
#     ),axis=1)

# R_dbs = np.zeros((3, sim_steps))
# u_dbs = np.ones((3, sim_steps))
# Is_dbs = np.zeros((3, sim_steps))

# for p in range(3):
#     print(p)
#     for i in range(td_syn, sim_steps - 1):
#         # u -> utilization factor -> resources ready for use
#         u_dbs[p][i] = u_dbs[p][i - 1] + -dt*u_dbs[p][i - 1]/t_f_E[p] + U_E[p]*(1 - u_dbs[p][i - 1])*I_dbs_pre[0][i- td_syn]
#         # x -> availabe resources -> Fraction of resources that remain available after neurotransmitter depletion
#         R_dbs[p][i] = R_dbs[p][i - 1] + dt*(1 - R_dbs[p][i - 1])/t_d_E[p] - u_dbs[p][i - 1]*R_dbs[p][i - 1]*I_dbs_pre[0][i- td_syn]
#         # PSC
#         Is_dbs[p][i] = Is_dbs[p][i - 1] + -dt*Is_dbs[p][i - 1]/t_s_E + A_E[p]*R_dbs[p][i - 1]*u_dbs[p][i - 1]*I_dbs_pre[0][i- td_syn]
        
# I_dbs_post = np.sum(Is_dbs, 0)

# I_dbs[0] = I_dbs_pre[0]
# I_dbs[1] = I_dbs_post

u = np.zeros((1, p))
R = np.ones((1, p))
I = np.zeros((1, p))

AP = 1
for j in range(3):
    print(j)
    print(u[0][j - 1])
    print(R[0][j - 1])
    print(I[0][j - 1])
    # u -> utilization factor -> resources ready for use
    u[0][j] = u[0][j - 1] + -dt*u[0][j - 1]/t_f_E[j] + U_E[j]*(1 - u[0][j - 1])*AP
    # x -> availabe resources -> Fraction of resources that remain available after neurotransmitter depletion
    R[0][j] = R[0][j - 1] + dt*(1 - R[0][j - 1])/t_d_E[j] - u[0][j]*R[0][j - 1]*AP
    # PSC
    I[0][j] = I[0][j - 1] + -dt*I[0][j - 1]/t_s_E + A_E[j]*R[0][j - 1]*u[0][j]*AP
    
Ipost = np.sum(I)

