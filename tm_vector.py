#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 21:29:56 2024

@author: celinesoeiro
"""

import numpy as np
from random import seed, random

from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

from model_functions import poissonSpikeGen, tm_synapse_poisson_eq, poisson_spike_generator

seed(1)
random_factor = random()

ms = 1000                               # 1 second = 1000 miliseconds
dt = 10/ms                              # time step of 10 ms
simulation_time = 100                     # simulation time in seconds (must be a multiplacative of 3 under PD+DBS condition)
samp_freq = int(1/dt)                  # sampling frequency in Hz
T = int((simulation_time + 0.5)*ms)          # Simulation time in ms with 1 extra second to reach the steady state and trash later
sim_steps = int(simulation_time/dt)         # number of simulation steps
chop_till = 1*samp_freq;                # Cut the first 1 seconds of the simulation
time = np.arange(1, sim_steps)

tm_synapse_params_excitatory = {
    't_f': [670, 17, 326],
    't_d': [138, 671, 329],
    'U': [0.09, 0.5, 0.29],
    'distribution': [0.2, 0.63, 0.17],
    'distribution_T_D': [0, 1, 0], # Depressing
    'distribution_D_T': [1, 0, 0], # Facilitating
    't_s': 3,
    }

tm_synapse_params_inhibitory = {
    't_f': [376, 21, 62],
    't_d': [45, 706, 144],
    'U': [0.016, 0.25, 0.32],
    'distribution': [0.08, 0.75, 0.17],
    't_s': 11,
    }

t_f = tm_synapse_params_excitatory['t_f']
t_d = tm_synapse_params_excitatory['t_d']
t_s = tm_synapse_params_excitatory['t_s']
U = tm_synapse_params_excitatory['U']
A = tm_synapse_params_excitatory['distribution']

R = np.zeros((3, sim_steps))
u = np.zeros((3, sim_steps))
I = np.zeros((3, sim_steps))
ap = np.zeros((1, sim_steps))

W_ps = [[1 * np.random.random() for _ in range(2)] for _ in range(6)]
poisson_firing = 20 + 2 * np.random.random()
[spike_T, I_T] = poisson_spike_generator(num_steps = sim_steps, dt = dt, num_neurons = 1, thalamic_firing_rate = poisson_firing, current_value=None)

plt.figure(2)
plt.eventplot(I_T)
plt.show()

for p in range(3):    
    for i in time:
        print(p, i)
        # u -> utilization factor -> resources ready for use
        u[p][i] = u[p - 1][i - 1] + -dt*u[p - 1][i - 1]/t_f[p - 1] + U[p - 1]*(1 - u[p - 1][i - 1])*I_T[0][i - 1]
        # x -> availabe resources -> Fraction of resources that remain available after neurotransmitter depletion
        R[p][i] = R[p - 1][i - 1] + dt*(1 - R[p - 1][i - 1])/t_d[p - 1] - u[p - 1][i - 1]*R[p - 1][i - 1]*I_T[0][i - 1]
        # PSC
        I[p][i] = I[p - 1][i - 1] + -dt*I[p - 1][i - 1]/t_s + A[p - 1]*R[p - 1][i - 1]*u[p - 1][i - 1]*I_T[0][i - 1]
        
        # r[p][i + 1] = r[p][i] + dt*(-r[p][i]/tau_f[p] + U[p]*(1 - r[p][i])*ap[0][i - t_delay])
        # x[p][i + 1] = x[p][i] + dt*((1 - x[p][i])/tau_d[p] - r[p][i]*x[p][i]*ap[0][i - t_delay])
        # Is[p][i + 1] = Is[p][i] + dt*(-Is[p][i]/tau_s + A[p]*r[p][i]*x[p][i]*ap[0][i - t_delay])
    
Ipost = np.sum(I)

plt.plot(Ipost)
