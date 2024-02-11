#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 22:47:01 2024

@author: celinesoeiro
"""

import numpy as np

random_factor = np.random.rand()

from model_functions import homogeneous_poisson, tm_syn_excit_dep, tm_syn_excit_fac, tm_syn_inib_dep, tm_syn_inib_fac

ms = 1000           # 1ms
rate = 20 * 1/ms    # spike rate 
bin_size = 1        # bin size 
tmax = 1 * ms       # the total lenght of the spike train
dt = rate

# Izhikevich neuron model
vp = 30     # voltage peak
vr = -65    # voltage threshold
I = 25

neuron_type = 'Disparo Regular (RS)'

a = 0.02
b = 0.2
c = -65 + 15*random_factor**2
d = 8 - 0.6*random_factor**2

# spikes = homogeneous_poisson(rate, tmax, bin_size) 
# time = np.arange(1,len(spikes)) * bin_size 
# sim_steps = len(spikes)

dt2 = 100/ms
f = 30
fs = int(ms/dt2)   
amplitude = 1   # 1mA

T_dbs = int(np.round(fs/f))
dbs_arr = np.arange(0, fs, T_dbs)
spikes = np.zeros((1, int(fs)))
time = np.arange(1, len(spikes))
sim_steps = len(spikes)

for i in dbs_arr:
    spikes[0][int(i)] = amplitude

W = 5e2

# =============================================================================
# EXCITATORY - DEPRESSION
# =============================================================================
print('-- Excitatory - Depression')
tm_syn_excit_dep(sim_steps, vr, vp, a, b, c, d, spikes[0], time, I, W, dt, neuron_type)

# =============================================================================
# EXCITATORY - FACILITATION
# =============================================================================
print('-- Excitatory - Facilitation')
tm_syn_excit_fac(sim_steps, vr, vp, a, b, c, d, time, dt, spikes[0], I, W, neuron_type)

# =============================================================================
# INHIBITORY - DEPRESSION
# =============================================================================
print('-- Inhibitory - Depression')
tm_syn_inib_dep(sim_steps, dt, time, a, b, c, d, vp, vr, spikes[0], I, W, neuron_type)

# =============================================================================
# INHIBITORY - FACILITATION
# =============================================================================
print('-- Inhibitory - Facilitation')
tm_syn_inib_fac(sim_steps, time, dt, a, b, c, d, vp, vr, I, spikes[0], W, neuron_type)