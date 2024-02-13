#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 16:17:53 2024

@author: celinesoeiro
"""

import numpy as np
from random import seed, random

seed(1)
random_factor = random()

from model_functions import homogeneous_poisson, tm_syn_excit_dep, tm_syn_excit_fac, tm_syn_inib_dep, tm_syn_inib_fac

ms = 1000           # 1ms
rate = 20 * 1/ms    # spike rate 
bin_size = 1        # bin size 
tmax = 1 * ms       # the total lenght of the spike train
dt = rate

# Izhikevich neuron model
vp = 30     # voltage peak
vr = -65    # voltage threshold

neuron_type = "Disparo em Rajada (IB)"

a = 0.02 
b = 0.2 
c = -55 + 15*random_factor**2
d = 4 - 0.6*random_factor**2

I = 4.1*4

W = 1
# =============================================================================
# Poisson spike gen
# =============================================================================
spikes = homogeneous_poisson(rate, tmax, bin_size) 
time = np.arange(1,len(spikes)) * bin_size 
sim_steps = len(spikes)

# =============================================================================
# EXCITATORY - DEPRESSION
# =============================================================================
tm_syn_excit_dep(sim_steps, vr, vp, a, b, c, d, spikes, time, I, W, dt, neuron_type)

# =============================================================================
# EXCITATORY - FACILITATION
# =============================================================================
tm_syn_excit_fac(sim_steps, vr, vp, a, b, c, d, time, dt, spikes, I, W, neuron_type)

# =============================================================================
# INHIBITORY - DEPRESSION
# =============================================================================
tm_syn_inib_dep(sim_steps, dt, time, a, b, c, d, vp, vr, spikes, I, W, neuron_type)

# =============================================================================
# INHIBITORY - FACILITATION
# =============================================================================
tm_syn_inib_fac(sim_steps, time, dt, a, b, c, d, vp, vr, I, spikes, W, neuron_type)


