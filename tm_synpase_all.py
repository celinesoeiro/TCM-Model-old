#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 20:31:42 2024

@author: celinesoeiro
"""

import numpy as np

from model_functions import homogeneous_poisson, tm_syn_excit_dep, tm_syn_excit_fac, tm_syn_inib_dep, tm_syn_inib_fac

ms = 1000           # 1ms
rate = 20 * 1/ms    # spike rate 
bin_size = 1        # bin size 
tmax = 1 * ms       # the total lenght of the spike train
dt = rate

# Izhikevich neuron model
vp = 30     # voltage peak
vr = -65    # voltage threshold

W = 1e2

random_factor = np.random.rand()

neuron_type_rs = 'Disparo Regular (RS)'
a_rs = 0.02
b_rs = 0.2
c_rs = -65 + 15*random_factor**2
d_rs = 8 - 0.6*random_factor**2
I_rs = 10

neuron_type_ib = "Disparo em Rajada (IB)"
a_ib = 0.02 
b_ib = 0.2 
c_ib = -55 + 15*random_factor**2
d_ib = 4 - 0.6*random_factor**2
I_ib = 10

neuron_type_fs = "Disparo Rapido (FS)"
a_fs = 0.1 + 0.008*random_factor
b_fs = 0.2 - 0.005*random_factor
c_fs = -65 
d_fs = 2 
I_fs = 10

neuron_type_lts = "Baixo Limiar de Disparo (LTS)"
a_lts = 0.02 + 0.008*random_factor
b_lts = 0.25 - 0.005*random_factor
c_lts = -65 
d_lts = 2 
I_lts = 10

neuron_type_tc = "Rele Talamo-cortical (TC)"
a_tc = 0.02 + 0.008*random_factor
b_tc = 0.25 - 0.005*random_factor
c_tc = -65 
d_tc = 0.05 
I_tc = 10

neuron_type_tr = "Talamo Reticular (TR)"
a_tr = 0.02 + 0.008*random_factor
b_tr = 0.25 - 0.005*random_factor
c_tr = -65 
d_tr = 2.05 
I_tr = 10

spikes = homogeneous_poisson(rate, tmax, bin_size) 
time = np.arange(1,len(spikes)) * bin_size 
sim_steps = len(spikes)

# =============================================================================
# REGULAR SPIKING
# =============================================================================
print('-- Excitatory - Depression - RS')
tm_syn_excit_dep(sim_steps=sim_steps, vr=vr, vp=vp, a=a_rs, b=b_rs, c=c_rs, d=d_rs, spikes=spikes, time=time, I=I_rs, W=W, dt=dt, neuron_type=neuron_type_rs)

print('-- Excitatory - Facilitation - RS')
tm_syn_excit_fac(sim_steps=sim_steps, vr=vr, vp=vp, a=a_rs, b=b_rs, c=c_rs, d=d_rs, spikes=spikes, time=time, I=I_rs, W=W, dt=dt, neuron_type=neuron_type_rs)

print('-- Inhibitory - Depression - RS')
tm_syn_inib_dep(sim_steps=sim_steps, vr=vr, vp=vp, a=a_rs, b=b_rs, c=c_rs, d=d_rs, spikes=spikes, time=time, I=I_rs, W=W, dt=dt, neuron_type=neuron_type_rs)

print('-- Inhibitory - Facilitation - RS')
tm_syn_inib_fac(sim_steps=sim_steps, vr=vr, vp=vp, a=a_rs, b=b_rs, c=c_rs, d=d_rs, spikes=spikes, time=time, I=I_rs, W=W, dt=dt, neuron_type=neuron_type_rs)

# =============================================================================
# INTRINSICALLY BURSTING
# =============================================================================
print('-- Excitatory - Depression - IB')
tm_syn_excit_dep(sim_steps=sim_steps, vr=vr, vp=vp, a=a_ib, b=b_ib, c=c_ib, d=d_ib, spikes=spikes, time=time, I=I_ib, W=W, dt=dt, neuron_type=neuron_type_ib)

print('-- Excitatory - Facilitation - IB')
tm_syn_excit_fac(sim_steps=sim_steps, vr=vr, vp=vp, a=a_ib, b=b_ib, c=c_ib, d=d_ib, spikes=spikes, time=time, I=I_ib, W=W, dt=dt, neuron_type=neuron_type_ib)

print('-- Inhibitory - Depression - IB')
tm_syn_inib_dep(sim_steps=sim_steps, vr=vr, vp=vp, a=a_ib, b=b_ib, c=c_ib, d=d_ib, spikes=spikes, time=time, I=I_ib, W=W, dt=dt, neuron_type=neuron_type_ib)

print('-- Inhibitory - Facilitation - IB')
tm_syn_inib_fac(sim_steps=sim_steps, vr=vr, vp=vp, a=a_ib, b=b_ib, c=c_ib, d=d_ib, spikes=spikes, time=time, I=I_ib, W=W, dt=dt, neuron_type=neuron_type_ib)

# =============================================================================
# FAST SPIKING
# =============================================================================
print('-- Excitatory - Depression - FS')
tm_syn_excit_dep(sim_steps=sim_steps, vr=vr, vp=vp, a=a_fs, b=b_fs, c=c_fs, d=d_fs, spikes=spikes, time=time, I=I_fs, W=W, dt=dt, neuron_type=neuron_type_fs)

print('-- Excitatory - Facilitation - FS')
tm_syn_excit_fac(sim_steps=sim_steps, vr=vr, vp=vp, a=a_fs, b=b_fs, c=c_fs, d=d_fs, spikes=spikes, time=time, I=I_fs, W=W, dt=dt, neuron_type=neuron_type_fs)

print('-- Inhibitory - Depression - FS')
tm_syn_inib_dep(sim_steps=sim_steps, vr=vr, vp=vp, a=a_fs, b=b_fs, c=c_fs, d=d_fs, spikes=spikes, time=time, I=I_fs, W=W, dt=dt, neuron_type=neuron_type_fs)

print('-- Inhibitory - Facilitation - FS')
tm_syn_inib_fac(sim_steps=sim_steps, vr=vr, vp=vp, a=a_fs, b=b_fs, c=c_fs, d=d_fs, spikes=spikes, time=time, I=I_fs, W=W, dt=dt, neuron_type=neuron_type_fs)

# =============================================================================
# LOW THRESHOLD SPIKING
# =============================================================================
print('-- Excitatory - Depression - LTS')
tm_syn_excit_dep(sim_steps=sim_steps, vr=vr, vp=vp, a=a_lts, b=b_lts, c=c_lts, d=d_lts, spikes=spikes, time=time, I=I_lts, W=W, dt=dt, neuron_type=neuron_type_lts)

print('-- Excitatory - Facilitation - LTS')
tm_syn_excit_fac(sim_steps=sim_steps, vr=vr, vp=vp, a=a_lts, b=b_lts, c=c_lts, d=d_lts, spikes=spikes, time=time, I=I_lts, W=W, dt=dt, neuron_type=neuron_type_lts)

print('-- Inhibitory - Depression - LTS')
tm_syn_inib_dep(sim_steps=sim_steps, vr=vr, vp=vp, a=a_lts, b=b_lts, c=c_lts, d=d_lts, spikes=spikes, time=time, I=I_lts, W=W, dt=dt, neuron_type=neuron_type_lts)

print('-- Inhibitory - Facilitation - LTS')
tm_syn_inib_fac(sim_steps=sim_steps, vr=vr, vp=vp, a=a_lts, b=b_lts, c=c_lts, d=d_lts, spikes=spikes, time=time, I=I_lts, W=W, dt=dt, neuron_type=neuron_type_lts)

# =============================================================================
# THALAMO CORTICAL
# =============================================================================
print('-- Excitatory - Depression - TC')
tm_syn_excit_dep(sim_steps=sim_steps, vr=vr, vp=vp, a=a_tc, b=b_tc, c=c_tc, d=d_tc, spikes=spikes, time=time, I=I_tc, W=W, dt=dt, neuron_type=neuron_type_tc)

print('-- Excitatory - Facilitation - TC')
tm_syn_excit_fac(sim_steps=sim_steps, vr=vr, vp=vp, a=a_tc, b=b_tc, c=c_tc, d=d_tc, spikes=spikes, time=time, I=I_tc, W=W, dt=dt, neuron_type=neuron_type_tc)

print('-- Inhibitory - Depression - TC')
tm_syn_inib_dep(sim_steps=sim_steps, vr=vr, vp=vp, a=a_tc, b=b_tc, c=c_tc, d=d_tc, spikes=spikes, time=time, I=I_tc, W=W, dt=dt, neuron_type=neuron_type_tc)

print('-- Inhibitory - Facilitation - TC')
tm_syn_inib_fac(sim_steps=sim_steps, vr=vr, vp=vp, a=a_tc, b=b_tc, c=c_tc, d=d_tc, spikes=spikes, time=time, I=I_tc, W=W, dt=dt, neuron_type=neuron_type_tc)

# =============================================================================
# THALAMIC RETICULAR
# =============================================================================
print('-- Excitatory - Depression - TR')
tm_syn_excit_dep(sim_steps=sim_steps, vr=vr, vp=vp, a=a_tr, b=b_tr, c=c_tr, d=d_tr, spikes=spikes, time=time, I=I_tr, W=W, dt=dt, neuron_type=neuron_type_tr)

print('-- Excitatory - Facilitation - TR')
tm_syn_excit_fac(sim_steps=sim_steps, vr=vr, vp=vp, a=a_tr, b=b_tr, c=c_tr, d=d_tr, spikes=spikes, time=time, I=I_tr, W=W, dt=dt, neuron_type=neuron_type_tr)

print('-- Inhibitory - Depression - TR')
tm_syn_inib_dep(sim_steps=sim_steps, vr=vr, vp=vp, a=a_tr, b=b_tr, c=c_tr, d=d_tr, spikes=spikes, time=time, I=I_tr, W=W, dt=dt, neuron_type=neuron_type_tr)

print('-- Inhibitory - Facilitation - TR')
tm_syn_inib_fac(sim_steps=sim_steps, vr=vr, vp=vp, a=a_tr, b=b_tr, c=c_tr, d=d_tr, spikes=spikes, time=time, I=I_tr, W=W, dt=dt, neuron_type=neuron_type_tr)



