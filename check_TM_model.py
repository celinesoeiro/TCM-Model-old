#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 21:42:52 2024

@author: celinesoeiro
"""
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

from model_functions import tm_synapse_eq
from tcm_params import TCM_model_parameters

ms = 1000           # 1ms
rate = 10 * 1/ms    # spike rate 
bin_size = 1        # bin size 
tmax = 1 * ms       # the total lenght of the spike train

syn_params = TCM_model_parameters()['synapse_params_excitatory']

t_f_E = syn_params['t_f']
t_d_E = syn_params['t_d']
t_s_E = syn_params['t_s']
U_E = syn_params['U']
A_E = syn_params['distribution']

t_f_I = syn_params['t_f']
t_d_I = syn_params['t_d']
t_s_I = syn_params['t_s']
U_I = syn_params['U']
A_I = syn_params['distribution']

def homogeneous_poisson(rate, tmax, bin_size): 
    nbins = np.floor(tmax/bin_size).astype(int) 
    prob_of_spike = rate * bin_size 
    spikes = np.random.rand(nbins) < prob_of_spike 
    return spikes * 1

spikes_poisson = homogeneous_poisson(rate, tmax, bin_size) 
time = np.arange(len(spikes_poisson)) * bin_size 

u = np.zeros((1, 3))
R = np.ones((1, 3))
I = np.zeros((1, 3))

PSC_E = np.zeros((1, len(time)))
# Synapse - Within layer  

for t in time:
    AP = spikes_poisson[t]
    
    syn = tm_synapse_eq(u = u, 
                          R = R, 
                          I = I, 
                          AP = AP, 
                          t_f = t_f_E, 
                          t_d = t_d_E, 
                          t_s = t_s_E, 
                          U = U_E, 
                          A = A_E, 
                          dt = rate, 
                          p = 3)
    
    PSC_E[0][t] = 1*syn['Ipost']
    
del u, R, I, syn, t

u = np.zeros((1, 3))
R = np.ones((1, 3))
I = np.zeros((1, 3))

PSC_I = np.zeros((1, len(time)))
# Synapse - Within layer  

for t in time:
    AP = spikes_poisson[t]
    
    syn = tm_synapse_eq(u = u, 
                          R = R, 
                          I = I, 
                          AP = AP, 
                          t_f = t_f_I, 
                          t_d = t_d_I, 
                          t_s = t_s_I, 
                          U = U_I, 
                          A = A_I, 
                          dt = rate, 
                          p = 3)
    
    PSC_I[0][t] = 1*syn['Ipost']


fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(15, 10))
ax1.plot(time, spikes_poisson)
ax2.plot(time,PSC_E[0])
ax3.plot(time,PSC_I[0])

ax1.set_title('Trem de pulsos gerado por Poisson')
ax2.set_title('PSC - Excitatoria')
ax3.set_title('PSC - Inibitoria')







    