# -*- coding: utf-8 -*-
"""
Created on Sat May 13 10:26:45 2023

@author: Avell
"""

import math
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

def plot_heat_map(matrix_normal, matrix_PD): 
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(17,7))
    
    fig.subplots_adjust(wspace=0.3)
    fig.suptitle('Matriz de conexão')
    
    sns.heatmap(matrix_normal, 
                vmin=-1, vmax=1, 
                yticklabels=['S', 'M', 'D', 'CI', 'TC', 'TR'], 
                annot=True, 
                fmt=".3f", 
                linewidth=.75,
                cmap=sns.color_palette("coolwarm", as_cmap=True),
                ax=ax1,
                )
    ax1.set(xlabel="", ylabel="")
    ax1.xaxis.tick_top()
    ax1.set_title('Condição normal')
    
    sns.heatmap(matrix_PD, 
                vmin=-1, vmax=1, 
                yticklabels=['S', 'M', 'D', 'CI', 'TC', 'TR'], 
                annot=True, 
                fmt=".3f", 
                linewidth=.75,
                cmap=sns.color_palette("coolwarm", as_cmap=True),
                ax=ax2,
                )
    ax2.set(xlabel="", ylabel="")
    ax2.xaxis.tick_top()
    ax2.set_title('Condição parkinsoniana')
    
    plt.show()
    
def plot_voltages(n_neurons, voltage, chop_till, sim_steps, title):
    new_time= np.transpose(np.arange(len(voltage)))

    if (n_neurons == 40):
        fig, axs = plt.subplots(20,2,sharex=True, figsize=(20,20))
    else:
        fig, axs = plt.subplots(50,2,sharex=True, figsize=(40,30))
        
    fig.suptitle(title)    
    
    for i in range(n_neurons):
        column = 0
        row = math.floor(i/2)
                
        if (i%2 == 0):
            column = 0
        else:
            column = 1
        
        axs[row,column].set_title(f'NEURONIO {i + 1}')
        axs[row,column].plot(new_time, voltage[:, i])
            
    plt.show()

# =============================================================================
# RASTER
# =============================================================================
def plot_rasters(neuron_dict, title):
    # Create a new figure
    plt.figure(figsize=(10, 16))
    plt.title(title)
    
    # Iterate over each neuron
    for neuron_idx, spike_times in enumerate(neuron_dict.values()):
        # Generate y-values for scatter plot
        y_values = np.full_like(spike_times, neuron_idx + 1)
        
        # Plot scatter points for spike times
        plt.scatter(spike_times, y_values, marker='o', color='black')
    
    # Set the y-axis limits and labels
    plt.ylim(0.5, len(neuron_dict) + 0.5)
    plt.yticks(range(1, len(neuron_dict) + 1), neuron_dict.keys())
    
    # Set the x-axis label
    plt.xlabel('Time')
    
    # Show the plot
    plt.show()
    
def plot_raster(
        sim_steps,
        chop_till, 
        n_TR, 
        n_TC, 
        n_CI, 
        n_D, 
        n_M, 
        n_S, 
        n_total,
        n_CI_FS,
        n_CI_LTS,
        n_D_RS,
        n_D_IB,
        n_S_RS,
        n_S_IB,
        spike_times_TR, 
        spike_times_TC, 
        spike_times_CI, 
        spike_times_D, 
        spike_times_M,
        spike_times_S):
    
    spike_TR_clean = np.zeros((n_TR, sim_steps - chop_till))
    spike_TC_clean = np.zeros((n_TC, sim_steps - chop_till))
    spike_CI_clean = np.zeros((n_CI, sim_steps - chop_till))
    spike_D_clean = np.zeros((n_D, sim_steps - chop_till))
    spike_M_clean = np.zeros((n_M, sim_steps - chop_till))
    spike_S_clean = np.zeros((n_S, sim_steps - chop_till))
    
    for i in range(n_TR):
        spike_TR_clean[i] = spike_times_TR[i][chop_till:]
        
    for i in range(n_TC):
        spike_TC_clean[i] = spike_times_TC[i][chop_till:]
        spike_CI_clean[i] = spike_times_CI[i][chop_till:]
        spike_D_clean[i] = spike_times_D[i][chop_till:]
        spike_M_clean[i] = spike_times_M[i][chop_till:]
        spike_S_clean[i] = spike_times_S[i][chop_till:]
    
    spikes = np.concatenate([spike_TR_clean, spike_TC_clean, spike_CI_clean, spike_D_clean, spike_M_clean, spike_S_clean])
    
    plt.figure(figsize=(8, 8))
    plt.title('raster plot')
    
    for i in range(n_total):  
        y_values = np.full_like(spikes[i], i + 1)
        plt.scatter(x=spikes[i], y=y_values, color='black', s=0.5)
        
    plt.ylim(1, n_total + 1)
    plt.yticks(np.arange(0, n_total + 10, 20))
    plt.xlim(left = chop_till - 1, right=sim_steps + 1)
    plt.xticks(np.arange(2000, sim_steps + 1, 1000))
    
    # TR neurons
    plt.axhline(y = n_TR, color = 'b', linestyle='-' )
    # TC neurons
    plt.axhline(y = n_TR + n_TC, color = 'g', linestyle='-' )
    # CI neurons
    plt.axhline(y = n_TR + n_TC + n_CI, color = 'r', linestyle='-' )
    plt.axhline(y = n_TR + n_TC + n_CI + n_CI_FS, color = 'lightcoral', linestyle='-')
    plt.axhline(y = n_TR + n_TC + n_CI + n_CI_FS + n_CI_LTS, color = 'lightcoral', linestyle='-')
    # D neurons
    plt.axhline(y = n_TR + n_TC + n_CI + n_D, color = 'c', linestyle='-' )
    plt.axhline(y = n_TR + n_TC + n_CI + n_D + n_D_RS, color = 'paleturquoise', linestyle='-' )
    plt.axhline(y = n_TR + n_TC + n_CI + n_D + n_D_RS + n_D_IB, color = 'paleturquoise', linestyle='-' )
    # M neurons
    plt.axhline(y = n_TR + n_TC + n_CI + n_D + n_M, color = 'm', linestyle='-' )
    # S neurons
    plt.axhline(y = n_TR + n_TC + n_CI + n_D + n_M + n_S, color = 'gold', linestyle='-' )
    plt.axhline(y = n_TR + n_TC + n_CI + n_D + n_M + n_S + n_S_RS, color = 'khaki', linestyle='-' )
    plt.axhline(y = n_TR + n_TC + n_CI + n_D + n_M + n_S + n_S_RS + n_S_IB, color = 'khaki', linestyle='-' )
    
    plt.ylabel('neurons')
    plt.xlabel('Time')
        
    plt.show()
    