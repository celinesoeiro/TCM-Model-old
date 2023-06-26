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
        sim_time,
        dt,
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
    
    TR_lim = n_TR
    TC_lim = TR_lim + n_TC
    CI_lim = TC_lim + n_CI
    CI_FS_lim = CI_lim - n_CI_LTS
    D_lim = CI_lim + n_D
    D_RS_lim = D_lim - n_D_IB
    M_lim = D_lim + n_M
    S_lim = M_lim + n_S
    S_RS_lim = S_lim - n_S_IB
    
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
    
    plt.figure(figsize=(10, 10))
    plt.title('raster plot')
    
    ax = plt.axes()
    
    for i in range(n_total):  
        y_values = np.full_like(spikes[i], i + 1)
        ax.scatter(x=spikes[i], y=y_values, color='black', s=0.5)
        
    TR_lim = n_TR
    TC_lim = TR_lim + n_TC
    CI_lim = TC_lim + n_CI
    CI_FS_lim = CI_lim - n_CI_LTS
    D_lim = CI_lim + n_D
    D_RS_lim = D_lim - n_D_IB
    M_lim = D_lim + n_M
    S_lim = M_lim + n_S
    S_RS_lim = S_lim - n_S_IB
        
    ax.set_ylabel('neurons')
    ax.set_ylim(1, n_total + 1)
    ax.set_yticks([0, 
                   TR_lim, 
                   TC_lim, 
                   CI_lim, 
                   CI_FS_lim - 20, 
                   CI_FS_lim + 20, 
                   D_RS_lim - 20, 
                   D_RS_lim + 10, 
                   D_lim, 
                   M_lim, 
                   S_RS_lim - 20, 
                   S_RS_lim + 20, 
                   S_lim])
    ax.set_yticklabels(['',
                        'TR',
                        'TC',
                        'CI - FS',
                        'CI - LTS',
                        'CI',
                        'D - RS',
                        'D - IB',
                        'D', 
                        'M', 
                        'S - RS', 
                        'S - IB', 
                        'S',
                        ])
    
    multiplier = 50000
    ax.set_xlabel('Time (s)')
    ax.set_xlim(left = chop_till - 1, right=sim_steps + 1)
    ax.set_xticks(np.arange(chop_till, sim_steps + dt*multiplier, int(dt*multiplier)))
    ax.set_xticklabels(np.arange(1, sim_time + dt*15, dt*5))
    
    # TR neurons
    ax.hlines(y = TR_lim, xmin=0, xmax=sim_steps, color = 'b', linestyle='solid' )
    # TC neurons
    ax.hlines(y = TC_lim, xmin=0, xmax=sim_steps, color = 'g', linestyle='solid' )
    # CI neurons
    ax.hlines(y = CI_lim, xmin=0, xmax=sim_steps, color = 'r', linestyle='solid' )
    ax.hlines(y = CI_FS_lim, xmin=0, xmax=sim_steps, color = 'lightcoral', linestyle='solid')
    # D neurons
    ax.hlines(y = D_lim, xmin=0, xmax=sim_steps, color = 'c', linestyle='solid' )
    ax.hlines(y = D_RS_lim, xmin=0, xmax=sim_steps, color = 'paleturquoise', linestyle='solid' )
    # M neurons
    ax.hlines(y = M_lim, xmin=0, xmax=sim_steps, color = 'm', linestyle='solid' )
    # S neurons
    ax.hlines(y = S_lim, xmin=0, xmax=sim_steps, color = 'gold', linestyle='solid' )
    ax.hlines(y = S_RS_lim, xmin=0, xmax=sim_steps, color = 'khaki', linestyle='solid' )
    plt.show()
    