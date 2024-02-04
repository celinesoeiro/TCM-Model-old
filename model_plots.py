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

from tcm_params import TCM_model_parameters
dt = TCM_model_parameters()['dt']
fs = TCM_model_parameters()['sampling_frequency']

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
    
def plot_voltages(n_neurons, voltage, title, neuron_types):
    # if (n_neurons == 40):
    #     fig, axs = plt.subplots(20,2,sharex=True, figsize=(20,20))
    # else:
    #     fig, axs = plt.subplots(50,2,sharex=True, figsize=(40,30))
    n_rows = int(n_neurons/2)
    
    fig, axs = plt.subplots(n_rows, 2, sharex=True, figsize=(n_neurons + 10,n_neurons + 10))
        
    fig.suptitle(title)    
    
    for i in range(n_neurons):
        column = 0
        row = math.floor(i/2)
                
        if (i%2 == 0):
            column = 0
        else:
            column = 1
        
        neuron_type = neuron_types[i]
        
        axs[row,column].set_title(f'NEURONIO {i + 1} - {neuron_type}')
        axs[row,column].plot(voltage[i])
    
    plt.show()
    
def plot_LFP(lfp, chop_till, sim_steps, title):
    new_time= np.transpose(np.arange(len(lfp)))
    
    plt.figure(figsize=(15, 15))
    
    plt.title(title)

    plt.plot(new_time, lfp)
    
    # Set the x-axis label
    plt.xlabel('Time')
    plt.ylabel('LFP')
    
    # Show the plot
    plt.show()
    
def plot_LFPs(LFP_S, LFP_M, LFP_D, LFP_CI, LFP_TC, LFP_TR, chop_till, sim_steps, title):
    new_time= np.transpose(np.arange(len(LFP_S)))
    
    fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3,figsize=(15, 10))
    
    ax1.plot(new_time, LFP_S)
    ax2.plot(new_time, LFP_M)
    ax3.plot(new_time, LFP_D)
    ax4.plot(new_time, LFP_CI)
    ax5.plot(new_time, LFP_TC)
    ax6.plot(new_time, LFP_TR)
    
    ax1.set_title('Layer S')
    ax2.set_title('Layer M')
    ax3.set_title('Layer D')
    ax4.set_title('CI')
    ax5.set_title('TC Nucleus')
    ax6.set_title('TR Nucleus')
    
    fig.suptitle(title)
        
    plt.show()
    
# =============================================================================
# RASTER
# =============================================================================
def layer_raster_plot(n, AP, sim_steps, layer_name, dt):
    fig, ax1 = plt.subplots()
    
    fig.canvas.manager.set_window_title(f'Raster plot layer {layer_name}')

    for i in range(n):
        y_values = np.full_like(AP[i], i + 1)
        ax1.scatter(x=AP[i], y=y_values, color='black', s=1)
        
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                    alpha=0.5)
    
    ax1.set(
        axisbelow=True,  # Hide the grid behind plot objects
        title=f'Raster plot layer {layer_name}',
        xlabel='Time (s)',
        ylabel='Neurons',
    )
    
    x_vec = np.arange(0, sim_steps + 1, int(fs/2))
    x_labels_vec = np.arange(0, n + 1, 1, dtype=int)
     
    ax1.set_ylim(1, n + 1)
    ax1.set_yticks(x_labels_vec)
    ax1.set_yticklabels(x_labels_vec)
    ax1.set_xlim(0, sim_steps)
    ax1.set_xticks(x_vec)
    ax1.set_xticklabels(x_vec)
    plt.show()
    

def plot_raster(
    dbs,
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
    
    fig, ax1 = plt.subplots(figsize=(10, 10))
    fig.canvas.manager.set_window_title(f'Raster plot dbs={dbs}')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)
        
    plt.title(f'Raster plot dbs={dbs}')
    
    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)
    
    ax1.set(
        axisbelow=True,  # Hide the grid behind plot objects
        title=f'Raster plot dbs={dbs}',
        xlabel='Time (s)',
        ylabel='Neurons',
    )
        
    for i in range(n_total):  
        y_values = np.full_like(spikes[i], i + 1)
        ax1.scatter(x=spikes[i], y=y_values, color='black', s=0.5)
        
    ax1.set_ylim(1, n_total + 1)
    ax1.set_yticks([0, 
                   TR_lim, 
                   TC_lim, 
                   CI_lim, 
                   CI_FS_lim, 
                   CI_FS_lim, 
                   D_RS_lim, 
                   D_RS_lim, 
                   D_lim, 
                   M_lim, 
                   S_RS_lim, 
                   S_RS_lim, 
                   S_lim])
    ax1.set_yticklabels(['',
                        'TR',
                        'TC',
                        'CI - FS',
                        'CI - LTS',
                        'CI',
                        'D - RS',
                        'D - IB',
                        'D', 
                        'M - RS', 
                        'S - RS', 
                        'S - IB', 
                        'S',
                        ])
    
    # For dt = 0.1
    multiplier = 1000
    lim_down = chop_till
    lim_up = sim_steps + multiplier*dt
    new_arr = np.arange(lim_down, lim_up, multiplier)
    
    # Transforming flot array to int array
    x_ticks = list(map(int,new_arr/multiplier))
    
    ax1.set_xlim(lim_down, lim_up)
    ax1.set_xticks(new_arr)
    ax1.set_xticklabels(x_ticks)
    
    # TR neurons
    ax1.hlines(y = TR_lim, xmin=0, xmax=sim_steps, color = 'b', linestyle='solid' )
    # TC neurons
    ax1.hlines(y = TC_lim, xmin=0, xmax=sim_steps, color = 'g', linestyle='solid' )
    # CI neurons
    ax1.hlines(y = CI_lim, xmin=0, xmax=sim_steps, color = 'r', linestyle='solid' )
    ax1.hlines(y = CI_FS_lim, xmin=0, xmax=sim_steps, color = 'lightcoral', linestyle='solid')
    # D neurons
    ax1.hlines(y = D_lim, xmin=0, xmax=sim_steps, color = 'c', linestyle='solid' )
    ax1.hlines(y = D_RS_lim, xmin=0, xmax=sim_steps, color = 'paleturquoise', linestyle='solid' )
    # M neurons
    ax1.hlines(y = M_lim, xmin=0, xmax=sim_steps, color = 'm', linestyle='solid' )
    # S neurons
    ax1.hlines(y = S_lim, xmin=0, xmax=sim_steps, color = 'gold', linestyle='solid' )
    ax1.hlines(y = S_RS_lim, xmin=0, xmax=sim_steps, color = 'khaki', linestyle='solid' )
    plt.show()
    
def plot_raster_2(
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
    
    fig, ax1 = plt.subplots(figsize=(10, 10))
    fig.canvas.manager.set_window_title(f'Raster plot')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)
        
    plt.title(f'Raster plot')
    
    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)
    
    ax1.set(
        axisbelow=True,  # Hide the grid behind plot objects
        title=f'Raster plot',
        xlabel='Time (s)',
        ylabel='Neurons',
    )
        
    for i in range(n_total):  
        y_values = np.full_like(spikes[i], i + 1)
        ax1.scatter(x=spikes[i], y=y_values, color='black', s=0.5)
        
    ax1.set_ylim(1, n_total + 1)
    ax1.set_yticks([0, 
                   TR_lim, 
                   TC_lim, 
                   CI_lim, 
                   CI_FS_lim, 
                   CI_FS_lim, 
                   D_RS_lim, 
                   D_RS_lim, 
                   D_lim, 
                   M_lim, 
                   S_RS_lim, 
                   S_RS_lim, 
                   S_lim])
    ax1.set_yticklabels(['',
                        'TR',
                        'TC',
                        'CI - FS',
                        'CI - LTS',
                        'CI',
                        'D - RS',
                        'D - IB',
                        'D', 
                        'M - RS', 
                        'S - RS', 
                        'S - IB', 
                        'S',
                        ])
    
    # For dt = 0.1
    multiplier = 1000
    lim_down = chop_till
    lim_up = sim_steps + multiplier*dt
    new_arr = np.arange(lim_down, lim_up, multiplier)
    
    # Transforming flot array to int array
    x_ticks = list(map(int,new_arr/multiplier))
    
    ax1.set_xlim(lim_down, lim_up)
    ax1.set_xticks(new_arr)
    ax1.set_xticklabels(x_ticks)
    
    # TR neurons
    ax1.hlines(y = TR_lim, xmin=0, xmax=sim_steps, color = 'b', linestyle='solid' )
    # TC neurons
    ax1.hlines(y = TC_lim, xmin=0, xmax=sim_steps, color = 'g', linestyle='solid' )
    # CI neurons
    ax1.hlines(y = CI_lim, xmin=0, xmax=sim_steps, color = 'r', linestyle='solid' )
    ax1.hlines(y = CI_FS_lim, xmin=0, xmax=sim_steps, color = 'lightcoral', linestyle='solid')
    # D neurons
    ax1.hlines(y = D_lim, xmin=0, xmax=sim_steps, color = 'c', linestyle='solid' )
    ax1.hlines(y = D_RS_lim, xmin=0, xmax=sim_steps, color = 'paleturquoise', linestyle='solid' )
    # M neurons
    ax1.hlines(y = M_lim, xmin=0, xmax=sim_steps, color = 'm', linestyle='solid' )
    # S neurons
    ax1.hlines(y = S_lim, xmin=0, xmax=sim_steps, color = 'gold', linestyle='solid' )
    ax1.hlines(y = S_RS_lim, xmin=0, xmax=sim_steps, color = 'khaki', linestyle='solid' )
    plt.show()
    
def plot_raster_comparison(
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
    spike_TR_ON, 
    spike_TC_ON, 
    spike_CI_ON, 
    spike_D_ON, 
    spike_M_ON,
    spike_S_ON,
    spike_TR_OFF, 
    spike_TC_OFF, 
    spike_CI_OFF, 
    spike_D_OFF, 
    spike_M_OFF,
    spike_S_OFF,
    ):
    
    TR_lim = n_TR
    TC_lim = TR_lim + n_TC
    CI_lim = TC_lim + n_CI
    CI_FS_lim = CI_lim - n_CI_LTS
    D_lim = CI_lim + n_D
    D_RS_lim = D_lim - n_D_IB
    M_lim = D_lim + n_M
    S_lim = M_lim + n_S
    S_RS_lim = S_lim - n_S_IB
    
    spikes_ON = np.concatenate([spike_TR_ON, spike_TC_ON, spike_CI_ON, spike_D_ON, spike_M_ON, spike_S_ON])
    spikes_OFF = np.concatenate([spike_TR_OFF, spike_TC_OFF, spike_CI_OFF, spike_D_OFF, spike_M_OFF, spike_S_OFF])
    
    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 18))
    fig.canvas.manager.set_window_title('Raster plots')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)
    
    fig.subplots_adjust(wspace=0.3)
    fig.suptitle('Raster plots')
        
    plt.title('Raster plots')
    
    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)
    
    ax1.set(
        axisbelow=True,  # Hide the grid behind plot objects
        title='Raster plot - DBS OFF',
        xlabel='Time (s)',
        ylabel='Neurons',
    )
    
    ax2.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)
    
    ax2.set(
        axisbelow=True,  # Hide the grid behind plot objects
        title='Raster plot - DBS ON',
        xlabel='Time (s)',
        ylabel='Neurons',
    )
        
    for i in range(n_total):  
        y_values_OFF = np.full_like(spikes_OFF[i], i + 1)
        ax1.scatter(x=spikes_OFF[i], y=y_values_OFF, color='black', s=0.5)
        
        y_values_ON = np.full_like(spikes_OFF[i], i + 1)
        ax2.scatter(x=spikes_ON[i], y=y_values_ON, color='black', s=0.5)
        
    ax1.set_ylim(1, n_total + 1)
    ax2.set_ylim(1, n_total + 1)
    
    ax1.set_yticks([0, 
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
    ax1.set_yticklabels(['',
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
    ax2.set_yticks([0, 
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
    ax2.set_yticklabels(['',
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
    
    # For dt = 0.1
    if (dt == 0.1):
        multiplier = 1000
        divider = multiplier
    elif (dt == 0.5):
        multiplier = 200
        divider = multiplier*dt
        
    lim_down = chop_till
    lim_up = sim_steps + multiplier*dt
    new_arr = np.arange(lim_down, lim_up, multiplier)
    
    # Transforming flot array to int array
    x_ticks = list(map(int,new_arr/divider))
    
    ax1.set_xlim(lim_down, lim_up)
    ax1.set_xticks(new_arr)
    ax1.set_xticklabels(x_ticks)
    
    ax2.set_xlim(lim_down, lim_up)
    ax2.set_xticks(new_arr)
    ax2.set_xticklabels(x_ticks)
    
    # TR neurons
    ax1.hlines(y = TR_lim, xmin=0, xmax=sim_steps, color = 'b', linestyle='solid' )
    # TC neurons
    ax1.hlines(y = TC_lim, xmin=0, xmax=sim_steps, color = 'g', linestyle='solid' )
    # CI neurons
    ax1.hlines(y = CI_lim, xmin=0, xmax=sim_steps, color = 'r', linestyle='solid' )
    ax1.hlines(y = CI_FS_lim, xmin=0, xmax=sim_steps, color = 'lightcoral', linestyle='solid')
    # D neurons
    ax1.hlines(y = D_lim, xmin=0, xmax=sim_steps, color = 'c', linestyle='solid' )
    ax1.hlines(y = D_RS_lim, xmin=0, xmax=sim_steps, color = 'paleturquoise', linestyle='solid' )
    # M neurons
    ax1.hlines(y = M_lim, xmin=0, xmax=sim_steps, color = 'm', linestyle='solid' )
    # S neurons
    ax1.hlines(y = S_lim, xmin=0, xmax=sim_steps, color = 'gold', linestyle='solid' )
    ax1.hlines(y = S_RS_lim, xmin=0, xmax=sim_steps, color = 'khaki', linestyle='solid' )
    
    # TR neurons
    ax2.hlines(y = TR_lim, xmin=0, xmax=sim_steps, color = 'b', linestyle='solid' )
    # TC neurons
    ax2.hlines(y = TC_lim, xmin=0, xmax=sim_steps, color = 'g', linestyle='solid' )
    # CI neurons
    ax2.hlines(y = CI_lim, xmin=0, xmax=sim_steps, color = 'r', linestyle='solid' )
    ax2.hlines(y = CI_FS_lim, xmin=0, xmax=sim_steps, color = 'lightcoral', linestyle='solid')
    # D neurons
    ax2.hlines(y = D_lim, xmin=0, xmax=sim_steps, color = 'c', linestyle='solid' )
    ax2.hlines(y = D_RS_lim, xmin=0, xmax=sim_steps, color = 'paleturquoise', linestyle='solid' )
    # M neurons
    ax2.hlines(y = M_lim, xmin=0, xmax=sim_steps, color = 'm', linestyle='solid' )
    # S neurons
    ax2.hlines(y = S_lim, xmin=0, xmax=sim_steps, color = 'gold', linestyle='solid' )
    ax2.hlines(y = S_RS_lim, xmin=0, xmax=sim_steps, color = 'khaki', linestyle='solid' )
    
    plt.show()