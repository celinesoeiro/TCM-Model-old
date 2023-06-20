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