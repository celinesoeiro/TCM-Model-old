"""
@description: Functions used to create the cortex model
@author: celinesoeiro
"""

import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

from scipy.signal import welch

def poisson_spike_generator(num_steps, dt, num_neurons, thalamic_firing_rate, current_value=None):
    # Initialize an array to store spike times for each neuron
    spike_times = [[] for _ in range(num_neurons)]

    # Calculate firing probability
    firing_prob = thalamic_firing_rate * dt  # Calculate firing probability

    # Generate spikes for each neuron using the Poisson distribution
    for t in range(num_steps):
        for neuron_id in range(num_neurons):
            # Generate a random number between 0 and 1
            rand_num = np.random.rand()
            
            # If the random number is less than the firing probability, spike
            if rand_num < firing_prob:
                spike_times[neuron_id].append(t)
            else: 
                spike_times[neuron_id].append(0)
    
    # Creating a vector to be used as current input
    input_current = np.zeros((1, num_steps))
    for sub_spike in spike_times:
        for spike in sub_spike:
            spike_indice = np.array(spike)
            value = np.random.normal(loc=0.25, scale=0.05)
            input_current[0][spike_indice.astype(int)] = value
                
    return spike_times, input_current

def izhikevich_dvdt(v, u, I):
    return 0.04*v**2 + 5*v + 140 - u + I

def izhikevich_dudt(v, u, a, b):
    return a*(b*v - u)

def tm_synapse_eq(r, x, Is, AP, tau_f, tau_d, tau_s, U, A, dt):        
    for p in range(0, 3):
        # Solve EDOs using Euler method
        r[0][p] = r[0][p] + dt*(-r[0][p]/tau_f[p] + U[p]*(1 - r[0][p])*AP)
        x[0][p] = x[0][p] + dt*((1 - x[0][p])/tau_d[p - 1] - (r[0][p] + U[p]*(1 - r[0][p]))*x[0][p]*AP)
        Is[0][p] = Is[0][p] + dt*(-Is[0][p]/tau_s + A[p]*x[0][p]*(r[0][p] + U[p]*(1 - r[0][p]))*AP)
        
    Ipost = np.sum(Is)
    
    tm_syn_inst = dict()
    tm_syn_inst['r'] = r
    tm_syn_inst['x'] = x
    tm_syn_inst['Is'] = Is
    tm_syn_inst['Ipost'] = np.around(Ipost, decimals=6)
        
    return tm_syn_inst

def get_frequency(signal, sim_time):
    # Couting non zero values in the signal per second
    return print('Signal frequency = ',int(np.count_nonzero(signal)/sim_time))
    
def plot_raster(title, spike_times, sim_time, dt, num_neurons):
    # plt.figure(figsize=(10, 6))

    for neuron_id, times in enumerate(spike_times):
        spikes = []
        for item in times:
            spikes.append(item*dt)
            
        plt.scatter(spikes, [neuron_id] * len(times), c='k', marker='|', linewidths=0.75)

    plt.xlabel('Time (s)')
    plt.ylabel('Neuron')
    plt.title(title)
    plt.ylim(-0.5, num_neurons)
    plt.xlim(0, sim_time + 0.25)
    plt.tight_layout()
    plt.show()

def plot_voltage(title, y, dt, sim_time):
    # plt.figure(figsize=(15, 15))
    
    x = np.arange(0,sim_time, dt)

    plt.title(title)

    plt.plot(x, y)

    # Set the x-axis label
    plt.xlabel('Time')
    plt.ylabel('Voltage')

    # Show the plot
    plt.show()
    
def plot_psd_welch(title, signal, frequency):
    frequencie, psd = welch(signal, fs = frequency,  nperseg=1024)

    # Create a plot
    # plt.figure(figsize=(8, 4))
    plt.semilogy(frequencie.reshape(1, len(frequencie)), psd)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.title(title)
    plt.grid(True)
    plt.show()
    
def plot_heat_map(matrix_normal, labels): 
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(17,7))
    
    fig.subplots_adjust(wspace=0.3)
    fig.suptitle('Matriz de conexão')
    
    sns.heatmap(matrix_normal, 
                vmin=-1, vmax=1, 
                yticklabels=labels, 
                annot=True, 
                fmt=".3f", 
                linewidth=.75,
                cmap=sns.color_palette("coolwarm", as_cmap=True),
                ax=ax1,
                )
    ax1.set(xlabel="", ylabel="")
    ax1.xaxis.tick_top()
    ax1.set_title('Condição normal')
    
    plt.show()
    
def plot_raster_cortex(
    dbs,
    sim_steps,
    dt,
    chop_till, 
    n_TR, 
    n_TC, 
    n_CI, 
    n_D, 
    n_M, 
    n_S, 
    n_total,
    n_CI_LTS,
    n_D_IB,
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
                   D_RS_lim , 
                   D_lim, 
                   M_lim, 
                   S_RS_lim , 
                   S_lim])
    ax1.set_yticklabels(['',
                        'TR',
                        'TC',
                        'CI - FS',
                        'CI - LTS',
                        'D - RS',
                        'D - IB',
                        'M - RS', 
                        'S - RS', 
                        'S - IB', 
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
