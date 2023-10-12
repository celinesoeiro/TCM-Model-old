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
    

