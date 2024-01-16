#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 21:08:50 2023

@author: celinesoeiro
"""
ms = 1000

dt = 1/ms
sim_time = 1
num_steps = int(sim_time/dt)

# Depressing
tau_mem = 40 #msec
tau_ina = 3 #msec
A_SE = 250 #pA
tau_rec = 800 #msec
U_SE = 0.5

# Facilitating
tau_mem = 40 #msec
tau_ina = 1.5 #msec
A_SE = 1540 #pA
tau_rec = 130 #msec
tau_fac = 530 #msec
U_SE = 0.03

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

# =============================================================================
# Functions
# =============================================================================

def plot_voltage(title, y, dt, sim_time):    
    x = np.arange(0,sim_time, dt)

    plt.title(title)
    plt.plot(x, y)

    plt.xlabel('Time')
    plt.ylabel('Voltage')

    plt.show()


def poisson_spike_generator(rate, T, dt):
    """
    Generate a Poisson spike train.

    Parameters:
    - rate: firing rate (spikes per second)
    - T: total simulation time
    - dt: time step

    Returns:
    - spike_train: binary array representing the spike train
    """
    num_steps = int(T / dt)
    spike_train = np.random.rand(num_steps) < rate * dt
    
    return spike_train

def resources_recovered(z, Use, x, AP):
    return z/tau_rec - Use*x*AP

def resources_active(y, Use, x, AP):
    return -y/tau_ina + Use*x*AP

def resources_inactive(y, z):
    return y/tau_ina - z/tau_rec

def utilization_factor(Use, AP):
    return -Use/tau_fac + U_SE*(1 - Use)*AP

# =============================================================================
# Run the model
# =============================================================================

x = np.zeros(num_steps)
y = np.zeros(num_steps)
z = np.zeros(num_steps)
Use = np.zeros(num_steps)
PSC = np.zeros(num_steps)
v = np.zeros(num_steps)

R = 100 # MOhm

spikes = poisson_spike_generator(
    rate = 40,
    T = sim_time,
    dt = dt, 
    )

for t in range(1, num_steps):
    x_aux = x[t - 1]
    y_aux = y[t - 1]
    z_aux = z[t - 1]
    Use_aux = Use[t - 1]
    AP_aux = spikes[t - 1]
    AP = 0
    
    if (AP_aux == True):
        AP = 1

    dx = resources_recovered(z_aux, Use_aux, x_aux, AP)
    dy = resources_active(y_aux, Use_aux, x_aux, AP)
    dz = resources_inactive(y_aux, z_aux)
    dUse = utilization_factor(Use_aux, AP)
    
    x[t] = x_aux + dx*dt
    y[t] = y_aux + dy*dt
    z[t] = z_aux + dz*dt
    Use[t] = Use_aux + dUse*dt
    PSC[t] = A_SE*y[t]
    
    dv = -v[t - 1] + R*PSC[t]
    v[t] = v[t - 1] + dt*dv

plot_voltage('PSC', PSC, dt, sim_time)
plot_voltage('v', v, dt, sim_time)
    
# =============================================================================
# ChatGPT
# =============================================================================
def tsodyks_markram_synapse(I, U, tau_rec, tau_facil, dt, T):
    """
    Simulate Tsodyks-Markram synapse model.

    Parameters:
    - I: input spike train (binary array)
    - U: utilization of synaptic efficacy
    - tau_rec: recovery time constant
    - tau_facil: facilitation time constant
    - dt: time step
    - T: total simulation time

    Returns:
    - t: time array
    - s: synaptic efficacy over time
    """
    num_steps = int(T / dt)
    t = np.linspace(0, T, num_steps)

    s = np.zeros(num_steps)
    u = U * np.ones(num_steps)
    x = np.zeros(num_steps)

    for i in range(1, num_steps):
        # dx = dt / tau_rec * (1 - x[i-1]) - dt * u[i-1] * x[i-1] * I[i-1]
        # du = dt / tau_facil * (U - u[i-1]) + dt * U * (1 - u[i-1]) * I[i-1]
        Use1 = u[i-1]*(1-U)+U
        dx = dt*( 1/tau_rec * (1 - x[i-1]) - Use1 * x[i-1] * I[i-1])
        du = dt*(-1/tau_facil * u[i-1] + U * (1 - u[i-1]) * I[i-1])

        x[i] = x[i-1] + dx
        u[i] = u[i-1] + du
        s[i] = u[i] * x[i]

    return t, s
    
    
# Parameters
dt = 1/1000  # time step
T = 1.0     # total simulation time
rate = 20.0  # Poisson spike generator firing rate (spikes per second)

# Generate Poisson spike train
poisson_spike_train = poisson_spike_generator(rate, T, dt)

# Parameters for the Tsodyks-Markram model
U = 0.5       # utilization of synaptic efficacy
tau_rec = 0.2  # recovery time constant
tau_facil = 1  # facilitation time constant

# Simulate Tsodyks-Markram synapse model
t, synaptic_efficacy = tsodyks_markram_synapse(poisson_spike_train, U, tau_rec, tau_facil, dt, T)

# Plotting
plt.plot(t, synaptic_efficacy)
plt.xlabel('Time (s)')
plt.ylabel('Synaptic Efficacy')
plt.title('Tsodyks-Markram Synapse Model Simulation with Poisson Spike Train')
plt.show()

