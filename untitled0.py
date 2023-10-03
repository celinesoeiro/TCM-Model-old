#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 10:02:26 2023

@author: celinesoeiro
"""

import numpy as np
import matplotlib.pyplot as plt

class IzhikevichNeuron:
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.v = -70.0  # Initial membrane potential (mV)
        self.u = self.b * self.v  # Initial recovery variable
        self.spike_times = []  # Store spike times

    def simulate(self, sim_time, dt, input_current):
        num_steps = int(sim_time / dt)
        membrane_potential = np.zeros(num_steps)

        for t in range(num_steps):
            # Update membrane potential and recovery variable
            dv = (0.04 * self.v ** 2) + (5 * self.v) + 140 - self.u + input_current[t]
            du = self.a * (self.b * self.v - self.u)

            self.v += dv * dt
            self.u += du * dt

            # Check for a spike and reset if the threshold is crossed
            if self.v >= 30:
                self.v = self.c
                self.u += self.d
                self.spike_times.append(t * dt)

            membrane_potential[t] = self.v

        return membrane_potential

# Parameters for the 10 neurons
neuron_params = [
    (0.02, 0.2, -65.0, 8.0),
    (0.02, 0.25, -65.0, 2.0),
    (0.02, 0.2, -50.0, 2.0),
    (0.02, 0.25, -55.0, 0.05),
    (0.01, 0.2, -65.0, 2.0),
    (0.03, 0.2, -64.0, 1.0),
    (0.01, 0.2, -65.0, 2.0),
    (0.03, 0.2, -64.0, 1.0),
    (0.03, 0.25, -52.0, 0.05),
    (0.03, 0.25, -50.0, 0.05)
]

# Create and simulate 10 neurons
sim_time = 1.0       # Simulation time (seconds)
dt = 0.01            # Time step (seconds), equivalent to 10 ms
num_neurons = len(neuron_params)

neurons = []
membrane_potentials = []

for params in neuron_params:
    neuron = IzhikevichNeuron(*params)
    input_current = np.random.randn(int(sim_time / dt)) * 5.0  # Random input current for each neuron
    membrane_potential = neuron.simulate(sim_time, dt, input_current)
    neurons.append(neuron)
    membrane_potentials.append(membrane_potential)

# Plot membrane potentials and spike times for all neurons
time_axis = np.arange(0, sim_time, dt)
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
for i, membrane_potential in enumerate(membrane_potentials):
    plt.plot(time_axis, membrane_potential, label=f'Neuron {i + 1}')
plt.xlabel('Time (s)')
plt.ylabel('Membrane Potential (mV)')
plt.title('Membrane Potentials of 10 Neurons')
plt.legend()

plt.subplot(2, 1, 2)
for i, neuron in enumerate(neurons):
    plt.scatter(neuron.spike_times, [i] * len(neuron.spike_times), c='r', marker='|', linewidths=0.5, label=f'Neuron {i + 1}')
plt.xlabel('Time (s)')
plt.ylabel('Neuron')
plt.title('Spike Times of 10 Neurons')
plt.legend()

plt.tight_layout()
plt.show()
