"""
Created on Wed Nov 15 21:27:23 2023

Depression
- RS to FS

Facilitation
- RS to LTS

STF - Short Term Facilitation
STD - Short Term Depression
PL - Pseudo Linear

R - Recovered state - synaptic efficacy
u - Utilization state - synaptic efficacy

@author: celinesoeiro
"""
import matplotlib.pyplot as plt
import numpy as np
from random import seed, random
import seaborn as sns
sns.set()

seed(1)
random_factor = random()

# =============================================================================
# PARAMETERS
# =============================================================================

ms = 100           # 1 second in miliseconds
dt = 0.1           # time step = 1ms
sim_time = 100
sim_steps = int(sim_time/dt)    # 1 second in miliseconds
time = np.arange(1, sim_steps)

# Izhikevich neuron model
vp = 30     # voltage peak
vr = -65    # voltage threshold
rs_params = {'a': 0.02, 'b': 0.2, 'c': -65, 'd': 8}  # Regular Spiking
I_rs = 3.5*7

a_rs = rs_params['a']
b_rs = rs_params['b']
c_rs = rs_params['c'] + 15*random_factor**2
d_rs = rs_params['d'] - 6*random_factor**2

# Tsodkys and Markram synapse model

## Without pseudo-linear synapse contribution
## column 0 is for STF synapses and column 1 is for STD synapses

p = 2 # if pseudo-linear is NOT considered

t_f_E = [670, 17]     # decay time constant of u (resources ready for use)
t_d_E = [138, 671]    # recovery time constant of x (available resources)
t_s_E = 3             # decay time constante of I (PSC current)
U_E = [0.09, 0.5]     # increment of u produced by a spike
A_E = [0.3, 0.7]      # absolute synaptic efficacy

t_f_I = [376, 21]     # decay time constant of u (resources ready for use)
t_d_I = [45, 706]     # recovery time constant of x (available resources)
t_s_I = 11            # decay time constante of I (PSC current)
U_I = [0.016, 0.25]   # increment of u produced by a spike
A_I = [0.18, 0.82]    # absolute synaptic efficacy

# =============================================================================
# VARIABLES
# =============================================================================
# u_E_aux = np.zeros((1, p))
# R_E_aux = np.ones((1, p))
# I_E_aux = np.zeros((1, p))

u_E = np.zeros((1, p))
R_E = np.ones((1, p))
I_E = np.zeros((1, p))

# u_I_aux = np.zeros((1, p))
# R_I_aux = np.ones((1, p))
# I_I_aux = np.zeros((1, p))

u_I = np.zeros((1, p))  
R_I = np.ones((1, p))
I_I = np.zeros((1, p))

n_rs = 1  # number of RS neurons

v_rs = np.zeros((n_rs, sim_steps)); v_rs[0][0] = vr
u_rs = np.zeros((n_rs, sim_steps)); u_rs[0][0] = vr*rs_params['b']

PSC_E = np.zeros((n_rs, sim_steps))
PSC_I = np.zeros((n_rs, sim_steps))

# =============================================================================
# EQUATIONS IN FORM OF FUNCTIONS
# =============================================================================
def tm_synapse_eq(u, R, I, AP, t_f, t_d, t_s, U, A, dt):
    # Solve EDOs using Euler method
    for j in range(p):
        # u -> utilization factor -> resources ready for use
        u[0][j] = u[0][j - 1] + -dt*u[0][j - 1]/t_f[j] + U[j]*(1 - u[0][j - 1])*AP
        # x -> availabe resources -> Fraction of resources that remain available after neurotransmitter depletion
        R[0][j] = R[0][j - 1] + dt*(1 - R[0][j - 1])/t_d[j - 1] - u[0][j]*R[0][j - 1]*AP
        # PSC
        I[0][j] = I[0][j - 1] + -dt*I[0][j - 1]/t_s + A[j - 1]*R[0][j - 1]*u[0][j - 1]*AP
        
    Ipost = np.sum(I)
    
    tm_syn_inst = dict()
    tm_syn_inst['u'] = u
    tm_syn_inst['R'] = R
    tm_syn_inst['I'] = I
    tm_syn_inst['Ipost'] = np.around(Ipost, decimals=6)
        
    return tm_syn_inst

def print_signal(signal, title):
    plt.figure(1)
    plt.suptitle(title)
    plt.plot(signal, 'b')
    plt.grid(True)
    plt.show()

def print_comparison(voltage, PSC, title):
    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 11))
    fig.suptitle(title)
    ax1.set_title('Voltage')
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)
    ax1.xaxis.grid(True, linestyle='-', which='major', color='lightcoral',
                   alpha=0.5)
    ax2.set_title('PSC')
    ax2.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)
    ax2.xaxis.grid(True, linestyle='-', which='major', color='lightcoral',
                   alpha=0.5)

    ax1.plot(voltage)
    ax2.plot(PSC)
    
def izhikevich_dvdt(v, u, I):
    return 0.04*v**2 + 5*v + 140 - u + I

def izhikevich_dudt(v, u, a, b):
    return a*(b*v - u)

def synapse_utilization(u, tau_f, U, AP, dt):
    return -(dt/tau_f)*u + U*(1 - u)*AP

def synapse_recovery(R, tau_d, u_next, AP, dt):
    return (dt/tau_d)*(1 - R) - u_next*R*AP

def synapse_current(I, tau_s, A, R, u_next, AP, dt):
    return -(dt/tau_s)*I + A*R*u_next*AP

# =============================================================================
# MAIN - STD
# =============================================================================
for t in time:
    # RS NEURON - Excitatory
    AP_aux = 0
    for i in range(n_rs):
        v_aux = 1*v_rs[i][t - 1]
        u_aux = 1*u_rs[i][t - 1]
                
        if (v_aux >= vp):
            AP_aux = 1
            v_aux = v_rs[i][t]
            v_rs[i][t] = c_rs
            u_rs[i][t] = u_aux + d_rs
        else:
            AP_aux = 0
            dv_rs = izhikevich_dvdt(v = v_aux, u = u_aux, I = I_rs)
            du_rs = izhikevich_dudt(v = v_aux, u = u_aux, a = a_rs, b = b_rs)
        
            v_rs[i][t] = v_aux + dt*dv_rs
            u_rs[i][t] = u_aux + dt*du_rs
            
        # Synapse        
        syn_E = tm_synapse_eq(u = u_E, R = R_E, I = I_E, AP = AP_aux, t_f = t_f_E, t_d = t_d_E, t_s = t_s_E, U = U_E, A = A_E, dt = dt)
        
        R_E = syn_E['R']
        u_E = syn_E['u']
        I_E = syn_E['I']
        PSC_E[0][t] = syn_E['Ipost']
        
        syn_I = tm_synapse_eq(u = u_I, R = R_I, I = I_I, AP = AP_aux, t_f = t_f_I, t_d = t_d_I, t_s = t_s_I, U = U_I, A = A_I, dt = dt)
        
        R_I = syn_I['R']
        u_I = syn_I['u']
        I_I = syn_I['I']
        PSC_I[0][t] = syn_I['Ipost']
        

print_comparison(voltage = v_rs[0], PSC = PSC_E[0], title = 'Regular Spiking - Excitatory')
print_comparison(voltage = v_rs[0], PSC = PSC_I[0], title = 'Regular Spiking - Inhibitory')


