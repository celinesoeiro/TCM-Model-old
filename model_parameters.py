"""
@author: Celine Soeiro

@description: Thalamo-Cortical microcircuit by AmirAli Farokhniaee and Madeleine M. Lowery - 2021

# Abreviations:
    PD: Parkinson Desease
    S: Superficial layer
    M: Medium layer
    D: Deep layer
    CI: Cortical Interneurons
    TC: Thalamo-Cortical Relay Nucleus (TC)
    TR: Thalamic Reticular Nucleus (TR)
    PD: Poissonian Distribution 
    DBS: Deep Brain Stimulation
"""

import numpy as np
from utils import pinkNoise
from random import seed, random

def TCM_model_parameters():
    seed(1)
    random_factor = random()
    
    number_trials = 1                           # number of trials
    dt = 0.1                                    # time step in ms
    fs = 1000/dt                                # sampling frequency in Hz
    Fs = int(np.round(fs))                      # 
    dbs_on = 5*67                               # value of synaptic fidelity when DBS on
    dbs_off = 0                                 # value of synaptic fidelity when DBS off
    synaptic_fidelity = dbs_off                 # synaptic fidelity
    simulation_time = 5                        # simulation time in seconds
    sim_time_ms = (simulation_time + 1)*1000    # Simulation time in ms with 1 extra second to reach the steady state and trash later
    sim_steps = int(np.round(sim_time_ms/dt))   # number of simulation steps
    chop_till = 1*Fs;                           # Cut the first 1 seconds of the simulation

    td_synapse = 1                              # Synaptic transmission delay (fixed for all synapses in the TCM)
    td_thalamus_cortex = 15 # 25                # time delay between thalamus and cortex (ms) (transmission time delay)
    td_cortex_thalamus = 20                     # time delay between cortex and thalamus (ms) (transmission time delay)  
    td_layers = 8                               # time delay between the layers in corticex and nuclei in thalamus (ms)
    td_within_layers = 1                        # time delay within a structure (ms)
    
    hyperdirect_neurons = 0.1                   # percentage of PNs that are hyperdirect
    
    connectivity_factor_normal = 2.5            # For 100 neurons
    connectivity_factor_PD = 5                  # For 100 neurons
    
    Idc_tune = 0.1                              # 
    vr = -65                                    # membrane potential resting value 
    vp = 30                                     # membrane peak voltage value
    
    # Time vector
    if (td_thalamus_cortex >= td_cortex_thalamus): 
        t_vec = np.arange(td_thalamus_cortex + td_synapse + 1, sim_steps)
    else:
        t_vec = np.arange(td_cortex_thalamus + td_synapse + 1, sim_steps)
        
    # Neuron quantities
    qnt_neurons_s = 100         # Excitatory
    qnt_neurons_m = 100         # Excitatory
    qnt_neurons_d = 100         # Excitatory
    qnt_neurons_ci = 100        # Excitatory
    qnt_neurons_tc = 100        # Inhibitory
    qnt_neurons_tr = 40         # Inhibitory
    
    neuron_quantities = {
        'S': qnt_neurons_s,                      # Number of neurons in Superficial layer
        'M': qnt_neurons_m,                      # Number of neurons in Medium layer
        'D': qnt_neurons_d,                      # Number of neurons in Deep layer
        'CI': qnt_neurons_ci,                    # Number of IC neurons
        'TC': qnt_neurons_tc,                    # Number of neurons in TC
        'TR': qnt_neurons_tr,                    # Number of neurons in TR
        'HD': qnt_neurons_d*hyperdirect_neurons, # Number of hyperdirect neurons
        'total': qnt_neurons_s + qnt_neurons_m + qnt_neurons_d + qnt_neurons_ci + qnt_neurons_tc + qnt_neurons_tr,
        }
    
    # Impact of DBS on the other cortical structures via D PNs axons:
    synaptic_fidelity_per_structure = {
        'CI': 1*synaptic_fidelity, # the synaptic fidelity, for dbs carriers (to be used to invade CIs)
        'M': 0*synaptic_fidelity, # the synaptic fidelity, for dbs carriers (to be used to invade layer M)
        'S': 1*synaptic_fidelity, # the synaptic fidelity, for dbs carriers (to be used to invade layer S)
        'TC': 1*synaptic_fidelity,# the synaptic fidelity, for dbs carriers (to be used to invade layer TCR)
        'TR': 1*synaptic_fidelity,# the synaptic fidelity, for dbs carriers (to be used to invade layer TRN)
        }
    
    nCI = 1; nS = 1; nR = 1; nN = 1; nM = 0;
    # Percentage of neurons that have synaptic contact with hyperdirect neurons axon arbors
    neurons_connected_with_hyperdirect_neurons = {
        'CI': nCI*hyperdirect_neurons*qnt_neurons_ci,# percentage of CI neurons that have synaptic contact with hyperdirect neurons axon arbors
        'S': nS*hyperdirect_neurons*qnt_neurons_s,   # percentage of S neurons that have synaptic contact with hyperdirect neurons axon arbors
        'M': nM*hyperdirect_neurons*qnt_neurons_m,   # percentage of M neurons that have synaptic contact with hyperdirect neurons axon arbors
        'TR': nR*hyperdirect_neurons*qnt_neurons_tr, # percentage of R neurons that have synaptic contact with hyperdirect neurons axon arbors
        'TC': nN*hyperdirect_neurons*qnt_neurons_tc, # percentage of N neurons that have synaptic contact with hyperdirect neurons axon arbors
        }
    
    # Distribution of neurons in each structure
    neurons_s_1 = int(0.5*qnt_neurons_s)
    neurons_s_2 = int(0.5*qnt_neurons_s)
    neurons_m_1 = int(1*qnt_neurons_m)
    neurons_m_2 = int(0*qnt_neurons_m)
    neurons_d_1 = int(0.7*qnt_neurons_d)
    neurons_d_2 = int(0.3*qnt_neurons_d)
    neurons_ci_1 = int(0.5*qnt_neurons_ci)
    neurons_ci_2 = int(0.5*qnt_neurons_ci)
    neurons_tcr_tr_1 = int(0.5*qnt_neurons_tr)
    neurons_tcr_tr_2 = int(0.5*qnt_neurons_tr)
    neurons_tcr_tc_1 = int(0.7*qnt_neurons_tc)
    neurons_tcr_tc_2 = int(0.3*qnt_neurons_tc)
    
    neuron_per_structure = {
        'neurons_s_1': neurons_s_1,             # Regular Spiking
        'neurons_s_2': neurons_s_2,             # Intrinsically Bursting
        'neurons_m_1': neurons_m_1,             # Regular Spiking
        'neurons_m_2': neurons_m_2,             # Regular Spiking
        'neurons_d_1': neurons_d_1,             # Regular Spiking
        'neurons_d_2': neurons_d_2,             # Intrinsically_bursting
        'neurons_ci_1': neurons_ci_1,           # Fast spiking
        'neurons_ci_2': neurons_ci_2,           # Low threshold_spiking
        'neurons_tcr_tc_1': neurons_tcr_tc_1,   # Reley
        'neurons_tcr_tc_2': neurons_tcr_tc_2,   # Relay
        'neurons_tcr_tr_1': neurons_tcr_tr_1,   # Reticular
        'neurons_tcr_tr_2': neurons_tcr_tr_2,   # Reticular
        }
    
    # Neuron parameters to model Izhikevich Neurons
    # 0 - RS - Regular Spiking
    # 1 - IB - Intrinsically Bursting
    # 2 - CH - Chattering
    # 3 - FS - Fast Spiking
    # 4 - LTS - Low Threshold Spiking
    # 5 - TC (rel) - Thalamo-Cortical Relay
    # 6 - CH (rel) - 
    # 7 - TR - Thalamic Reticular

        # 0-RS 1-IB  2-CH 3-FS 4-LTS 5-TC  6-CH  7-TR 
    a = [0.02, 0.02, 0.5, 0.1, 0.02, 0.02, 0.02, 0.02]
    b = [0.2,  0.2,  0.2, 0.2, 0.25, 0.25, 0.25, 0.25]
    c = [-65,  -55,  -50, -65, -65,  -65,  -65,  -65]
    d = [8,    4,    2,   2,   2,    0.05, 0.05, 2.05]
    
    neuron_params = {
        'S1': {
            'a': a[0],
            'b': b[0],
            'c': c[0],
            'd': d[0],
            },
        'S2': {
            'a': a[1],
            'b': b[1],
            'c': c[1],
            'd': d[1],
            },
        'M1': {
            'a': a[0],
            'b': b[0],
            'c': c[0],
            'd': d[0],
            },
        'M2': {
            'a': a[0],
            'b': b[0],
            'c': c[0],
            'd': d[0],
            },
        'D1': {
            'a': a[0],
            'b': b[0],
            'c': c[0],
            'd': d[0],
            },
        'D2': {
            'a': a[1],
            'b': b[1],
            'c': c[1],
            'd': d[1],
            },
        'CI1': {
            'a': a[3],
            'b': b[3],
            'c': c[3],
            'd': d[3],
            },
        'CI2': {
            'a': a[4],
            'b': b[4],
            'c': c[4],
            'd': d[4],
            },
        'TR1': {
            'a': a[7],
            'b': b[7],
            'c': c[7],
            'd': d[7],
            },
        'TR2': {
            'a': a[7],
            'b': b[7],
            'c': c[7],
            'd': d[7],
            },
        'TC1': {
            'a': a[5],
            'b': b[5],
            'c': c[5],
            'd': d[5],
            },
        'TC2': {
            'a': a[5],
            'b': b[5],
            'c': c[5],
            'd': d[5],
            },
        }

    model_global_parameters = {
        'number_trials': number_trials,
        'synaptic_fidelity': synaptic_fidelity, # DBS off - To turn DBS on set this value to 5*67
        'hyperdirect_neurons': hyperdirect_neurons, # Percentage of PNs affected in D by DBS
        'simulation_time': simulation_time, # simulation time in seconds (must be a multiplacative of 3 under PD+DBS condition)
        'simulation_time_ms': sim_time_ms,
        'dt': dt, # time step
        'sampling_frequency': fs, # in Hz
        'simulation_steps': sim_steps,
        'chop_till': chop_till, # cut the first 1s of simulation
        'time_delay_between_layers': 8,
        'time_delay_within_layers': 1,
        'time_delay_thalamus_cortex': 15,
        'time_delay_cortex_thalamus': 20,
        'transmission_delay_synapse': 1,
        'time_vector': t_vec,
        'connectivity_factor_normal_condition': connectivity_factor_normal,
        'connectivity_factor_PD_condition': connectivity_factor_PD,
        'vr': vr,
        'vp': vp,
        'Idc': Idc_tune,
        }
    
    # Noise terms
    w_g_n_add = 1.5 # additive white Gaussian noise strength
    w_g_n_thr = 0.5 # additive white Gaussian noise strength
    cn = 1
    p_add = 0.0 # additive pink noise strength
    
    kisi_S_E = [w_g_n_add*np.random.rand(qnt_neurons_s, Fs), w_g_n_add*cn*np.random.rand(qnt_neurons_s, sim_steps-Fs)]
    kisi_M_E = [w_g_n_add*np.random.rand(qnt_neurons_m, Fs), w_g_n_add*cn*np.random.rand(qnt_neurons_m, sim_steps-Fs)]
    kisi_D_E = [w_g_n_add*np.random.rand(qnt_neurons_d, Fs), w_g_n_add*cn*np.random.rand(qnt_neurons_d, sim_steps-Fs)]
    kisi_CI_I = [w_g_n_add*np.random.rand(qnt_neurons_ci, Fs), w_g_n_add*cn*np.random.rand(qnt_neurons_ci, sim_steps-Fs)]
    kisi_TC_E = [w_g_n_add*np.random.rand(qnt_neurons_tc, Fs), w_g_n_add*cn*np.random.rand(qnt_neurons_tc, sim_steps-Fs)]
    kisi_TR_I = [w_g_n_add*np.random.rand(qnt_neurons_tr, Fs), w_g_n_add*cn*np.random.rand(qnt_neurons_tr, sim_steps-Fs)]    
    
    zeta_S_E = [w_g_n_thr*np.random.rand(qnt_neurons_s, Fs), w_g_n_thr*cn*np.random.rand(qnt_neurons_s, sim_steps-Fs)]
    zeta_M_E = [w_g_n_thr*np.random.rand(qnt_neurons_m, Fs), w_g_n_thr*cn*np.random.rand(qnt_neurons_m, sim_steps-Fs)]
    zeta_D_E = [w_g_n_thr*np.random.rand(qnt_neurons_d, Fs), w_g_n_thr*cn*np.random.rand(qnt_neurons_d, sim_steps-Fs)]
    zeta_CI_I = [w_g_n_thr*np.random.rand(qnt_neurons_ci, Fs), w_g_n_thr*cn*np.random.rand(qnt_neurons_ci, sim_steps-Fs)]
    zeta_TC_E = [w_g_n_thr*np.random.rand(qnt_neurons_tc, Fs), w_g_n_thr*cn*np.random.rand(qnt_neurons_tc, sim_steps-Fs)]
    zeta_TR_I = [w_g_n_thr*np.random.rand(qnt_neurons_tr, Fs), w_g_n_thr*cn*np.random.rand(qnt_neurons_tr, sim_steps-Fs)]   
    
    pn_s_1 = pinkNoise(qnt_neurons_s, Fs);  pn_s_2 = pinkNoise(qnt_neurons_s, sim_steps-Fs);
    pn_m_1 = pinkNoise(qnt_neurons_s, Fs);  pn_m_2 = pinkNoise(qnt_neurons_s, sim_steps-Fs);
    pn_d_1 = pinkNoise(qnt_neurons_s, Fs);  pn_d_2 = pinkNoise(qnt_neurons_s, sim_steps-Fs);
    pn_ci_1 = pinkNoise(qnt_neurons_s, Fs);  pn_ci_2 = pinkNoise(qnt_neurons_s, sim_steps-Fs);
    pn_tc_1 = pinkNoise(qnt_neurons_s, Fs);  pn_tc_2 = pinkNoise(qnt_neurons_s, sim_steps-Fs);
    pn_tr_1 = pinkNoise(qnt_neurons_s, Fs);  pn_tr_2 = pinkNoise(qnt_neurons_s, sim_steps-Fs);
    
    pn_S_E = [p_add*pn_s_1, p_add*cn*pn_s_2]
    pn_M_E = [p_add*pn_m_1, p_add*cn*pn_m_2]
    pn_D_E = [p_add*pn_d_1, p_add*cn*pn_d_2]
    pn_CI_I = [p_add*pn_ci_1, p_add*cn*pn_ci_2]
    pn_TC_E = [p_add*pn_tc_1, p_add*cn*pn_tc_2]
    pn_TR_I = [p_add*pn_tr_1, p_add*cn*pn_tr_2]
    
    noise = {
        'kisi_S_E': kisi_S_E,
        'kisi_M_E': kisi_M_E,
        'kisi_D_E': kisi_D_E,
        'kisi_CI_I': kisi_CI_I,
        'kisi_TC_E': kisi_TC_E,
        'kisi_TR_I': kisi_TR_I,
        'zeta_S_E': zeta_S_E,
        'zeta_M_E': zeta_M_E,
        'zeta_D_E': zeta_D_E,
        'zeta_CI_I': zeta_CI_I,
        'zeta_TC_E': zeta_TC_E,
        'zeta_TR_I': zeta_TR_I,
        'pn_S_E': pn_S_E,
        'pn_M_E': pn_M_E,
        'pn_D_E': pn_D_E,
        'pn_CI_I': pn_CI_I,
        'pn_TC_E': pn_TC_E,
        'pn_TR_I': pn_TR_I,
        }
    
    # Bias currents (Subthreshold CTX and Suprethreshold THM) - Will be used in the neurons
    Idc=[3.5,3.6,3.5,3.8,0.4,0.6,0.5,0.6]
    
    I_S_1 = Idc[0]
    I_S_2 = Idc[1]
    I_M_1 = Idc[0]
    I_M_2 = Idc[0]
    I_D_1 = Idc[5]
    I_D_2 = Idc[1]
    I_CI_1 = Idc[3]
    I_CI_2 = Idc[4]
    I_TR_1 = Idc[7]
    I_TR_2 = Idc[7]
    I_TC_1 = Idc[5]
    I_TC_2 = Idc[5]
    
    # I_S = [Idc[0][0]*np.ones((50, 1)), Idc[0][1]*np.ones((50, 1))]
    # I_M = [Idc[0][0]*np.ones((50, 1)), Idc[0][0]*np.ones((50, 1))]
    # I_D = [Idc[0][0]*np.ones((50, 1)), Idc[0][1]*np.ones((50, 1))]
    # I_CI = [Idc[0][3]*np.ones((50, 1)), Idc[0][4]*np.ones((50, 1))]
    # I_Ret = [Idc[0][7]*np.ones((50, 1)), Idc[0][7]*np.ones((50, 1))]
    # I_Rel = [Idc[0][5]*np.ones((20, 1)), Idc[0][5]*np.ones((20, 1))]
    
    currents_per_structure = {
        'I_S_1': I_S_1,
        'I_M_1': I_M_1,
        'I_D_1': I_D_1,
        'I_CI_1': I_CI_1,
        'I_TR_1': I_TR_1,
        'I_TC_1': I_TC_1,
        'I_S_2': I_S_2,
        'I_M_2': I_M_2,
        'I_D_2': I_D_2,
        'I_CI_2': I_CI_2,
        'I_TR_2': I_TR_2,
        'I_TC_2': I_TC_2,
        }
    
    # Export all dictionaries
    data = {
        'neuron_quantities': neuron_quantities,
        'neuron_per_structure': neuron_per_structure,
        'model_global_parameters': model_global_parameters,
        'synaptic_fidelity_per_structure': synaptic_fidelity_per_structure,
        'neurons_connected_with_hyperdirect_neurons': neurons_connected_with_hyperdirect_neurons,
        'neuron_paramaters': neuron_params,
        'bias_current': Idc,
        'currents_per_structure': currents_per_structure,
        'noise': noise,
        'random_factor': random_factor,
        }
    
    return data

def coupling_matrix_normal(facilitating_factor, n_s, n_m, n_d, n_ci, n_tc, n_tr):
    division_factor = facilitating_factor
    
    initial = 0
    final = 1
    interval = final - initial
    
    # =============================================================================
    #     These are to restrict the normalized distribution variance or deviation from the mean
    # =============================================================================
    r_s = initial + interval*np.random.rand(n_s, 1)
    r_m = initial + interval*np.random.rand(n_m, 1)
    r_d = initial + interval*np.random.rand(n_d, 1)
    r_ci = initial + interval*np.random.rand(n_ci, 1)
    r_tr = initial + interval*np.random.rand(n_tr, 1)
    r_tc = initial + interval*np.random.rand(n_tc, 1)
    
    # =============================================================================
    #     COUPLING STRENGTHs within each structure (The same in Normal and PD)
    # EE -> Excitatory to Excitatory 
    # II -> Inhibitory to Inhibitory 
    # =============================================================================
    ## Layer S (was -1e-2 for IEEE paper)
    aee_s = -1e1/division_factor;            W_EE_s = aee_s*r_s;
    ## Layer M (was -1e-2 for IEEE paper)
    aee_m = -1e1/division_factor;            W_EE_m = aee_m*r_m;
    ## Layer D (was -1e-2 for IEEE paper)
    aee_d = -1e1/division_factor;            W_EE_d = aee_d*r_d;
    ## INs 
    aii_INs = -5e2/division_factor;          W_II_ci = aii_INs*r_ci;
    ## Reticular cells
    aii_tr = -5e1/division_factor;          W_II_tr = aii_tr*r_tr;
    ## Relay cells
    aee_tc = 0/division_factor;             W_EE_tc = aee_tc*r_tc;
    
    # =============================================================================
    #     COUPLING STRENGTHs between structures
    # =============================================================================
    #     S
    # =============================================================================
    # M to S coupling
    aee_sm = 1e1/division_factor;          W_EE_s_m = aee_sm*r_s;
    # D to S coupling
    aee_sd = 5e2/division_factor;          W_EE_s_d = aee_sd*r_s;
    # CI (INs) to S coupling
    aei_sci = -5e2/division_factor;        W_EI_s_ci = aei_sci*r_s;
    # Reticular to S coupling
    aei_str = 0/division_factor;           W_EI_s_tr = aei_str*r_s;
    # Rel. to S couplings
    aee_stc = 0/division_factor;           W_EE_s_tc = aee_stc*r_s;     
    # =============================================================================
    #     M
    # =============================================================================
    # S to M
    aee_ms = 3e2/facilitating_factor;       W_EE_m_s = aee_ms*r_m; 
    # D to M couplings
    aee_md = 0/facilitating_factor;         W_EE_m_d = aee_md*r_m;            
    # INs to M couplings
    aei_mci = -3e2/facilitating_factor;     W_EI_m_ci = aei_mci*r_m;
    # Ret. to M couplings    
    aei_mtr = 0/facilitating_factor;        W_EI_m_tr = aei_mtr*r_m;
    # Rel. to M couplings
    aee_mtc = 0/facilitating_factor;        W_EE_m_tc = aee_mtc*r_m;
    # =============================================================================
    #     D
    # =============================================================================
    # S to D couplings
    aee_ds = 3e2/facilitating_factor;       W_EE_d_s = aee_ds*r_d;
    # M to D couplings
    aee_dm = 0/facilitating_factor;         W_EE_d_m = aee_dm*r_d;
    # INs to D couplings
    aei_dci = -7.5e3/facilitating_factor;   W_EI_d_ci = aei_dci*r_d;
    # Ret. to D couplings
    aei_dtr = 0/facilitating_factor;        W_EI_d_tr = aei_dtr*r_d;
    # Rel. to D couplings
    aee_dtc = 1e1/facilitating_factor;      W_EE_d_tc = aee_dtc*r_d;
    # =============================================================================
    #     INs (CI)
    # =============================================================================
    # S to CIs couplings
    aie_CIs = 2e2/facilitating_factor;     W_IE_ci_s = aie_CIs*r_ci;
    # M to CIs couplings
    aie_CIm = 2e2/facilitating_factor;     W_IE_ci_m = aie_CIm*r_ci;
    # D to CIs couplings
    aie_CId = 2e2/facilitating_factor;     W_IE_ci_d = aie_CId*r_ci;
    # Ret. to CIs couplings
    aii_CITR = 0/facilitating_factor;      W_II_ci_tr = aii_CITR*r_ci;
    # Rel. to CIs couplings
    aie_CITC = 1e1/facilitating_factor;    W_IE_ci_tc = aie_CITC*r_ci;
    # =============================================================================
    #     Reticular
    # =============================================================================
    # S to Ret couplings
    aie_trs = 0/facilitating_factor;       W_IE_tr_s = aie_trs*r_tr;
    # M to Ret couplings
    aie_trm = 0/facilitating_factor;       W_IE_tr_m = aie_trm*r_tr;
    # D to Ret couplings
    aie_trd = 7e2/facilitating_factor;     W_IE_tr_d = aie_trd*r_tr;
    # Ret. Ret INs couplings
    aii_trci = 0/facilitating_factor;      W_II_tr_ci = aii_trci*r_tr;
    # Rel. Ret INs couplings
    aie_trtc = 1e1/facilitating_factor;    W_IE_tr_tc = aie_trtc*r_tr;
    # =============================================================================
    #     Rele
    # =============================================================================
    # S to Rel couplings
    aee_tcs = 0/facilitating_factor;       W_EE_tc_s = aee_tcs*r_tc;   
    # M to Rel couplings
    aee_tcm = 0/facilitating_factor;       W_EE_tc_m = aee_tcm*r_tc;
    # D to Rel couplings
    aee_tcd = 7e2/facilitating_factor;     W_EE_tc_d = aee_tcd*r_tc;
    # INs to Rel couplings
    aei_tcci = 0/facilitating_factor;      W_EI_tc_ci = aei_tcci*r_tc;
    # Ret to Rel couplings
    aei_tctr = -5e2/facilitating_factor;   W_EI_tc_tr = aei_tctr*r_tc;
    
    # Initialize matrix (6 structures -> 6x6 matrix)
    matrix = np.zeros((6,6))
    
    # Populating the matrix
    # Main Diagonal
    matrix[0][0] = np.mean(W_EE_s)
    matrix[1][1] = np.mean(W_EE_m)
    matrix[2][2] = np.mean(W_EE_d)
    matrix[3][3] = np.mean(W_II_ci)
    matrix[4][4] = np.mean(W_II_tr)
    matrix[5][5] = np.mean(W_EE_tc)
    # First column - Layer S
    matrix[1][0] = np.mean(W_EE_s_m)
    matrix[2][0] = np.mean(W_EE_s_d)
    matrix[3][0] = np.mean(W_EI_s_ci)
    matrix[4][0] = np.mean(W_EI_s_tr)
    matrix[5][0] = np.mean(W_EE_s_tc)
    # Second column - Layer M
    matrix[0][1] = np.mean(W_EE_m_s)
    matrix[2][1] = np.mean(W_EE_m_d)
    matrix[3][1] = np.mean(W_EI_m_ci)
    matrix[4][1] = np.mean(W_EI_m_tr)
    matrix[5][1] = np.mean(W_EE_m_tc)
    # Thid column - Layer D
    matrix[0][2] = np.mean(W_EE_d_s)
    matrix[1][2] = np.mean(W_EE_d_m)
    matrix[3][2] = np.mean(W_EI_d_ci)
    matrix[4][2] = np.mean(W_EI_d_tr)
    matrix[5][2] = np.mean(W_EE_d_tc)
    # Fourth column - Structure CI
    matrix[0][3] = np.mean(W_IE_ci_s)
    matrix[1][3] = np.mean(W_IE_ci_m)
    matrix[2][3] = np.mean(W_IE_ci_d)
    matrix[4][3] = np.mean(W_II_ci_tr)
    matrix[5][3] = np.mean(W_IE_ci_tc)
    # Fifth column - Structure TCR
    matrix[0][4] = np.mean(W_IE_tr_s)
    matrix[1][4] = np.mean(W_IE_tr_m)
    matrix[2][4] = np.mean(W_IE_tr_d)
    matrix[3][4] = np.mean(W_II_tr_ci)
    matrix[5][4] = np.mean(W_IE_tr_tc)
    # Sixth column - Structure TRN
    matrix[0][5] = np.mean(W_EE_tc_s)
    matrix[1][5] = np.mean(W_EE_tc_m)
    matrix[2][5] = np.mean(W_EE_tc_d)
    matrix[3][5] = np.mean(W_EI_tc_ci)
    matrix[4][5] = np.mean(W_EI_tc_tr)
    
    weights = {
        'W_EE_s': W_EE_s,
        'W_EE_m': W_EE_m,
        'W_EE_d': W_EE_d,
        'W_II_ci': W_II_ci,
        'W_II_tr': W_II_tr,
        'W_EE_tc': W_EE_tc,
        
        'W_EE_s_m': W_EE_s_m,
        'W_EE_s_d': W_EE_s_d,
        'W_EI_s_ci': W_EI_s_ci,
        'W_EI_s_tr': W_EI_s_tr,
        'W_EE_s_tc': W_EE_s_tc,
        
        'W_EE_m_s': W_EE_m_s,
        'W_EE_m_d': W_EE_m_d,
        'W_EI_m_ci': W_EI_m_ci,
        'W_EI_m_tr': W_EI_m_tr,
        'W_EE_m_tc': W_EE_m_tc,
        
        'W_EE_d_s': W_EE_d_s,
        'W_EE_d_m': W_EE_d_m,
        'W_EI_d_ci': W_EI_d_ci,
        'W_EI_d_tr': W_EI_d_tr,
        'W_EE_d_tc': W_EE_d_tc,
        
        'W_IE_ci_s': W_IE_ci_s,
        'W_IE_ci_m': W_IE_ci_m,
        'W_IE_ci_d': W_IE_ci_d,
        'W_II_ci_tr': W_II_ci_tr,
        'W_IE_ci_tc': W_IE_ci_tc,
        
        'W_IE_tr_s': W_IE_tr_s,
        'W_IE_tr_m': W_IE_tr_m,
        'W_IE_tr_d': W_IE_tr_d,
        'W_II_tr_ci': W_II_tr_ci,
        'W_IE_tr_tc': W_IE_tr_tc,
        
        'W_EE_tc_s': W_EE_tc_s,
        'W_EE_tc_m': W_EE_tc_m,
        'W_EE_tc_d': W_EE_tc_d,
        'W_EI_tc_ci': W_EI_tc_ci,
        'W_EI_tc_tr': W_EI_tc_tr,
        }
    
    return { 'matrix': matrix, 'weights': weights }
    
    
def coupling_matrix_PD(facilitating_factor, n_s, n_m, n_d, n_ci, n_tc, n_tr):
    division_factor = facilitating_factor
    
    initial = 0
    final = 1
    interval = final - initial
    
    # =============================================================================
    #     These are to restrict the normalized distribution variance or deviation from the mean
    # =============================================================================
    r_s = initial + interval*np.random.rand(n_s, 1)
    r_m = initial + interval*np.random.rand(n_m, 1)
    r_d = initial + interval*np.random.rand(n_d, 1)
    r_ci = initial + interval*np.random.rand(n_ci, 1)
    r_tc = initial + interval*np.random.rand(n_tc, 1)
    r_tr = initial + interval*np.random.rand(n_tr, 1)
    
    # =============================================================================
    #     COUPLING STRENGTHs within each structure (The same in Normal and PD)
    # EE -> Excitatory to Excitatory  
    # II -> Inhibitory to Inhibitory 
    # =============================================================================
    ## Layer S (was -1e-2 for IEEE paper)
    aee_s = -5e1/division_factor;           W_EE_s = aee_s*r_s;
    ## Layer M (was -1e-2 for IEEE paper)
    aee_m = -5e1/division_factor;           W_EE_m = aee_m*r_m;
    ## Layer D (was -1e-2 for IEEE paper)
    aee_d = -5e1/division_factor;           W_EE_d = aee_d*r_d;
    ## INs 
    aii_ci = -5e1/division_factor;          W_II_ci = aii_ci*r_ci;
    ## Reticular cells
    aii_tr = -5e1/division_factor;          W_II_tr = aii_tr*r_tr;
    ## Relay cells
    aee_tc = 0/division_factor;             W_EE_tc = aee_tc*r_tc;
    
    # =============================================================================
    #     COUPLING STRENGTHs between structures
    # =============================================================================
    #     S
    # =============================================================================
    # M to S coupling
    aee_sm = 3e2/division_factor;          W_EE_s_m = aee_sm*r_s;
    # D to S coupling
    aee_sd = 5e2/division_factor;          W_EE_s_d = aee_sd*r_s;
    # CI (INs) to S coupling
    aei_sci = -7.5e2/division_factor;      W_EI_s_ci = aei_sci*r_s;
    # Reticular to S coupling
    aei_str = 0/division_factor;           W_EI_s_tr = aei_str*r_s;
    # Rel. to S couplings
    aee_stc = 0/division_factor;           W_EE_s_tc = aee_stc*r_s;     
    # =============================================================================
    #     M
    # =============================================================================
    # S to M
    aee_ms = 1e1/facilitating_factor;       W_EE_m_s = aee_ms*r_m; 
    # D to M couplings
    aee_md = 0/facilitating_factor;         W_EE_m_d = aee_md*r_m;            
    # INs to M couplings
    aei_mci = -7.5e2/facilitating_factor;   W_EI_m_ci = aei_mci*r_m;
    # Ret. to M couplings    
    aei_mtr = 0/facilitating_factor;        W_EI_m_tr = aei_mtr*r_m;
    # Rel. to M couplings
    aee_mtc = 0/facilitating_factor;        W_EE_m_tc = aee_mtc*r_m;
    # =============================================================================
    #     D
    # =============================================================================
    # S to D couplings
    aee_ds = 3e2/facilitating_factor;       W_EE_d_s = aee_ds*r_d;
    # M to D couplings
    aee_dm = 0/facilitating_factor;         W_EE_d_m = aee_dm*r_d;
    # INs to D couplings
    aei_dci = -5e3/facilitating_factor;     W_EI_d_ci = aei_dci*r_d;
    # Ret. to D couplings
    aei_dtr = 0/facilitating_factor;        W_EI_d_tr = aei_dtr*r_d;
    # Rel. to D couplings
    aee_dtc = 1e3/facilitating_factor;      W_EE_d_tc = aee_dtc*r_d;
    # =============================================================================
    #     INs (CI)
    # =============================================================================
    # S to INs couplings
    aie_cis = 2e2/facilitating_factor;      W_IE_ci_s = aie_cis*r_ci;
    # M to INs couplings
    aie_cim = 2e2/facilitating_factor;      W_IE_ci_m = aie_cim*r_ci;
    # D to INs couplings
    aie_cid = 2e2/facilitating_factor;      W_IE_ci_d = aie_cid*r_ci;
    # Ret. to INs couplings
    aii_citr = 0/facilitating_factor;       W_II_ci_tr = aii_citr*r_ci;
    # Rel. to INs couplings
    aie_citc = 1e3/facilitating_factor;     W_IE_ci_tc = aie_citc*r_ci;
    # =============================================================================
    #     Reticular
    # =============================================================================
    # S to Ret couplings
    aie_trs = 0/facilitating_factor;        W_IE_tr_s = aie_trs*r_tr;
    # M to Ret couplings
    aie_trm = 0/facilitating_factor;        W_IE_tr_m = aie_trm*r_tr;
    # D to Ret couplings
    aie_trd = 1e2/facilitating_factor;      W_IE_tr_d = aie_trd*r_tr;
    # Ret. Ret INs couplings
    aii_trci = 0/facilitating_factor;       W_II_tr_ci = aii_trci*r_tr;
    # Rel. Ret INs couplings
    aie_trtc = 5e2/facilitating_factor;     W_IE_tr_tc = aie_trtc*r_tr;
    # =============================================================================
    #     Rele
    # =============================================================================
    # S to Rel couplings
    aee_tcs = 0/facilitating_factor;       W_EE_tc_s = aee_tcs*r_tc;   
    # M to Rel couplings
    aee_tcm = 0/facilitating_factor;       W_EE_tc_m = aee_tcm*r_tc;
    # D to Rel couplings
    aee_tcd = 1e2/facilitating_factor;     W_EE_tc_d = aee_tcd*r_tc;
    # INs to Rel couplings
    aei_tcci = 0/facilitating_factor;     W_EI_tc_ci = aei_tcci*r_tc;
    # Ret to Rel couplings
    aei_tctr = -2.5*1e3/facilitating_factor;  W_EI_tc_tr = aei_tctr*r_tc;
    
    # Initialize matrix (6 structures -> 6x6 matrix)
    matrix = np.zeros((6,6))
    
    # Populating the matrix
    # Main Diagonal
    matrix[0][0] = np.mean(W_EE_s)
    matrix[1][1] = np.mean(W_EE_m)
    matrix[2][2] = np.mean(W_EE_d)
    matrix[3][3] = np.mean(W_II_ci)
    matrix[4][4] = np.mean(W_EE_tc)
    matrix[5][5] = np.mean(W_II_tr)
    # First column - Layer S
    matrix[1][0] = np.mean(W_EE_s_m)
    matrix[2][0] = np.mean(W_EE_s_d)
    matrix[3][0] = np.mean(W_EI_s_ci)
    matrix[4][0] = np.mean(W_EE_s_tc)
    matrix[5][0] = np.mean(W_EI_s_tr)
    # Second column - Layer M
    matrix[0][1] = np.mean(W_EE_m_s)
    matrix[2][1] = np.mean(W_EE_m_d)
    matrix[3][1] = np.mean(W_EI_m_ci)
    matrix[4][1] = np.mean(W_EE_m_tc)
    matrix[5][1] = np.mean(W_EI_m_tr)
    # Thid column - Layer D
    matrix[0][2] = np.mean(W_EE_d_s)
    matrix[1][2] = np.mean(W_EE_d_m)
    matrix[3][2] = np.mean(W_EI_d_ci)
    matrix[4][2] = np.mean(W_EE_d_tc)
    matrix[5][2] = np.mean(W_EI_d_tr)
    # Fourth column - Structure CI
    matrix[0][3] = np.mean(W_IE_ci_s)
    matrix[1][3] = np.mean(W_IE_ci_m)
    matrix[2][3] = np.mean(W_IE_ci_d)
    matrix[4][3] = np.mean(W_IE_ci_tc)
    matrix[5][3] = np.mean(W_II_ci_tr)
    # Fifth column - Structure TCR
    matrix[0][4] = np.mean(W_EE_tc_s)
    matrix[1][4] = np.mean(W_EE_tc_m)
    matrix[2][4] = np.mean(W_EE_tc_d)
    matrix[3][4] = np.mean(W_EI_tc_ci)
    matrix[5][4] = np.mean(W_EI_tc_tr)
    # Sixth column - Structure TRN
    matrix[0][5] = np.mean(W_IE_tr_s)
    matrix[1][5] = np.mean(W_IE_tr_m)
    matrix[2][5] = np.mean(W_IE_tr_d)
    matrix[3][5] = np.mean(W_II_tr_ci)
    matrix[4][5] = np.mean(W_IE_tr_tc)
    
    weights = {
        'W_EE_s': W_EE_s,
        'W_EE_m': W_EE_m,
        'W_EE_d': W_EE_d,
        'W_II_ci': W_II_ci,
        'W_II_tr': W_II_tr,
        'W_EE_tc': W_EE_tc,
        'W_EE_s_m': W_EE_s_m,
        'W_EE_s_d': W_EE_s_d,
        'W_EI_s_ci': W_EI_s_ci,
        'W_EI_s_tr': W_EI_s_tr,
        'W_EE_s_tc': W_EE_s_tc,
        'W_EE_m_s': W_EE_m_s,
        'W_EE_m_d': W_EE_m_d,
        'W_EI_m_ci': W_EI_m_ci,
        'W_EI_m_tr': W_EI_m_tr,
        'W_EE_m_tc': W_EE_m_tc,
        'W_EE_d_s': W_EE_d_s,
        'W_EE_d_m': W_EE_d_m,
        'W_EI_d_ci': W_EI_d_ci,
        'W_EI_d_tr': W_EI_d_tr,
        'W_EE_d_tc': W_EE_d_tc,
        'W_IE_ic_s': W_IE_ci_s,
        'W_IE_ic_m': W_IE_ci_m,
        'W_IE_ic_d': W_IE_ci_d,
        'W_II_ic_tr': W_II_ci_tr,
        'W_IE_ic_tc': W_IE_ci_tc,
        'W_IE_tr_s': W_IE_tr_s,
        'W_IE_tr_m': W_IE_tr_m,
        'W_IE_tr_d': W_IE_tr_d,
        'W_II_tr_ic': W_II_tr_ci,
        'W_IE_tr_tc': W_IE_tr_tc,
        'W_EE_tc_s': W_EE_tc_s,
        'W_EE_tc_m': W_EE_tc_m,
        'W_EE_tc_d': W_EE_tc_d,
        'W_EI_tc_ci': W_EI_tc_ci,
        'W_EI_tc_tr': W_EI_tc_tr,
        }
    
    return { 'matrix': matrix, 'weights': weights }