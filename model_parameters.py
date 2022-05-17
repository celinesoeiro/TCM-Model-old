"""
@author: Celine Soeiro

@description: Thalamo-Cortical microcircuit by AmirAli Farokhniaee and Madeleine M. Lowery - 2021

# Abreviations:
    PD: Parkinson Desease
    CI: Cortical Interneurons
    TC: Thalamo-Cortical Relay Nucleus (TCR)
    TR: Thalamic Reticular Nucleus (TRN)
    PD: Poissonian Distribution 
"""

import numpy as np

def TCM_model_parameters():
    simulation_time = 10
    number_trials = 1
    dt = 0.1
    fs = 1000/dt
    dbs_on = 5*67
    dbs_off = 0
    synaptic_fidelity = dbs_off
    sim_time = 10
    sim_time_ms = (sim_time + 1)*1000
    sim_steps = np.round(sim_time_ms/dt)
    td_synapse = 1
    td_thalamus_cortex = 15 # 25
    td_cortex_thalamus = 20
    hyperdirect_neurons = 0.1
    connectivity_factor_normal = 2.5
    connectivity_factor_PD = 5
    
    # Neuron quantities
    qnt_neurons_s = 100
    qnt_neurons_m = 100
    qnt_neurons_d = 100
    qnt_neurons_ci = 100
    qnt_neurons_tc = 100
    qnt_neurons_tr = 40
    
    if (td_thalamus_cortex >= td_cortex_thalamus): 
        t_vec = np.arange(td_thalamus_cortex + td_synapse + 1, sim_steps)
    else:
        t_vec = np.arange(td_cortex_thalamus + td_synapse + 1, sim_steps)
    
    neuron_quantities = {
        'qnt_neurons_s': 100,
        'qnt_neurons_m': 100,
        'qnt_neurons_d': 100,
        'qnt_neurons_ci': 100,
        'qnt_neurons_tc': 100,
        'qnt_neurons_tr': 40,
        'qnt_neurons_hyperdirect': qnt_neurons_d*hyperdirect_neurons, 
        'total': qnt_neurons_s + qnt_neurons_m + qnt_neurons_d + qnt_neurons_ci + qnt_neurons_tc + qnt_neurons_tr,
        }
    
    # Distribution of neurons in each structure
    neuron_per_structure = {
        'neurons_s_1': 0.5*neuron_quantities['qnt_neurons_s'],      # Regular Spiking
        'neurons_s_2': 0.5*neuron_quantities['qnt_neurons_s'],      # Intrinsically Bursting
        'neurons_m_1': 1*neuron_quantities['qnt_neurons_m'],        # Regular Spiking
        'neurons_m_2': 1*neuron_quantities['qnt_neurons_m'],        # Regular Spiking
        'neurons_d_1': 0.7*neuron_quantities['qnt_neurons_d'],      # Regular Spiking
        'neurons_d_2': 0.3*neuron_quantities['qnt_neurons_d'],      # Intrinsically_bursting
        'neurons_ci_1': 0.5*neuron_quantities['qnt_neurons_ci'],    # Fast spiking
        'neurons_ci_2': 0.5*neuron_quantities['qnt_neurons_ci'],    # Low threshold_spiking
        'neurons_tcr_tc_1': 0.7*neuron_quantities['qnt_neurons_tc'],# Reley
        'neurons_tcr_tc_2': 0.3*neuron_quantities['qnt_neurons_tc'],# Relay
        'neurons_tcr_tr_1': 0.5*neuron_quantities['qnt_neurons_tr'],# Reticular
        'neurons_tcr_tr_2': 0.5*neuron_quantities['qnt_neurons_tr'],# Reticular
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
        'chop_till': 1*fs, # cut the first 1s of simulation
        'time_delay_between_layers': 8,
        'time_delay_within_layers': 1,
        'time_delay_thalamus_cortex': 15,
        'time_delay_cortex_thalamus': 20,
        'transmission_delay_synapse': 1,
        'time_vector': t_vec,
        'connectivity_factor_normal_condition': connectivity_factor_normal,
        'connectivity_factor_PD_condition': connectivity_factor_PD,
        }
    
    # Impact of DBS on the other cortical structures via D PNs axons:
    synaptic_fidelity = {
        'synaps_fidelity_CI': 1*synaptic_fidelity,
        'synapse_fidelity_M': 0*synaptic_fidelity,
        'synapse_fidelity_S': 1*synaptic_fidelity,
        'synapse_fidelity_TR': 1*synaptic_fidelity,
        'synapse_fidelity_TC': 1*synaptic_fidelity,
        }
    
    # Percentage of neurons that have synaptic contact with hyperdirect neurons axon arbors
    neurons_connected_with_hyperdirect_neurons = {
        'neurons_synapse_hyperd_CI': 1*hyperdirect_neurons*qnt_neurons_ci,
        'neurons_synapse_hyperd_S': 1*hyperdirect_neurons*qnt_neurons_s,
        'neurons_synapse_hyperd_M': 0*hyperdirect_neurons*qnt_neurons_m,
        'neurons_synapse_hyperd_TR': 1*hyperdirect_neurons*qnt_neurons_tr,
        'neurons_synapse_hyperd_TC': 1*hyperdirect_neurons*qnt_neurons_tc,
        }
    
    connectivity_matrix_normal_condition = {
        ''
        }
    
    connectivity_matrix_PD_condition = {
        ''
        }
    
    # Export all dictionaries
    data = {
        'neuron_quantities': neuron_quantities,
        'neuron_per_structure': neuron_per_structure,
        'model_global_parameters': model_global_parameters,
        'synaptic_fidelity': synaptic_fidelity,
        'neurons_connected_with_hyperdirect_neurons': neurons_connected_with_hyperdirect_neurons,
        'connectivity_matrix_normal_condition': connectivity_matrix_normal_condition,
        'connectivity_matrix_PD_condition': connectivity_matrix_PD_condition,
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
    r_tc = initial + interval*np.random.rand(n_tc, 1)
    r_tr = initial + interval*np.random.rand(n_tr, 1)
    
    # =============================================================================
    #     COUPLING STRENGTHs within each structure (The same in Normal and PD)
    # EE -> Excitatory to Excitatory (?) 
    # II -> Inhibitory to Inhibitory (?)
    # =============================================================================
    ## Layer S (was -1e-2 for IEEE paper)
    aee_s = -1e1/division_factor
    W_EE_s = aee_s*r_s
    ## Layer M (was -1e-2 for IEEE paper)
    aee_m = -1e1/division_factor
    W_EE_m = aee_m*r_m
    ## Layer D (was -1e-2 for IEEE paper)
    aee_d = -1e1/division_factor
    W_EE_d = aee_d*r_d
    ## INs 
    aii_INs = -5e2/division_factor
    W_II_ci = aii_INs*r_ci
    ## Reticular cells
    aii_ret = -5e1/division_factor
    W_II_ret = aii_ret*r_tc
    ## Relay cells
    aee_rel = 0/division_factor
    W_EE_rel = aee_rel*r_tr
    
    # =============================================================================
    #     COUPLING STRENGTHs between structures
    # =============================================================================
    #     S
    # =============================================================================
    # M to S coupling
    aee_sm = 1e1/division_factor
    W_EE_s_m = aee_sm*r_s
    # D to S coupling
    aee_ds = 5e2/division_factor
    W_EE_s_d = aee_ds*r_s
    # CI (INs) to S coupling
    aei_sINs = -5e2/division_factor
    W_EI_s_ci = aei_sINs*r_s
    # Reticular to S coupling
    aei_sRet = 0/division_factor      
    W_EI_s_ret = aei_sRet*r_s
    # Rel. to S couplings
    aee_sRel = 0/division_factor      
    W_EE_s_rel = aee_sRel*r_s         
    # =============================================================================
    #     M
    # =============================================================================
    # S to M
    aee_ms = 3e2/facilitating_factor      
    W_EE_m_s = aee_ms*r_m             
    # D to M couplings
    aee_md = 0/facilitating_factor        
    W_EE_m_d = aee_md*r_m             
    # INs to M couplings
    aei_mINs = -3e2/facilitating_factor    
    W_EI_m_ci = aei_mINs*r_m         
    # Ret. to M couplings    
    aei_mRet = 0/facilitating_factor      
    W_EI_m_ret = aei_mRet*r_m      
    # Rel. to M couplings
    aee_mRel = 0/facilitating_factor      
    W_EE_m_rel = aee_mRel*r_m
    # =============================================================================
    #     D
    # =============================================================================
    # S to D couplings
    aee_ds = 3e2/facilitating_factor      
    W_EE_d_s = aee_ds*r_d
    # M to D couplings
    aee_dm = 0/facilitating_factor        
    W_EE_d_m = aee_dm*r_d
    # INs to D couplings
    aei_dINs = -7.5e3/facilitating_factor   
    W_EI_d_ci = aei_dINs*r_d
    # Ret. to D couplings
    aei_dRet = 0/facilitating_factor      
    W_EI_d_ret = aei_dRet*r_d
    # Rel. to D couplings
    aee_dRel = 1e1/facilitating_factor    
    W_EE_d_rel = aee_dRel*r_d
    # =============================================================================
    #     INs (CI)
    # =============================================================================
    # S to INs couplings
    aie_inss = 2e2/facilitating_factor    
    W_IE_ci_s = aie_inss*r_ci
    # M to INs couplings
    aie_insm = 2e2/facilitating_factor    
    W_IE_ci_m = aie_insm*r_ci
    # D to INs couplings
    aie_insd = 2e2/facilitating_factor    
    W_IE_ci_d = aie_insd*r_ci
    # Ret. to INs couplings
    aii_InsRet = 0/facilitating_factor    
    W_II_ci_ret = aii_InsRet*r_ci
    # Rel. to INs couplings
    aie_InsRel = 1e1/facilitating_factor  
    W_IE_ci_rel = aie_InsRel*r_ci
    # =============================================================================
    #     Reticular
    # =============================================================================
    # S to Ret couplings
    aie_rets = 0/facilitating_factor      
    W_IE_ret_s = aie_rets*r_tc     
    # M to Ret couplings
    aie_retm = 0/facilitating_factor      
    W_IE_ret_m = aie_retm*r_tc     
    # D to Ret couplings
    aie_retd = 7e2/facilitating_factor   
    W_IE_ret_d = aie_retd*r_tc;     
    # Ret. Ret INs couplings
    aii_RetIns = 0/facilitating_factor    
    W_II_ret_ci = aii_RetIns*r_tc 
    # Rel. Ret INs couplings
    aie_RetRel = 1e3/facilitating_factor  
    W_IE_ret_rel = aie_RetRel*r_tc 
    # =============================================================================
    #     Rele
    # =============================================================================
    # S to Rel couplings
    aee_rels = 0/facilitating_factor     
    W_EE_rel_s = aee_rels*r_tr       
    # M to Rel couplings
    aee_relm = 0/facilitating_factor      
    W_EE_rel_m = aee_relm*r_tr 	    
    # D to Rel couplings
    aee_reld = 7e2/facilitating_factor
    W_EE_rel_d = aee_reld*r_tr       
    # INs to Rel couplings
    aei_RelINs = 0/facilitating_factor    
    W_EI_rel_ci = aei_RelINs*r_tr   
    # Ret to Rel couplings
    aei_RelRet = -5e2/facilitating_factor 
    W_EI_rel_ret = aei_RelRet*r_tr   
    
    # Initialize matrix (6 structures -> 6x6 matrix)
    matrix = np.zeros((6,6))
    
    # Populating the matrix
    # Main Diagonal
    matrix[0][0] = np.mean(W_EE_s)
    matrix[1][1] = np.mean(W_EE_m)
    matrix[2][2] = np.mean(W_EE_d)
    matrix[3][3] = np.mean(W_II_ci)
    matrix[4][4] = np.mean(W_EE_rel)
    matrix[5][5] = np.mean(W_II_ret)
    # First row - Layer S
    matrix[0][1] = np.mean(W_EE_s_m)
    matrix[0][2] = np.mean(W_EE_s_d)
    matrix[0][3] = np.mean(W_EI_s_ci)
    matrix[0][4] = np.mean(W_EE_s_rel)
    matrix[0][5] = np.mean(W_EI_s_ret)
    # Second row - Layer M
    matrix[1][0] = np.mean(W_EE_m_s)
    matrix[1][2] = np.mean(W_EE_m_d)
    matrix[1][3] = np.mean(W_EI_m_ci)
    matrix[1][4] = np.mean(W_EE_m_rel)
    matrix[1][5] = np.mean(W_EI_m_ret)
    # Thid row - Layer D
    matrix[2][0] = np.mean(W_EE_d_s)
    matrix[2][1] = np.mean(W_EE_d_m)
    matrix[2][3] = np.mean(W_EI_d_ci)
    matrix[2][4] = np.mean(W_EE_d_rel)
    matrix[2][5] = np.mean(W_EI_d_ret)
    # Fourth row - Structure CI
    matrix[3][0] = np.mean(W_IE_ci_s)
    matrix[3][1] = np.mean(W_IE_ci_m)
    matrix[3][2] = np.mean(W_IE_ci_d)
    matrix[3][4] = np.mean(W_IE_ci_rel)
    matrix[3][5] = np.mean(W_II_ci_ret)
    # Fifth row - Structure TCR
    matrix[4][0] = np.mean(W_EE_rel_s)
    matrix[4][1] = np.mean(W_EE_rel_m)
    matrix[4][2] = np.mean(W_EE_rel_d)
    matrix[4][3] = np.mean(W_EI_rel_ci)
    matrix[4][5] = np.mean(W_EI_rel_ret)
    # Sixth row - Structure TRN
    matrix[5][0] = np.mean(W_IE_ret_s)
    matrix[5][1] = np.mean(W_IE_ret_m)
    matrix[5][2] = np.mean(W_IE_ret_d)
    matrix[5][3] = np.mean(W_II_ret_ci)
    matrix[5][4] = np.mean(W_IE_ret_rel)
    
    weights = {
        'W_EE_s': W_EE_s,
        'W_EE_m': W_EE_m,
        'W_EE_d': W_EE_d,
        'W_II_ins': W_II_ci,
        'W_II_ret': W_II_ret,
        'W_EE_rel': W_EE_rel,
        'W_EE_s_m': W_EE_s_m,
        'W_EE_s_d': W_EE_s_d,
        'W_EI_s_ci': W_EI_s_ci,
        'W_EI_s_ret': W_EI_s_ret,
        'W_EE_s_rel': W_EE_s_rel,
        'W_EE_m_s': W_EE_m_s,
        'W_EE_m_d': W_EE_m_d,
        'W_EI_m_ci': W_EI_m_ci,
        'W_EI_m_ret': W_EI_m_ret,
        'W_EE_m_rel': W_EE_m_rel,
        'W_EE_d_s': W_EE_d_s,
        'W_EE_d_m': W_EE_d_m,
        'W_EI_d_ci': W_EI_d_ci,
        'W_EI_d_ret': W_EI_d_ret,
        'W_EE_d_rel': W_EE_d_rel,
        'W_IE_ci_s': W_IE_ci_s,
        'W_IE_ci_m': W_IE_ci_m,
        'W_IE_ci_d': W_IE_ci_d,
        'W_II_ci_Ret': W_II_ci_ret,
        'W_IE_ci_Rel': W_IE_ci_rel,
        'W_EE_rel_s': W_EE_rel_s,
        'W_EE_rel_m': W_EE_rel_m,
        'W_EE_rel_d': W_EE_rel_d,
        'W_EI_rel_ci': W_EI_rel_ci,
        'W_EI_rel_ret': W_EI_rel_ret,
        'W_IE_ret_s': W_IE_ret_s,
        'W_IE_ret_m': W_IE_ret_m,
        'W_IE_ret_d': W_IE_ret_d,
        'W_II_ret_ci': W_II_ret_ci,
        'W_IE_ret_rel': W_IE_ret_rel,
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
    # EE -> Excitatory to Excitatory (?) 
    # II -> Inhibitory to Inhibitory (?)
    # =============================================================================
    ## Layer S (was -1e-2 for IEEE paper)
    aee_s = -1e1/division_factor
    W_EE_s = aee_s*r_s
    ## Layer M (was -1e-2 for IEEE paper)
    aee_m = -1e1/division_factor
    W_EE_m = aee_m*r_m
    ## Layer D (was -1e-2 for IEEE paper)
    aee_d = -1e1/division_factor
    W_EE_d = aee_d*r_d
    ## INs 
    aii_INs = -5e2/division_factor
    W_II_ci = aii_INs*r_ci
    ## Reticular cells
    aii_ret = -5e1/division_factor
    W_II_ret = aii_ret*r_tc
    ## Relay cells
    aee_rel = 0/division_factor
    W_EE_rel = aee_rel*r_tr
    
    # =============================================================================
    #     COUPLING STRENGTHs between structures
    # =============================================================================
    #     S
    # =============================================================================
    # M to S couplings
    aee_sm = 3e2/division_factor  
    W_EE_s_m = aee_sm*r_s             
    # D to S couplings
    aee_sd = 5e2/division_factor      
    W_EE_s_d = aee_sd*r_s          
    # INs to S couplings
    aei_sINs = -7.5e2/division_factor    
    W_EI_s_ci = aei_sINs*r_s         
    # Ret. to S couplings
    aei_sRet = 0/division_factor      
    W_EI_s_ret = aei_sRet*r_s         
    # Rel. to S couplings
    aee_sRel = 0/division_factor    
    W_EE_s_rel = aee_sRel*r_s         
    # =============================================================================
    #     M
    # =============================================================================
    # S to M couplings
    aee_ms = 1e1/division_factor    
    W_EE_m_s = aee_ms*r_m            
    # D to M couplings
    aee_md = 0/division_factor        
    W_EE_m_d = aee_md*r_m            
    # INs to M couplings
    aei_mINs = -7.5e2/division_factor    
    W_EI_m_ci = aei_mINs*r_m      
    # Ret. to M couplings
    aei_mRet = 0/division_factor      
    W_EI_m_ret = aei_mRet*r_m         
    # Rel. to M couplings
    aee_mRel = 0/division_factor      
    W_EE_m_rel = aee_mRel*r_m         
    # =============================================================================
    #     D
    # =============================================================================
    # M to S couplings
    aee_ds = 3e2/facilitating_factor    
    W_EE_d_s = aee_ds*r_d             
    # M to D couplings
    aee_dm = 0/facilitating_factor    
    W_EE_d_m = aee_dm*r_d          
    # INs to D couplings
    aei_dINs = -5e3/facilitating_factor   
    W_EI_d_ci = aei_dINs*r_d         
    # Ret. to D couplings
    aei_dRet = 0/facilitating_factor
    W_EI_d_ret = aei_dRet*r_d
    # Rel. to D couplings
    aee_dRel = 1e3/facilitating_factor
    W_EE_d_rel = aee_dRel*r_d         
    # =============================================================================
    #     INs
    # =============================================================================
    # S to INs couplings
    aie_inss = 2e2/facilitating_factor    
    W_IE_ci_s = aie_inss*r_ci
    # M to INs couplings
    aie_insm = 2e2/facilitating_factor    
    W_IE_ci_m = aie_insm*r_ci     
    # D to INs couplings
    aie_insd = 2e2/facilitating_factor    
    W_IE_ci_d = aie_insd*r_ci  
    # Ret. to INs couplings
    aii_InsRet = 0/facilitating_factor   
    W_II_ci_ret = aii_InsRet*r_ci 
    # Rel. to INs couplings
    aie_InsRel = 1e3/facilitating_factor  
    W_IE_ci_rel = aie_InsRel*r_ci
    # =============================================================================
    #     Reticular
    # =============================================================================
    # S to Ret couplings
    aie_rets = 0/facilitating_factor      
    W_IE_ret_s = aie_rets*r_tc     
    # M to Ret couplings
    aie_retm = 0/facilitating_factor     
    W_IE_ret_m = aie_retm*r_tc     
    # D to Ret couplings
    aie_retd = 1e2/facilitating_factor    
    W_IE_ret_d = aie_retd*r_tc     
    # Ret. Ret INs couplings
    aii_RetIns = 0/facilitating_factor
    W_II_ret_ci = aii_RetIns*r_tc 
    # Rel. Ret INs couplings
    aie_RetRel = 5e2/facilitating_factor  
    W_IE_ret_rel = aie_RetRel*r_tc 
    # =============================================================================
    #     Rele
    # =============================================================================
    # S to Rel couplings
    aee_rels = 0/facilitating_factor      
    W_EE_rel_s = aee_rels*r_tr       
    # M to Rel couplings
    aee_relm = 0/facilitating_factor      
    W_EE_rel_m = aee_relm*r_tr 	    
    # D to Rel couplings
    aee_reld = 1e2/facilitating_factor    
    W_EE_rel_d = aee_reld*r_tr       
    # INs to Rel couplings
    aei_RelINs = 0/facilitating_factor
    W_EI_rel_ci = aei_RelINs*r_tr   
    # Ret to Rel couplings
    aei_RelRet = -2.5*1e3/facilitating_factor
    W_EI_rel_ret = aei_RelRet*r_tr   
    
    # Initialize matrix (6 structures -> 6x6 matrix)
    matrix = np.zeros((6,6))
    
    # Populating the matrix
    # Main Diagonal
    matrix[0][0] = np.mean(W_EE_s)
    matrix[1][1] = np.mean(W_EE_m)
    matrix[2][2] = np.mean(W_EE_d)
    matrix[3][3] = np.mean(W_II_ci)
    matrix[4][4] = np.mean(W_EE_rel)
    matrix[5][5] = np.mean(W_II_ret)
    # First row - Layer S
    matrix[0][1] = np.mean(W_EE_s_m)
    matrix[0][2] = np.mean(W_EE_s_d)
    matrix[0][3] = np.mean(W_EI_s_ci)
    matrix[0][4] = np.mean(W_EE_s_rel)
    matrix[0][5] = np.mean(W_EI_s_ret)
    # Second row - Layer M
    matrix[1][0] = np.mean(W_EE_m_s)
    matrix[1][2] = np.mean(W_EE_m_d)
    matrix[1][3] = np.mean(W_EI_m_ci)
    matrix[1][4] = np.mean(W_EE_m_rel)
    matrix[1][5] = np.mean(W_EI_m_ret)
    # Thid row - Layer D
    matrix[2][0] = np.mean(W_EE_d_s)
    matrix[2][1] = np.mean(W_EE_d_m)
    matrix[2][3] = np.mean(W_EI_d_ci)
    matrix[2][4] = np.mean(W_EE_d_rel)
    matrix[2][5] = np.mean(W_EI_d_ret)
    # Fourth row - Structure CI
    matrix[3][0] = np.mean(W_IE_ci_s)
    matrix[3][1] = np.mean(W_IE_ci_m)
    matrix[3][2] = np.mean(W_IE_ci_d)
    matrix[3][4] = np.mean(W_IE_ci_rel)
    matrix[3][5] = np.mean(W_II_ci_ret)
    # Fifth row - Structure TCR
    matrix[4][0] = np.mean(W_EE_rel_s)
    matrix[4][1] = np.mean(W_EE_rel_m)
    matrix[4][2] = np.mean(W_EE_rel_d)
    matrix[4][3] = np.mean(W_EI_rel_ci)
    matrix[4][5] = np.mean(W_EI_rel_ret)
    # Sixth row - Structure TRN
    matrix[5][0] = np.mean(W_IE_ret_s)
    matrix[5][1] = np.mean(W_IE_ret_m)
    matrix[5][2] = np.mean(W_IE_ret_d)
    matrix[5][3] = np.mean(W_II_ret_ci)
    matrix[5][4] = np.mean(W_IE_ret_rel)
    
    weights = {
        'W_EE_s': W_EE_s,
        'W_EE_m': W_EE_m,
        'W_EE_d': W_EE_d,
        'W_II_ci': W_II_ci,
        'W_II_ret': W_II_ret,
        'W_EE_rel': W_EE_rel,
        'W_EE_s_m': W_EE_s_m,
        'W_EE_s_d': W_EE_s_d,
        'W_EI_s_ci': W_EI_s_ci,
        'W_EI_s_ret': W_EI_s_ret,
        'W_EE_s_rel': W_EE_s_rel,
        'W_EE_m_s': W_EE_m_s,
        'W_EE_m_d': W_EE_m_d,
        'W_EI_m_ci': W_EI_m_ci,
        'W_EI_m_ret': W_EI_m_ret,
        'W_EE_m_rel': W_EE_m_rel,
        'W_EE_d_s': W_EE_d_s,
        'W_EE_d_m': W_EE_d_m,
        'W_EI_d_ci': W_EI_d_ci,
        'W_EI_d_ret': W_EI_d_ret,
        'W_EE_d_rel': W_EE_d_rel,
        'W_IE_ci_s': W_IE_ci_s,
        'W_IE_ci_m': W_IE_ci_m,
        'W_IE_ci_d': W_IE_ci_d,
        'W_II_ci_ret': W_II_ci_ret,
        'W_IE_ci_rel': W_IE_ci_rel,
        'W_IE_ret_s': W_IE_ret_s,
        'W_IE_ret_m': W_IE_ret_m,
        'W_IE_ret_d': W_IE_ret_d,
        'W_II_ret_ci': W_II_ret_ci,
        'W_IE_ret_rel': W_IE_ret_rel,
        'W_EE_rel_s': W_EE_rel_s,
        'W_EE_rel_m': W_EE_rel_m,
        'W_EE_rel_d': W_EE_rel_d,
        'W_EI_rel_ci': W_EI_rel_ci,
        'W_EI_rel_ret': W_EI_rel_ret,
        }
    
    return { 'matrix': matrix, 'weights': weights }