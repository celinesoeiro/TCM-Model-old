"""
@author: Avell
@description: Comparing values with original simulation
"""

import numpy as np
import csv

from model_plots import plot_raster

file_D = r"C:\Users\Avell\Desktop\masters\vEd.dat"
file_M = r"C:\Users\Avell\Desktop\masters\vEm.dat"
file_S = r"C:\Users\Avell\Desktop\masters\vEs.dat"
file_CI = r"C:\Users\Avell\Desktop\masters\vIs.dat"
file_TC = r"C:\Users\Avell\Desktop\masters\vRel.dat"
file_TR = r"C:\Users\Avell\Desktop\masters\vRet.dat"

file_zeta_D = r"C:\Users\Avell\Desktop\masters\zetaDE.dat"
file_zeta_M = r"C:\Users\Avell\Desktop\masters\zetaME.dat"
file_zeta_S = r"C:\Users\Avell\Desktop\masters\zetaSE.dat"
file_zeta_CI = r"C:\Users\Avell\Desktop\masters\zetaSI.dat"
file_zeta_TC = r"C:\Users\Avell\Desktop\masters\zetaErel.dat"
file_zeta_TR = r"C:\Users\Avell\Desktop\masters\zetaIret.dat"

dbs = 335
sim_steps = 14000 
sim_time = 6
dt = 0.5
chop_till = 2000
n_TR = 20
n_TC = 50
n_CI = 50
n_D = 50
n_M = 50
n_S = 50
n_total = 270
n_CI_FS = 25
n_CI_LTS = 25
n_D_RS = 35
n_D_IB = 15
n_S_RS = 25
n_S_IB = 25
vp = 30

def handle_import(path, rows, is_noise):
    data = []
    with open(path, 'r') as file:
        csv_reader = csv.reader(file, delimiter=' ')
        
        for row in csv_reader:
            data.append(row)
            
    for d in data:
        d.pop(0)
        if (is_noise == False):
            d.pop(1)

    return data

def handle_format(data):  
    arr_rows = int(len(data))
    arr_cols = int(len(data[0]))
    arr = np.zeros((arr_rows, arr_cols))
    
    row_counter = 0
    for row in data:
        for j in range(len(row)):
            decimal = float(row[j])
            arr[row_counter][j] = '{:.2f}'.format(decimal)
        row_counter += 1
        
    return arr

def create_spikes(arr, vp, noise):
    arr_rows = int(len(arr[:,0]))
    arr_cols = int(len(arr[0]))
    spikes = np.zeros((arr_rows, arr_cols))
        
    for i in range(arr_rows):
        for j in range(arr_cols):
            aux = arr[i][j]
            if (aux >= vp + noise[i][j]):
                aux = vp + noise[i][j]
                spikes[i][j] = j
                
    return spikes

o_v_TR = handle_import(file_TR, n_TR, False)
o_v_TC = handle_import(file_TC, n_TC, False)
o_v_CI = handle_import(file_CI, n_CI, False)
o_v_D = handle_import(file_D, n_D, False)
o_v_M = handle_import(file_M, n_M, False)
o_v_S = handle_import(file_S, n_S, False)  

o_zeta_TR = handle_import(file_zeta_TR, n_TR, True)
o_zeta_TC = handle_import(file_zeta_TC, n_TC, True)
o_zeta_CI = handle_import(file_zeta_CI, n_CI, True)
o_zeta_D = handle_import(file_zeta_D, n_D, True)
o_zeta_M = handle_import(file_zeta_M, n_M, True)
o_zeta_S = handle_import(file_zeta_S, n_S, True)

o_v_TR = handle_format(o_v_TR)
o_v_TC = handle_format(o_v_TC)
o_v_CI = handle_format(o_v_CI)
o_v_D = handle_format(o_v_D)
o_v_M = handle_format(o_v_M)
o_v_S = handle_format(o_v_S)

o_zeta_TR = handle_format(o_zeta_TR)
o_zeta_TC = handle_format(o_zeta_TC)
o_zeta_CI = handle_format(o_zeta_CI)
o_zeta_D = handle_format(o_zeta_D)
o_zeta_M = handle_format(o_zeta_M)
o_zeta_S = handle_format(o_zeta_S)

o_spikes_TR = create_spikes(o_v_TR, vp, o_zeta_TR)
o_spikes_TC = create_spikes(o_v_TC, vp, o_zeta_TC)
o_spikes_CI = create_spikes(o_v_CI, vp, o_zeta_CI)
o_spikes_D = create_spikes(o_v_D, vp, o_zeta_D)
o_spikes_M = create_spikes(o_v_M, vp, o_zeta_M)
o_spikes_S = create_spikes(o_v_S, vp, o_zeta_S)

plot_raster(
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
    spike_times_TR = o_spikes_TR, 
    spike_times_TC = o_spikes_TC, 
    spike_times_CI = o_spikes_CI, 
    spike_times_D = o_spikes_D, 
    spike_times_M = o_spikes_M,
    spike_times_S = o_spikes_S
    )
