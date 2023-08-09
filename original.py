"""
@author: Avell
@description: Comparing values with original simulation
"""

# =============================================================================
# INPUTING MEMBRANE VOLTAGES
# =============================================================================
import numpy as np
import csv

from matplotlib import pyplot as plt
import seaborn as sns
sns.set()


file_D = r"C:\Users\Avell\Desktop\masters\vEd.dat"
file_M = r"C:\Users\Avell\Desktop\masters\vEm.dat"
file_S = r"C:\Users\Avell\Desktop\masters\vEs.dat"
file_CI = r"C:\Users\Avell\Desktop\masters\vIs.dat"
file_TC = r"C:\Users\Avell\Desktop\masters\vRel.dat"
file_TR = r"C:\Users\Avell\Desktop\masters\vRet.dat"

    
def handle_import(path, rows):
    data = []
    # Open and read the file using the csv.reader
    with open(path, 'r') as file:
        csv_reader = csv.reader(file, delimiter=' ')
        
        # Iterate through the rows and print them
        for row in csv_reader:
            data.append(row)
            
    for d in data:
        d.pop(0)

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

def create_spikes(arr, vp):
    arr_rows = int(len(arr[:,0]))
    arr_cols = int(len(arr[0]))
    spikes = np.zeros((arr_rows, arr_cols))
        
    for i in range(arr_rows):
        for j in range(arr_cols):
            if (arr[i][j] >= vp):
                spikes[i][j] = j
                
    return spikes

o_v_TR = handle_import(file_TR, 20)
o_v_TC = handle_import(file_TC, 50)
o_v_CI = handle_import(file_CI, 50)
o_v_D = handle_import(file_D, 50)
o_v_M = handle_import(file_M, 50)
o_v_S = handle_import(file_S, 50)    

o_v_TR = handle_format(o_v_TR)
o_v_TC = handle_format(o_v_TC)
o_v_CI = handle_format(o_v_CI)
o_v_D = handle_format(o_v_D)
o_v_M = handle_format(o_v_M)
o_v_S = handle_format(o_v_S)

o_spikes_TR = create_spikes(o_v_TR, 30)
o_spikes_TC = create_spikes(o_v_TC, 30)
o_spikes_CI = create_spikes(o_v_CI, 30)
o_spikes_D = create_spikes(o_v_D, 30)
o_spikes_M = create_spikes(o_v_M, 30)
o_spikes_S = create_spikes(o_v_S, 30)



TR_lim = 20
TC_lim = TR_lim + 50
CI_lim = TC_lim + 50
CI_FS_lim = CI_lim - 25
D_lim = CI_lim + 50
D_RS_lim = D_lim - 15
M_lim = D_lim + 50
S_lim = M_lim + 50
S_RS_lim = S_lim - 25

spikes = np.concatenate([o_spikes_TR, o_spikes_TC, o_spikes_CI, o_spikes_D, o_spikes_M, o_spikes_S])

fig, ax1 = plt.subplots(figsize=(10, 10))
fig.canvas.manager.set_window_title('Raster plot')
fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)
    
plt.title('Raster plot')

# Add a horizontal grid to the plot, but make it very light in color
# so we can use it for reading data values but not be distracting
ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)

ax1.set(
    axisbelow=True,  # Hide the grid behind plot objects
    title='Raster plot',
    xlabel='Time (s)',
    ylabel='Neurons',
)
    
for i in range(270):  
    y_values = np.full_like(spikes[i], i + 1)
    ax1.scatter(x=spikes[i], y=y_values, color='black', s=0.5)
    
ax1.set_ylim(1, 270 + 1)
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
                    'M - RS', 
                    'S - RS', 
                    'S - IB', 
                    'S',
                    ])

# For dt = 0.1
sim_steps = 8000
multiplier = 1000
lim_down = 2000
lim_up = 8000 + multiplier*0.5
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

