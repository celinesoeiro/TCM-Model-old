# -*- coding: utf-8 -*-
"""
Created on Sat May 13 10:26:45 2023

@author: Avell
"""

from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

def print_heat_map(matrix_normal, matrix_PD): 
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