import json
import os
import matplotlib.pyplot as plt
import numpy as np


from functools import reduce
from itertools import chain



def plot(y_values, x_values, xlabel='x-axis' ,ylabel='y-axis' , name='output.png'):
    fig = plt.figure()    
    ax = plt.subplot(111)      
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)

       
    plt.plot(x_values, y_values, marker='X',color='darkblue', linewidth=2.3, markersize=6)
    plt.ylabel(ylabel)      
    plt.xlabel(xlabel)
       

    if name=='figs/Data_Scaling.png':
        width=1.95
        plt.bar([28.8], [1.070], width, color='coral', edgecolor='k', alpha=1.0, linewidth=1.8, label='CODEC \ndatabase size')
        plt.legend(loc='lower right', fontsize=19)
    
    # Show the major grid lines with dark grey lines
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    
    # Show the minor grid lines with very faint and almost transparent grey lines
    #plt.minorticks_on()
    #plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

    axes2 = plt.twinx()
    axes2.set_yticklabels([]) 

    plt.tight_layout()
    plt.savefig(name)

def plot2(y_values, x_values, xlabel='x-axis' ,ylabel='y-axis' , name='output.png'):
    fig = plt.figure()    
    ax = plt.subplot(111)      
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)

       
    plt.plot(x_values, y_values, marker='X',color='darkblue', linewidth=2.3, markersize=6)
    plt.ylabel(ylabel)      
    plt.xlabel(xlabel)
       
    
    # Show the major grid lines with dark grey lines
    plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.5)
    
    # Show the minor grid lines with very faint and almost transparent grey lines
    #plt.minorticks_on()
    #plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.1)

    width = 0.25
    plt.bar(x_values, y_values, width, color='coral', edgecolor='k', alpha=1.0, linewidth=1.8) #, yerr=menStd, label='Men means')

    ax.set_xticks([1,2,4,6,8,10,12,14,16]) #list(ax.get_xticks()) + [1])
    ax.set_yticks([0.5,1.0,1.5,2.0])
    plt.xlim(0, 17)
    plt.ylim(0, 2.249)
    plt.tight_layout()
    plt.savefig(name)


def plot3(y_values, x_values, xlabel='x-axis' ,ylabel='y-axis' , name='output.png'):
    fig = plt.figure()    
    ax = plt.subplot(111)      
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)

    y_values = np.insert(y_values, 1, y_values[0])
    x_values = np.insert(x_values, 0, x_values[0])

    freq=5
    x_values = x_values[::freq]
    y_values = y_values[::freq]
       
    plt.plot(x_values, y_values, marker='X',color='darkblue', linewidth=1.0, markersize=6, linestyle='dashed')
    plt.ylabel(ylabel)      
    plt.xlabel(xlabel)
       
    
    # Show the major grid lines with dark grey lines
    plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.5)
    
    # Show the minor grid lines with very faint and almost transparent grey lines
    #plt.minorticks_on()
    #plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.1)

    width = 1 
    plt.bar(x_values, y_values, width, color='coral', edgecolor='k', alpha=1.0, linewidth=1.8) #, yerr=menStd, label='Men means')


    y_ticks = []
    for val in y_values:
       if(val > 1000):
           y_ticks.append(str(int(val//1000))+'K')
       else:
           y_ticks.append(str(val))

    locs, labels = plt.yticks()           
    plt.yticks(locs, y_ticks)    
    ax.set_ylim(500, 32300)
    ax.set_xlim(0, 32)
    ax.set_xticks([1,5,10,15,20,25,30]) #list(ax.get_xticks()) + [1])
    plt.tight_layout()
    #ax.set_yticks([0.5,1.0,1.5,2.0])
    plt.savefig(name)

if __name__ == '__main__':

    y_values = [0.04,	0.11,	0.22,	0.33,	0.43,	0.74,	0.95,	1.39,	1.70,	2.66,	3.49] 
    x_values = [1.6,	3.2,	6.4,	9.6,	12.8,	19.2,	25.6,	38.4,	51.2,	76.8,	102.4] 

    #y_values = [val*1000 for val in y_values] 
    plot(y_values, x_values, xlabel='Data Size in Million', ylabel='Runtime in Seconds' , name='figs/Data_Scaling.png')
    
    y_values = [2.185,	1.182,	0.672,	0.403,	0.254] 
    #y_values = [val*1000 for val in y_values] 
    x_indexes = [1,2,4,8,16]
    plot2(y_values, x_indexes, xlabel='Number of GPUs', ylabel='Runtime in Seconds', name='figs/GPU_scaling.png')


    mc_iter = np.arange(30)+1
    slowdowns = [1062.619048,	2130,	3192.380952,	4246.666667,	5304.047619,	6355.47619,	7401.666667,	8445.47619,	9488.571429,	10530.47619,	11571.66667,	12613.57143,	13654.7619,	14700,	15744.7619,	16789.04762,	17834.52381,	18880.47619,	19925.47619,	20971.19048,	22017.38095,	23064.7619,	24113.33333,	25157.14286,	26201.90476,	27248.0952428295.47619,	29344.7619	,30393.09524,	31441.42857]

    plot3(slowdowns, mc_iter, xlabel='Monte Carlo Iterations', ylabel='Slowdown', name='figs/decoder_slowdown.png')
     

