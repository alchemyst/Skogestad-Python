# -*- coding: utf-8 -*-
"""
Created on Fri May 24 07:05:10 2013

@author: Simon Streicher
"""

"""
This function is similar to the MatLab step function
Models must be defined as tf objects
"""

from steptest_models_tf import G, Gd
import matplotlib.pyplot as plt
import matplotlib as ml
import numpy as np

ml.rcParams['font.family'] = 'serif'
ml.rcParams['font.serif'] = ['Cambria'] + ml.rcParams['font.serif']


def step(model, t_final=100, steps=100, initial_val=0):

    rows = np.shape(model())[0]
    columns = np.shape(model())[1]

    system = model()  # Do model calculations only once

    fig = plt.figure(1, figsize=(12, 8))
    bigax = fig.add_subplot(111)
    bigax.spines['top'].set_color('none')
    bigax.spines['bottom'].set_color('none')
    bigax.spines['left'].set_color('none')
    bigax.spines['right'].set_color('none')
    bigax.tick_params(labelcolor='grey', top='off', bottom='off',
                      left='off', right='off')
    plt.setp(bigax.get_xticklabels(), visible=False)
    plt.setp(bigax.get_yticklabels(), visible=False)

    cnt = 0
    for i in range(rows):
        for k in range(columns):
            cnt += 1
            tspace = np.linspace(0, t_final, steps)
            nulspace = np.zeros(steps)
            ax = fig.add_subplot(rows + 1, columns, cnt)
            tf = system[i, k]
            if all(tf.numerator) != 0:
                realstep = np.real(tf.step(initial_val, tspace))
                ax.plot(realstep[0], realstep[1])
            else:
                ax.plot(tspace, nulspace)
            
            if i == 0:
                xax = ax
            else:
                ax.sharex = xax
                
            if k == 0:
                yax = ax
            else:
                ax.sharey =yax
            
            if i == range(rows)[-1]:            
                plt.setp(ax.get_xticklabels(), visible=True, fontsize=10)
            else:
                plt.setp(ax.get_xticklabels(), visible=False)
                
            plt.setp(ax.get_yticklabels(), fontsize=10)
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 * 1.05 - 0.05,
                             box.width * 0.85, box.height])
            
            
#            if k == 0:            
#                plt.setp(ax.get_yticklabels(), visible=True)
#            else:
#                plt.setp(ax.get_yticklabels(), visible=False)
    box = bigax.get_position()    
    bigax.set_position([box.x0 - 0.05, box.y0 - 0.02,
                        box.width * 1.1, box.height * 1.1])        
    bigax.set_ylabel('Output magnitude')
    bigax.set_xlabel('Time (min)')   
    plt.savefig("steps.eps")
    plt.show()

step(G, 120)
