'''
Created on 21 Mar 2013

@author: St Elmo Wilken
'''

import control as cn
import numpy as np
import matplotlib.pyplot as plt

def peaks(sys):
    """
    Returns the H infinity norms and the corresponding frequencies for
    the sensitivity and complementary sensitivity functions.
    Currently works for SISO only.
    
    Parameter: sys => GK (loop transfer function) in tf format
    Returns:   2 tuples of the format ('Type', peak, peak freq)
    """
    S = 1/(1+sys)
    T = sys/(1+sys)
    fg = plt.figure()
    mag_S, phase_S, omega_S = cn.bode_plot(S)
    mag_T, phase_T, omega_T = cn.bode_plot(T)
    plt.close(fg) #so that this does not interfere with your current plotting ;)
    pos_S = np.argmax(mag_S)
    pos_T = np.argmin(mag_T)
    s_data = ("Peak |S|", mag_S[pos_S], phase_S[pos_S])
    t_data = ("Peak |T|", mag_T[pos_T], phase_T[pos_T])
    return s_data, t_data

def plot_slope(sys, *args, **dict):
    """
    Plots the slope (in dB) of the transfer function parameter (from a bode diagram).
    It also plots the position of the cross over frequency as a black vertical line
    (for slope comparisons). 
    Currently works for SISO only.
    
    Parameter: sys => a transfer function object
               *args, **dict => the usual plotting parameters
    Returns:   a matplotlib figure contraining the slope as a function of frequency
    
    Notes: This function assumes you input the loop transfer function (L(s)).
    As such it will show you where the cross over frequency is so that you may
    compare slopes against it.
    """

    mag, phase, omega = cn.bode_plot(sys) #get bode information 
    plt.clf() #clear the previous bode from the figure
    end = len(mag)    
    slope = []
    freqs = []
    
    for x in range(end-1): #calculate the slope incrementally
        slope.append( (np.log(mag[x+1])- np.log(mag[x])) /( np.log(omega[x+1])-np.log(omega[x])) )
        freqs.append((omega[x+1]+omega[x])/2)
    
    #w = cross_over_freq(sys) # something is throwing an error but this returns just wp
    gm, pm, wg, wp = cn.margin(sys)
    length = len(slope)
    cross_freqs = [wp]*length
    
    plt.plot(cross_freqs, slope, color="black", linewidth = 3.0)
    plt.plot(freqs, slope, *args, **dict)
    plt.plot()
    plt.xscale('log')
    plt.xlabel("Frequency")
    plt.ylabel("Logarithmic Slope")
    plt.grid()
    current_fig = plt.gcf()
    return current_fig
   
def cross_over_freq(sys, tol=0.05):
    """
    Only returns the cross over frequency for a transfer function.
    The tf may cross from above or below... 
    Very similar to margin due to a programming glitch on my part...
    
    
    Parameter: sys => Transfer function like object
               tol => tolerance (will increase by 20 % for each cycle of searching)
    Returns:   cross_over_freq => the cross over frequency i.e. where mag ~ 1
    
    Problems: It raises an exception when I try to close the figure fig...
    """

    fig = plt.figure()
    mag, phase, omega = cn.bode_plot(sys)
    plt.close(fig) #python doesn't like this at all... Help?

    
    cross_over_dist = np.abs(1-mag)
    # there is a more elegant solution using argmin but that doesn't guarantee that you
    # return the first possible match...
    index = -1
    flag = True
    while flag:
        for x in range(len(cross_over_dist)):
            if cross_over_dist[x] < tol:
                index = x 
                break
        if index != -1:
            flag = False
        else:
            tol = 1.2*tol
    
    
    return omega[index]
    
def weight_function(option):
    """This should return a weight function as a transfer function.
    """
    pass
 
    