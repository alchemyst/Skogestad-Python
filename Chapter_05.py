import numpy as np
import matplotlib.pyplot as plt

from utils import tf, margins


s = tf([1, 0], 1)

# Example plant based on Example 2.9 and Example 2.16
G = (s + 200) / ((10 * s + 1) * (0.05 * s + 1)**2)
Gd = 33 / (10 * s + 1)
K = 0.4 * ((s + 2) / s) * (0.075 * s + 1)
R = 3.0
wr = 10
Gm = 1 #Measurement model

''' 
All of the below function are from pages 206-207 and the associated rules that 
analyse the controllability of SISO problems
'''


def rule1(G, Gd, K=1, message=False, plot=False, w1=-4, w2=2):
    '''
    This is rule one of chapter five
    
    Calculates the speed of response to reject distrurbances. Condition require
    |S(jw)| <= |1/Gd(jw)|
    
    Parameters
    ----------
    G : tf
        plant model   
    
    Gd : tf
        plant distrubance model
        
    K : tf
        control model
    
    message : boolean 
        show the rule message (optional)
        
    plot : boolean
        show the bode plot with constraints (optional)
        
    w1 : integer
        start frequency, 10^w1 (optional)        
    
    w2 : integer
        end frequency, 10^w2 (optional)        
    
    Returns
    -------
    valid1 : boolean
        value if rule conditions was met
    
    wc : real
        crossover frequency where | G(jwc) | = 1
    
    wd : real
        crossover frequency where | Gd(jwd) | = 1         
    '''

    _,_,wc,_ = margins(G)
    _,_,wd,_  = margins(Gd)
    
    valid1 = wc > wd

    if message:
        print 'Rule 1: Speed of response to reject distrubances'
        if valid1:
            print 'First condition met, wc > wd'               
        else: 
            print 'First condition no met, wc < wd'
        print 'Seconds conditions requires |S(jw)| <= |1/Gd(jw)|'
        
    if plot:
        plt.figure('Rule 1')
        
        w = np.logspace(w1, w2, 1000)
        s = 1j * w   
        
        S = 1 / (1 + G*K)
        mag_s = np.abs(S(s))        
                    
        inv_gd = 1 / Gd        
        mag_i = np.abs(inv_gd(s))
        
        plt.loglog(w, mag_s)
        plt.loglog(w, mag_i, ls = '--')
        plt.legend(['|S|', '1/|Gd|'],
                   bbox_to_anchor=(0, 1.01, 1, 0), loc=3, ncol=3)
        plt.grid()
        plt.xlabel('Frequency [rad/s]')
        plt.ylabel('Magnitude')  
        plt.show()
        
    return valid1, wc, wd

#rule1(G, Gd, K, True, True)


def rule2(G, R, K, wr, message=False, plot=False, w1=-4, w2=2):
    '''
    This is rule two of chapter five
    
    Calculates speed of response to track reference changes. Conditions require
    |S(jw)| <= 1/R
    
    Parameters
    ----------
    G : tf
        plant model   
        
    R : real
        reference change 
        
    K : tf
        control model  
    
    wr : real
        reference frequency where tracking is required
    
    message : boolean 
        show the rule message (optional)
        
    plot : boolean
        show the bode plot with constraints (optional)
        
    w1 : integer
        start frequency, 10^w1 (optional)        
    
    w2 : integer
        end frequency, 10^w2 (optional)   
    
    Returns
    -------
    invref : real
       1 / R                
                             
    '''
    
    invref = 1/R   
    
    if message:
        print 'Conditions requires |S(jw)| <= ', invref
        
    if plot:
        plt.figure('Rule 2')
        
        w = np.logspace(w1, w2, 1000)
        s = 1j * w   
    
        S = 1 / (1 + G*K)
        mag_s = np.abs(S(s)) 
        
        plt.loglog(w, mag_s)
        plt.loglog(w,  invref * np.ones(len(w)), ls = '--')
        plt.loglog(wr * np.ones(2), [np.max(mag_s), np.min(mag_s)], ls=':')
        plt.legend(['|S|', '1/R', '$w_r$'],
                   bbox_to_anchor=(0, 1.01, 1, 0), loc=3, ncol=3)
        plt.grid()
        plt.xlabel('Frequency [rad/s]')
        plt.ylabel('Magnitude')
        plt.show()
    
    return invref

#rule2(G, R, K, 50, True, True)


def rule3(G, Gd, message=False, w1=-4, w2=2):
    '''
    This is rule three of chapter five
    
    Calculates input constraints arising from disturbance rejection
    
    Acceptable control conditions require |G(jw)| > |Gd(jw)| - 1'
    at frequencies where |Gd(jw) > 1|'
    
    Perfect control conditions require |G(jw)| > |Gd(jw)|'    
    
    Parameters
    ----------
    G : tf
        plant model   
    
    Gd : tf
        plant distrubance model
    
    message : boolean 
        show the rule message (optional)
        
    w1 : integer
        start frequency, 10^w1 (optional)        
    
    w2 : integer
        end frequency, 10^w2 (optional) 
                
    '''
        
    w = np.logspace(w1, w2, 1000)        
    s = 1j * w
        
    mag_g = np.abs(G(s))
    mag_gd = np.abs(Gd(s))
    
    if message:
        print 'Acceptable control conditions require |G(jw)| > |Gd(jw)| - 1 at frequencies where |Gd(jw) > 1|'            
        print 'Perfect control conditions require |G(jw)| > |Gd(jw)|'
        
    plt.figure('Rule 3')
    plt.subplot(211)
    plt.title('Acceptable control')
    plt.loglog(w, mag_g, label = '|G|')
    plt.loglog(w, mag_gd - 1, label = '|Gd - 1|', ls = '--')
    plt.loglog(w,  1 * np.ones(len(w)))
    plt.legend(loc = 3)
    plt.grid()
    plt.ylabel('Magnitude')
    
    plt.subplot(212)
    plt.title('Perfect control')
    plt.loglog(w, mag_g, label = '|G|')
    plt.loglog(w, mag_gd, label = '|Gd|', ls = '--')
    plt.legend(loc = 3)
    plt.grid()
    plt.xlabel('Frequency [rad/s]')
    plt.ylabel('Magnitude')
    plt.show()    

#rule3(G, Gd)


def rule4(G, R, wr, w1=-4, w2=2):
    '''
    This is rule four of chapter five
    
    Calculates input constraints arising from setpoints
    
    Conditions require |G(jw)| > R - 1 up to frequency wr
    
    Parameters
    ----------
    G : tf
        plant model  
        
    R : real
        reference change  
    
    wr : real
        reference frequency where tracking is required
        
    w1 : integer
        start frequency, 10^w1 (optional)        
    
    w2 : integer
        end frequency, 10^w2 (optional)     
                  
    '''

    w = np.logspace(w1, w2, 1000)
    s = 1j * w
    
    mag_g = np.abs(G(s)) 
    mag_rr = (R - 1) * np.ones(len(w))
    
    plt.loglog(w, mag_g)
    plt.loglog(w,  mag_rr, ls = '--')
    plt.loglog(wr * np.ones(2), [np.max(mag_g), np.min(mag_g)], ls=':')
    plt.legend(['|G|', 'R - 1', '$w_r$'],
               bbox_to_anchor=(0, 1.01, 1, 0), loc=3, ncol=3)
    plt.grid()
    plt.xlabel('Frequency [rad/s]')
    plt.ylabel('Magnitude')     
    plt.show()

#rule4(G, R, wr)


def rule5(G, Gm=1, message=False):
    '''
    This is rule five of chapter five
    
    Calculates constraints for time delay, wc < 1 / theta
    
    Parameters
    ----------
    G : tf
        plant model  
        
    Gm : tf
        measurement model
            
    message : boolean 
        show the rule message (optional)
    
    Returns
    -------
    valid5 : boolean 
        value if rule conditions was met
    
    wtd: real
        time delay frequency                  
    '''

    GGm = G * Gm
    TimeDelay = GGm.deadtime
    _,_,wc,_ = margins(GGm)
    
    valid5 = False
    if TimeDelay == 0:
        wtd = 0
    else:     
        wtd = 1 / TimeDelay     
        valid5 = wc < wtd 
    
    if message:
        if TimeDelay == 0:
            print 'There isn t any deadtime in the system'
        if valid5:
            print 'wc < 1 / theta :', wc , '<' , wtd
        else:
            print 'wc > 1 / theta :', wc , '>' , wtd
            
    return valid5, wtd

#G.deadtime = 0.002
#rule5(G, Gm, True)


def rule6(G, Gm, message=False):
    '''
    This is rule six of chapter five
    
    Calculates if tight control at low frequencies with ZHP-zeros is possible
    
    Parameters
    ----------
    G : tf
        plant model   
    
    Gm : tf
        measurement model
    
    message : boolean 
        show the rule message (optional)
    
    Returns
    -------
    valid6 : boolean value if rule conditions was met
    
    wc : crossover frequency where | G(jwc) | = 1
    
    wd : crossover frequency where | Gd(jwd) | = 1     
            
    '''

    GGm = G * Gm
    zeros = np.roots(GGm.numerator)

    _,_,wc,_ = margins(GGm)

    wz = 0
    if len(zeros) > 0:
        if np.imag(np.min(zeros)) == 0:
            # If the roots aren't imaginary.
            # Looking for the minimum values of the zeros = > results
            # in the tightest control.
            wz = (np.min(np.abs(zeros)))/2.000
            valid6 = wc < 0.86 * np.abs(wz)
        else:
            wz = 0.8600*np.abs(np.min(zeros))
            valid6 = wc < wz /2
    else: valid6 = False
    
    if message:
        if (wz != 0):
            print 'These are the roots of the transfer function matrix GGm' , zeros
        if valid6:    
            print 'The critical frequency of S for the system to be controllable is' , wz
        else: print 'No zeros in the system to evaluate'
    return valid6, wz
    
#rule6(G, Gm, message=True)


def rule7(G, Gm, message=False):
    '''
    This is rule one of chapter five
    
    Calculates the phase lag constraints
    
    Parameters
    ----------
    G : tf
        plant model   
    
    Gm : tf
        measurement model
    
    message : boolean 
        show the rule message (optional)
    
    Returns
    -------
    valid1 : boolean
        value if rule conditions was met
    
    wc : real
        crossover frequency where | G(jwc) | = 1
    
    wd : real
        crossover frequency where | Gd(jwd) | = 1    
            
    '''
    # Rule 7 determining the phase of GGm at -180 deg.
    # This is solved visually from a plot.
    
    GGm = G * Gm
    _,_,wc,w_180 = margins(GGm)
    
    valid7 = wc < w_180   
    
    if message: 
        if valid7:
            print 'wc < wu :' , wc , '<' , w_180
        else:
            print 'wc > wu :' , wc , '>' , w_180
    
    return valid7
    
#rule7(G, Gm, True)


def rule8(G, message=False):
    '''
    This is rule one of chapter five
    
    This function determines if the plant is open-loop stable at its poles
    
    Parameters
    ----------
    G : tf
        plant model   
    
    Gd : tf
        plant distrubance model
    
    message : boolean 
        show the rule message (optional)
    
    Returns
    -------
    valid1 : boolean 
        value if rule conditions was met
    
    wc : real
        crossover frequency where | G(jwc) | = 1            
    '''
    #Rule 8 for critical frequency min value due to poles

    poles = np.roots(G.denominator)
    _,_,wc,_ = margins(G)

    wp = 0
    if np.max(poles) < 0:
        wp = 2 * np.max(np.abs(poles))
        valid8 = wc > wp
    else: valid8 = False
        
    if message:
        if valid8:
            print 'wc > 2p :', wc , '>' , wp
        else:
            print 'wc < 2p :', wc , '<' , wp

    return valid8, wp

#rule8(G, True)
