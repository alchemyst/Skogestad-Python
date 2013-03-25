import numpy as np
import scipy.linalg as sc_lin 
import matplotlib.pyplot as plt 
import scipy.optimize as sc_opt
import scipy.signal as scs
from utils import phase

def G():
    """polynomial coefficients in the denominator and numerator"""
    Pz = [40]
    Pp = [1, 2, 1]
    return Pz, Pp

def R(): 
    #R is the inv(De)*Dr 
    #therefore r_max/e_max
    R=3.0000000
    return R

def Gm():
    """measuring elements dynamics"""
    Pz = [1]
    Pp = [1, 1]
    return Pz, Pp


def Time_Delay():
    """matrix with theta values, combined time delay of system and measuring element"""
    Delay = [-1]
    return Delay


def Gd():
    """polynomial coefficients in the denominator and numerator"""
    Pz = [8]
    Pp = [1, 1]
    return Pz, Pp

#all of the below function are from page 206-207 and the associated rules that solve the controllability of SISO problems

def Rule_1():
    #this is rule one
    #the function calculates the frequency where |Gd| crosses 1

    def Gd_mod_1(w):
        return np.abs(scs.freqs(Gd()[0], Gd()[1], w)[1])-1

    wd = sc_opt.fsolve(Gd_mod_1, 10)
    wc_min_1 = wd

    print 'wc > wd = ', wc_min_1
    print 'This should be the critical freqeuncy of S for the system to be controllable'

    return wc_min_1


#Rule_1()

def Rule_2(wr):
    #this function is for Rule 2 of the controllability analysis on pg-206
    #this function gives values for which S needs to be smaller than up to a frequency of wr
    #|S(jw)|<=1/R
    print '|S(jw)|<= ', 1/R()
    print 'This should be up to a freqeuncy w<= ', wr

    return 1/R()

#Rule_2(0.1)

def Rule_3(w_start, w_end):
    #this function is for Rule 3 of the controllability analysis on pg 207
    #checks input constriants for disturbance rejection

    def G_Gd_1(w):
        f = scs.freqs(G()[0], G()[1], w)[1]
        g = scs.freqs(Gd()[0], Gd()[1], w)[1]
        return np.abs(f)-np.abs(g)

    w_G_Gd = sc_opt.fsolve(G_Gd_1, 0.001)

    plt.figure(1)
    if np.abs(scs.freqs(G()[0], G()[1], [w_G_Gd+0.0001])[1])>np.abs(scs.freqs(Gd()[0], Gd()[1], [w_G_Gd+0.0001])[1]):
        print "Acceptable control"
        print "control only at high frequencies", w_G_Gd, "< w < inf"

        w = np.logspace(w_start, np.log10(w_G_Gd), 100)
        plt.loglog(w, np.abs(scs.freqs(G()[0], G()[1], w)[1]), 'r')
        plt.loglog(w, np.abs(scs.freqs(Gd()[0], Gd()[1], w)[1]), 'r.')

        max_p = np.max([np.abs(scs.freqs(G()[0], G()[1], w)[1]), np.abs(scs.freqs(Gd()[0], Gd()[1], w)[1])])

        w = np.logspace(np.log10(w_G_Gd), w_end, 100)
        plt.loglog(w, np.abs(scs.freqs(G()[0], G()[1], w)[1]), 'b')
        plt.loglog(w, np.abs(scs.freqs(Gd()[0], Gd()[1], w)[1]), 'b.')

        min_p = np.min([np.abs(scs.freqs(G()[0], G()[1], w)[1]), np.abs(scs.freqs(Gd()[0], Gd()[1], w)[1])])

    if np.abs(scs.freqs(G()[0], G()[1], [w_G_Gd-0.0001])[1])>= np.abs(scs.freqs(Gd()[0], Gd()[1], [w_G_Gd-0.0001])[1]):
        print "Acceptable control"
        print "control up to frequency 0 < w < ", w_G_Gd

        w = np.logspace(w_start, np.log10(w_G_Gd), 100)
        plt.loglog(w, np.abs(scs.freqs(G()[0], G()[1], w)[1]), 'b')
        plt.loglog(w, np.abs(scs.freqs(Gd()[0], Gd()[1], w)[1]), 'b.')

        max_p = np.max([np.abs(scs.freqs(G()[0], G()[1], w)[1]), np.abs(scs.freqs(Gd()[0], Gd()[1], w)[1])])

        w = np.logspace(np.log10(w_G_Gd), w_end, 100)
        plt.loglog(w, np.abs(scs.freqs(G()[0], G()[1], w)[1]), 'r')
        plt.loglog(w, np.abs(scs.freqs(Gd()[0], Gd()[1], w)[1]), 'r.')


        min_p = np.min([np.abs(scs.freqs(G()[0], G()[1], w)[1]), np.abs(scs.freqs(Gd()[0], Gd()[1], w)[1])])

    print 'The dotted line is for Gd and the straight line for G'
    plt.loglog(w_G_Gd*np.ones(2), [max_p, min_p], 'g')
    plt.show()


#Rule_3(-4, 6)

def Rule_4(w_star, w_end, wr):
    #this is rule 4 on pg-207 for input constraint check towards reference changes
    wr = 10**wr
    w=np.logspace(w_star, w_end, 1000)

    [w, h]=scs.freqs(G()[0], G()[1], w)

    plt.loglog(w, abs(h), 'b')
    plt.loglog([w[0], w[-1]], [(R()-1), (R()-1)], 'r')
    plt.loglog([wr, wr], [0.8*np.min([(R()-1), np.min(abs(h))]), 1.2*np.max([(R()-1), np.max(abs(h))])], 'g')

    print 'The blue line needs to be larger than the red line up to the freqeuncy where control is needed'
    print 'This plot is done up to the wr'
    print 'The green vertical line is the wr'

    plt.show()

#Rule_4(-4, 5, 0.0001)

def Rule_5():
    #this is rule 5 on pg-207 for determining the wc for S 

    print 'The is calculated with respect to the amount of deadtime in the system'

    if Time_Delay()==0:
        print 'there isn t any deadtime in the system'
    else:
        print 'wc < ', 1/Time_Delay()

    return 1/Time_Delay()

def Rule_6():
    #Rule 6 on page-270
    #peak value of wc for S in the case of RHP-Zeros

    Pz_G_Gm = np.polymul(G()[0], Gm()[0])

    print 'These are the roots of the transfer function matrix GGm '

    Pz_roots = np.roots(Pz_G_Gm)
    print Pz_roots
    print ''

    if np.real(np.max(Pz_roots))>0:

        if np.imag(np.min(Pz_roots)) == 0:
            # it the roots aren't imagenary
            # looking for the minimum values of the zeros = > results in the tightest control
            wc_6 = (np.min(np.abs(Pz_roots)))/2.000

        else:
            wc_6 = 0.8600*np.abs(np.min(Pz_roots))
    return wc_6

def Rule_7(w_start, w_end):
    #Rule 7 determining the phase of GGm at -180deg
    #this is solved visually from a plot

    w = np.logspace(w_start, w_end, 1000)

    Pz = np.polymul(G()[0], Gm()[0])
    Pp = np.polymul(G()[1], Gm()[1])
    [w, h] = scs.freqs(Pz, Pp, w)


    plt.semilogx(w, (180/np.pi)*(phase(h)+w*Time_Delay()))
    plt.show()

#Rule_7(-4, -2)

def Rule_8():
    #Rule 8 on pg-207 for critical frequency min value due to poles

    Poles = np.roots(G()[1])

    if np.max(Poles)<0:
        wc=2*np.max(np.abs(Poles))
        print 'The minimum critical frequency wc> ', wc
    else:
        print 'Check Equation 5-86 for eniquality check for unstable plants'

    return wc

Rule_8()
