import numpy as np 
import scipy as sc
import scipy.signal as scs 
import matplotlib.pyplot as plt 


def G(w):
    """for any other system this is where all the system transfer functions are stored"""
    """in the Laplace domain"""
    s=w*1j
    G=40/((s+1)*(10*s+1))
    
    return G

def Gd(w):
    s=w*1j
    Gd=8/(s+1)
    return Gd


def SISO_RULES(w_start,w_end):
    """spesifie transfer functions in laplace domian  (function G)"""
    """this function would calculate the different constraints in terms of Skogestad's 8 rules on pg 206"""
    
    w=np.logspace(w_start,w_end,100)   

    
    """rule 1 checks and limitations"""
    def Gd_w(w_cal_1):
        return np.abs(Gd(w_cal_1))-1
    
    """frequency where |Gd|=1"""
    wd  =sc.optimize.fsolve(Gd_w,0.01)
    print wd
    print """wc >= """ , wd 
        
    #plt.figure(1) 
    #plt.title('Show where |S|<=|1/Gd|')
    #plt.loglog(w,1/np.abs(Gd(w)))
       
        
    """rule 2 checks and limitations"""
    
    
    
    
    
    
    
    
    
    """rule 3 checks and limitations"""
    def G_Gd(w_cal):
        """|G|=|Gd|-1"""
        return np.abs(G(w_cal))-np.abs(Gd(w_cal))+1

    w_G_Gd=sc.optimize.fsolve(G_Gd,0.01)
    
    print w_G_Gd
    
    if np.abs(G(w_G_Gd+1))>np.abs(Gd(w_G_Gd+1)):
        print """control only at high frequencies""",w_G_Gd,"""< w < inf"""
        
    elif np.abs(G(w_G_Gd-1))>np.abs(Gd(w_G_Gd-1)):
        print """control up to frequency 0 < w < """,w_G_Gd
        
    """plot to check """
    """color coded plots"""
    
    plt_good_G  =[]
    plt_good_Gd =[]
    w_good      =[]
    
    plt_bad_G   =[]
    plt_bad_Gd  =[]
    w_bad       =[]
    
    for w_iter in w:
        
        if np.abs(G(w_iter))>np.abs(Gd(w_iter)):
            plt_good_G.append(np.abs(G(w_iter)))
            plt_good_Gd.append(np.abs(Gd(w_iter)))
            w_good.append(w_iter)
            
        elif np.abs(G(w_iter))<=np.abs(Gd(w_iter)):
            plt_bad_G.append(np.abs(G(w_iter)))
            plt_bad_Gd.append(np.abs(Gd(w_iter)))
            w_bad.append(w_iter)
        
    plt.loglog(w_good,plt_good_G,'b.')
    plt.loglog(w_good,plt_good_Gd,'y.')
    plt.loglog(w_bad,plt_bad_G,'r.')

    plt.loglog(w_bad,plt_bad_Gd,'g.')
    
    print w_G_Gd*np.ones(2)
    print np.min(G(w))
    print np.abs(G(w_G_Gd))
    

    plt.loglog(w_G_Gd*np.ones(2),[np.min(np.abs(G(w))),np.abs(G(w_G_Gd))])
    
    
    """rule 4 checks and limitations"""
    
    
    """rule 5 checks and limitations"""
    
    
    """rule 6 checks and limitations"""
    
    
    """rule 7 checks and limitations"""
    
    
    """rule 8 checks and limitations"""
    
    
    plt.show()


   
SISO_RULES(-3,3)