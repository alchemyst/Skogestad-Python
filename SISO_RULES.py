import numpy as np 
import scipy as sc 
import scipy.signal as scs
import matplotlib.pyplot as plt 


"""this function is different that the previous one in its inputs it takes"""
"""this is easier than geving the actaul transfer function"""
"""due to the ease of which the roots of the transfer function could be determined from the numerator and denominators' polynomial 
expansion""" 

def G():
    """polynomial coefficients in the denominator and numerator"""
    Pz=[40]
    Pp=[10,11,1]
    return Pz, Pp

def Gm ():
    """measuring elements dyanmics"""
    Pz=[-1,1,2]
    Pp=[1]
    return Pz,Pp

def Time_Delay():
    """matrix with theta values, combined time delay of system and measuring element"""
    Delay=[1]
    return Delay 

def Gd():
    """polynomial coefficients in the denominator and numerator"""
    Pz=[8]
    Pp=[1,1]
    return Pz, Pp






def RULES():
    
    
    """rule 1 wc>wd"""
    
    def Gd_mod_1(w):
        return np.abs(scs.freqs(Gd()[0],Gd()[1],w)[1])-1
    
    wd=sc.optimize.fsolve(Gd_mod_1,10)
    
    wc_min=wd
    
    print """wc > """,wd 
    
    
    """rule 2"""
    
    """rule 3"""
    
    
    
    """for perfect control"""
    
    def G_Gd_1(w):
        f=scs.freqs(G()[0],G()[1],w)[1]
        g=scs.freqs(Gd()[0],Gd()[1],w)[1]
        return np.abs(f)-np.abs(g)
    
    w_G_Gd=sc.optimize.fsolve(G_Gd_1,0.001)
        
    
    if np.abs(scs.freqs(G()[0],G()[1],[w_G_Gd+0.0001])[1])>np.abs(scs.freqs(Gd()[0],Gd()[1],[w_G_Gd+0.0001])[1]):
        print """Acceptable control"""
        print """control only at high frequencies""",w_G_Gd,"""< w < inf"""
        
        w=np.logspace(-3,np.log10(w_G_Gd),100)
        plt.loglog(w,np.abs(scs.freqs(G()[0],G()[1],w)[1]),'r')
        plt.loglog(w,np.abs(scs.freqs(Gd()[0],Gd()[1],w)[1]),'r.')
        
        max_p=np.max([np.abs(scs.freqs(G()[0],G()[1],w)[1]),np.abs(scs.freqs(Gd()[0],Gd()[1],w)[1])])
        
        w=np.logspace(np.log10(w_G_Gd),5,100)
        plt.loglog(w,np.abs(scs.freqs(G()[0],G()[1],w)[1]),'b')
        plt.loglog(w,np.abs(scs.freqs(Gd()[0],Gd()[1],w)[1]),'b.')
        
        min_p=np.min([np.abs(scs.freqs(G()[0],G()[1],w)[1]),np.abs(scs.freqs(Gd()[0],Gd()[1],w)[1])])
        
    if np.abs(scs.freqs(G()[0],G()[1],[w_G_Gd-0.0001])[1])>=np.abs(scs.freqs(Gd()[0],Gd()[1],[w_G_Gd-0.0001])[1]):
        print """Acceptable control"""
        print """control up to frequency 0 < w < """,w_G_Gd
        
        w=np.logspace(-3,np.log10(w_G_Gd),100)
        plt.loglog(w,np.abs(scs.freqs(G()[0],G()[1],w)[1]),'b')
        plt.loglog(w,np.abs(scs.freqs(Gd()[0],Gd()[1],w)[1]),'b.')
       
        max_p=np.max([np.abs(scs.freqs(G()[0],G()[1],w)[1]),np.abs(scs.freqs(Gd()[0],Gd()[1],w)[1])]) 
        
        w=np.logspace(np.log10(w_G_Gd),5,100)
        plt.loglog(w,np.abs(scs.freqs(G()[0],G()[1],w)[1]),'r')
        plt.loglog(w,np.abs(scs.freqs(Gd()[0],Gd()[1],w)[1]),'r.')
        

        min_p=np.min([np.abs(scs.freqs(G()[0],G()[1],w)[1]),np.abs(scs.freqs(Gd()[0],Gd()[1],w)[1])])
        
        
    plt.loglog(w_G_Gd*np.ones(2),[max_p,min_p],'g')
    
    
    """rule 4 """
    
    
    
    """rule 5 """
    """critical freqeuncy of controller needs to smaller than"""
    wc_max=Time_Delay()[0]/2.0000
    
    
    """rule 6"""
    """control over RHP zeros"""
  
    Pz_G_Gm=np.polymul(G()[0],Gm()[0])
    
    if len(Pz_G_Gm)==1:
        wc_max=wc_max
    else:
        Pz_roots=np.roots(Pz_G_Gm)

        if np.real(np.max(Pz_roots))>0:
      
            if np.imag(np.min(Pz_roots))==0:
                """it the roots aren't imagenary"""
                """looking for the minimum values of the zeros => results in the tightest control"""
                wc_max=np.min([wc_max,(np.min(Pz_roots))/2.000])
       
            else:
                wc_max=np.min([wc_max,0.8600*np.abs(np.min(Pz_roots))])
             
        else:
            wc_max=wc_max
    
    print wc_max
    """rule 7"""
    def G_GM(w):
        G_w=scs.freqs(np.polymul(G()[0],Gm()[0]),np.polymul(G()[1],Gm()[1]),w)[1]
        return np.arctan2(np.imag(G_w),np.real(G_w))-np.pi
    
    wu=sc.optimize.fsolve(G_GM,0.0010)
    
    print wu
    
    wc_max=np.min(wu,wc_max)
    print wc_max
    
    
    """rule 8"""
        
    plt.show()
    
RULES()
    