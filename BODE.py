import numpy as np 
import scipy as sc 
import scipy.signal as scs
import matplotlib.pyplot as plt 



def G(w):
    """system equations in here (laplace domian)"""
    """only for SISO problems for now"""
    s   =w*1j
    
    # enter what ever system under inspections, transfer function in here
    G=8/((s+8)*(s+1))
    
    return G

def Bode():
    """give the Bode plot along with GM and PM"""
    
    def mod(x):
        """to give the function to calculate |G(jw)|=1"""
        return np.abs(G(x))-1
    
    # how to calculate the freqeuncy at which |G(jw)|=1
    wc=sc.optimize.fsolve(mod,0.1)

    def arg(w):
        """function to calculate the phase angle at -180 deg"""
        return np.arctan2(np.imag(G(w)),np.real(G(w)))+np.pi
    
    # where the freqeuncy is calculated where arg G(jw)=-180 deg
    w_180=sc.optimize.fsolve(arg,-1)
    
    
    PM=(np.arctan2(np.imag(G(wc)),np.real(G(wc)))+np.pi)*180/np.pi
    GM=1/(np.abs(G(w_180)))
    
    # plotting of Bode plot and with corresponding freqeuncies for PM and GM 
    w=np.logspace(-5,np.log(w_180),1000)
    plt.subplot(211)
    plt.loglog(w,np.abs(G(w)))
    plt.loglog(wc*np.ones(2),[np.max(np.abs(G(w))),np.min(np.abs(G(w)))])
    plt.text(w_180,np.average([np.max(np.abs(G(w))),np.min(np.abs(G(w)))]),'<G(jw)=-180 Deg')
    plt.loglog(w_180*np.ones(2),[np.max(np.abs(G(w))),np.min(np.abs(G(w)))])
    plt.loglog(w,1*np.ones(len(w)))
    
    # argument of G
    plt.subplot(212)
    plt.semilogx(w,180/np.pi*np.unwrap(np.arctan2(np.imag(G(w)),np.real(G(w)))))
    plt.semilogx(wc*np.ones(2),[np.max(180/np.pi*np.unwrap(np.arctan2(np.imag(G(w)),np.real(G(w))))),np.min(180/np.pi*np.unwrap(np.arctan2(np.imag(G(w)),np.real(G(w)))))])
    plt.semilogx(w_180*np.ones(2),[-180,0])
    plt.show()
    
    return GM,PM

    
[GM,PM]=Bode()

print GM,PM
    
