import numpy as np 
import matplotlib.pyplot as plt 


def Stability(Poles_poly):
    """determine if the characteristic equation of a transfer function is stable
    Poles_poly is a polynomial expansion of the characteristic equation """
    
    return any(np.real(r) > 0 for r in np.roots(Poles_poly))
    vec=np.zeros(len(roots))
    
    # creating a vector with ones and zeros representing a RHP pole or LHP pole
    for i in range(len(roots)):
        if roots[i]<0:
            vec[i]=1
        elif roots[i]>=0:
            vec[i]=0
    
    # if the sum of all the elements are equal to the length => its stable (all the poles are in the LHP)
    if np.sum(vec)==len(vec):
        return 1 
    else:
        return 0
    

def System(KC):
    """giving the characteristic equation in terms of the Kc value"""
    
    Poles_poly=[50,15-6*KC,1+3*KC]
    return Poles_poly 

vec=[]
Kc=np.linspace(-2,3,200)


# creating a plot to indicate over what range of Kc the system would be stable
for i in range(len(Kc)):
    vec.append(Stability(System(Kc[i])))
    

plt.plot(Kc,vec,'rD')
plt.show()
