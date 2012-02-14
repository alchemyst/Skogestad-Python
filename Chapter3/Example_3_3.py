import numpy as np 
import scipy as sc
import scipy.signal as scs 
import matplotlib.pyplot as plt 

G=np.matrix([[5,4],[3,2]])

"""SVD decomposition"""

[U,S,T]=np.linalg.svd(G)

"""first function is to create the unit circle"""

def Unit_circle():
    x=np.linspace(-0.99,0.99,100)
    y1=np.sqrt(1-x**2)
    y2=-1*np.sqrt(1-x**2)

    
    x_vec,y_vec=np.zeros(2*len(x)),np.zeros(2*len(x))
    x_vec[0:len(x)]=x
    y_vec[0:len(x)]=y1
    x_vec[len(x):]=x
    y_vec[len(x):]=y2
    
    return x_vec,y_vec

[d1,d2]=Unit_circle()




"""generating the plot in figure 3.5"""


for i in range(len(d1)):
    d =np.matrix([[d1[i]],[d2[i]]])
    y_out=G*d
    y_axis=np.sqrt(y_out[0]**2+y_out[1]**2)/np.sqrt(d1[i]**2+d2[i]**2)
    x_axis=d1[i]/d2[i]
    plt.plot(x_axis,y_axis,'b.')
    
plt.show()

