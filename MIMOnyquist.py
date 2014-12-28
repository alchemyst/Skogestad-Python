"""
Nyquist plot for a MIMO system.

Adapt the already specified transfer funtion matrix to match your system. 
"""


import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt


K = np.array([[1., 2.],
                 [3., 4.]])
t1 = np.array([[5., 5.],
                 [5., 5.]])
t2 = np.array([[5., 6.],
               [7., 8.]]) 
#Controller
Kc = np.array([[0.1, 0.], 
               [0., 0.1]])*6


def G(s):
    return(K*np.exp(-t1*s)/(t2*s + 1))


def L(s):
    return(Kc*G(s))


def MIMOnyqPlot(L):
    w = np.logspace(-3, 3, 1000)    
    Lin = np.zeros((len(w)), dtype=complex)
    x = np.zeros((len(w)))
    y = np.zeros((len(w)))
    dim = np.shape(L(0))
    for i in range(len(w)):        
        Lin[i] = la.det(np.eye(dim[0]) + L(w[i]*1j))
        x[i] = np.real(Lin[i])
        y[i] = np.imag(Lin[i])        
    plt.figure()
    plt.clf()
    plt.plot(x, y, 'k-', lw=1)
    plt.xlabel('Re G(wj)')
    plt.ylabel('Im G(wj)')
    # plotting a unit circle
    x = np.linspace(-1, 1, 200)
    y_up = np.sqrt(1-(x)**2)
    y_down = -1*np.sqrt(1-(x)**2)
    plt.plot(x, y_up, 'b:', x, y_down, 'b:', lw=2)
    plt.plot(0, 0, 'r*', ms = 10)
    plt.grid(True)
    n = 2               # Sets x-axis limits
    plt.axis('equal')   # Ensure the unit circle remains round on resizing the figure
    plt.xlim(-n, n)
    fig = plt.gcf()
    BG = fig.patch
    BG.set_facecolor('white')
    plt.show()  
    
MIMOnyqPlot(L)

print('Done!')
