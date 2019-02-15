import numpy as np
from utils import tf, zeros
from scipy import signal
import sympy as sp

s = tf([1,0],[1])

G_tf = (1 - s)/(1 + s)

G_ss = signal.tf2ss(G_tf.numerator,G_tf.denominator)
A,B,C,D = G_ss

def checkdimensions(x):
    if x.ndim == 1:
        x = np.array([x])
    return x

A = checkdimensions(A)
B = checkdimensions(B)
C = checkdimensions(C)
D = checkdimensions(D)
    
AB = np.concatenate((A,B),axis=1)
CD = np.concatenate((C,D), axis=1)
M  = np.concatenate((AB,CD),axis=0)

rowA, colA = np.shape(A)
rowB, colB = np.shape(B)
rowC, colC = np.shape(C)
rowD, colD = np.shape(D)

I = np.eye(rowA,colA)
top = np.concatenate((I,np.zeros((rowB,colB))),axis=1)
bottom = np.concatenate((np.zeros((rowC,colC)),np.zeros((rowD,colD))),axis=1)
Ig = sp.Matrix(np.concatenate((top,bottom),axis=0))

z = sp.Symbol('z')
zIg = z * Ig
f = zIg - M
zf = f.det()
zero = sp.solve(zf, z)
print("Zero calculated by manual method = ",zero)

# or using utils function which I only discovered after doing all this
zeros = zeros(None,A,B,C,D)
print("Zero calculated by utils = ",zeros)
        
