from __future__ import print_function
import numpy as np
import numpy.linalg as nplinalg
from utils import tf, BoundST, RHPonly
from scipy.linalg import sqrtm
from math import sqrt

s = tf([1,0],[1])

G = ((s - 1) * (s - 3)) / ((s - 2) * (s + 1)**2)

poles = G.poles()
zeros = G.zeros()

RHPzeros = RHPonly(zeros)
RHPpoles  = RHPonly(poles)

#directions = one because SISO system
pole_dir = np.ones(len(RHPpoles)) 
zero_dir = np.ones(len(RHPzeros))

def Q(x, x_dir, y=0, y_dir=0):
    if y==0 and y_dir == 0:
        row = np.shape(x)[0]
        col = row
        Q_return = np.zeros((row, col))
        for i in range(row):
            for j in range(col):
                Q_return[i,j] = (x_dir[i] * x_dir[j]) / (x[i] + x[j])
    else:
        row = np.shape(x)[0]
        col = np.shape(y)[0]
        Q_return = np.zeros((row,col))
        for i in range(row):
            for j in range(col):
                Q_return[i,j] = (x_dir[i] * y_dir[j]) / (x[i] - y[j])
    return Q_return

Qz  = Q(RHPzeros, zero_dir)
Qp  = Q(RHPpoles, pole_dir)
Qzp = Q(RHPzeros, zero_dir, RHPpoles, pole_dir) 

A = sqrtm(nplinalg.inv(Qz)).dot(Qzp).dot(sqrtm(nplinalg.inv(Qp)))
_,sv,_ = nplinalg.svd(A)

M_Smin = sqrt(1+max(np.abs(sv))**2)
print("M_Smin using eq 6.8 = ",np.round(M_Smin, 2))

# alternative because system has only one pole:
M_Smin = 1
for j in range(len(RHPzeros)):
    M_Smin *= np.abs(RHPzeros[j] + RHPpoles[0]) / np.abs(RHPzeros[j] - RHPpoles[0])

print('MSmin from alternative calculation = ', M_Smin)
