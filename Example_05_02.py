from utils import tf, RHPonly
import numpy as np

s = tf([1,0],[1])
G = 10*(s - 2)/(s**2 - 2*s + 5)

RHPpoles = RHPonly(G.poles())
RHPzeros = RHPonly(G.zeros())

print("The RHP poles of the system are:",RHPpoles)
print("The RHP zeros of the system are:",RHPzeros)

Ms = 1
for p in RHPpoles:
    Ms *= np.abs(RHPzeros + p)/np.abs(RHPzeros - p)
    
print("The tight lower bound on ||S|| and ||T|| is:", Ms)