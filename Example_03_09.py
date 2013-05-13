import numpy as np
from utils import RGA

# Blending process: mix sugar (u1) and water (u2) to control amount
# (y1 = F) and sugar fraction(y2 = x) in a soft drink

# y = G*u
# After linearization the equations become:
# y1 = u1+u2
# y2 = (1-x*)/(F*)u1 - (x*/F*)u2
# given that at steady-state
# x* = 0.2
# F* = 2 kg/s
# therefore;
# G(s) = np.matrix([[1, 1], [(1-x*)/F* -x*/F*]])
# after substitution
G = np.matrix([[1, 1], [0.4, -0.1]])
print RGA(G)

# pairing rule 1: prefer pairing on RGA elements close to 1
# RGA = [[0.2, 0.8], [0.8, 0.2]]
# RGA11 = 0.2; effect of u1 on y1
# RGA12 = 0.8 close to 1; effect of u2 (water) on the amount y1
# RGA21 = 0.8 close to 1; effect of u1(sugar) on the sugar fraction y2
# RGA22 = 0.2; effect of u2 (water) on the sugar fraction y2
# It is reasonable to use input u2 (water) to control amount y1 hence RGA = 0.8
# and use input u1 (sugar) to control the sugar fraction y2
