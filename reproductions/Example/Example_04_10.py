from __future__ import print_function
import sympy
import robustcontrol

s = sympy.Symbol('s')
M = sympy.Matrix([[(s - 1)*(s + 2), 0, (s - 1)**2],
                  [-(s + 1)*(s + 2), (s - 1)*(s + 1),  (s - 1)*(s + 1)]])
G = (1/((s + 1)*(s + 2)*(s - 1)))*M

poles = robustcontrol.poles(G)

print("The poles of the system are = ", poles)

s = robustcontrol.tf([1, 0], 1)
M = robustcontrol.mimotf([[(s - 1) * (s + 2), 0, (s - 1) ** 2],
                  [-(s + 1) * (s + 2), (s - 1) * (s + 1), (s - 1) * (s + 1)]])
G = 1/((s + 1)*(s + 2)*(s - 1)) * M
print("The poles of the system are = ", G.poles())
