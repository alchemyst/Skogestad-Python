from __future__ import print_function
import sympy as sp
import itertools

s = sp.Symbol('s')
G = (1/((s+1)*(s+2)*(s-1)))*sp.Matrix([[(s-1)*(s+2), 0, (s-1)**2],[-(s+1)*(s+2), (s-1)*(s+1), (s-1)*(s+1)]])

def minors(G,order):
    retlist = []
    Nrows, Ncols = G.shape
    for rowstokeep in itertools.combinations(range(Nrows),order):
        for colstokeep in itertools.combinations(range(Ncols),order):
            retlist.append(G[rowstokeep,colstokeep].det().simplify())
    return retlist   

Nrows, Ncols = G.shape
allminors = []
lcm = 1
for i in range(1,min(Nrows,Ncols)+1,1):
    allminors = minors(G,i)
    for m in allminors:
        numer, denom = m.as_numer_denom()
        lcm = sp.lcm(lcm,denom)
print('The lowest common denominator is: ', lcm.factor())

poles = sp.solve(lcm)
print('The poles of the system are: ',poles)
