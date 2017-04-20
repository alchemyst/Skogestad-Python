"""

Test function checking dimensions of tf2ss

"""

import utils

s = utils.tf([1, 0], 1)

G = utils.mimotf([
    [0.66/(6.7*s + 1), -0.61/(8.64*s + 1), -0.0049/(9.06*s + 1)],
    [1.11/(3.25*s + 1), -2.36/(5.0*s + 1), -0.012/(7.09*s + 1)],
    [-34.68/(8.15*s + 1), 46.2/(10.9*s + 1), 0.87*(11.61*s + 1)/((3.89*s + 1)*(18.8*s + 1))]
    ])

A, B, C, D = utils.tf2ss(G)

Nstates = A.shape[0]
Ninputs = B.shape[1]
Noutputs = C.shape[0]

print(A.shape, B.shape)
print(C.shape, D.shape)

assert A.shape[0] == A.shape[1], "A must be square"
assert B.shape[0] == Nstates, "B must have the same number of rows as A"
assert C.shape[1] == Nstates, "C must have the same number or columns as A"
assert D.shape[0] == Ninputs, "D must have the same number of rows as C"
