import numpy as np

from robustcontrol import RGA

G = np.matrix([[16.8, 30.5, 4.30],
               [-16.7, 31.0, -1.41],
               [1.27, 54.1, 5.4]])

# Iterative evaluation of the RGA
k = 1
while (k <= 5):
    G = RGA(G)
    print('Iteration**', k, '\n', G.round(3))
    k = k + 1
# result confirms diagonal dominance pairing since G converges
# to identity matrix
