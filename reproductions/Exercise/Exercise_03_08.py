import numpy as np

from robustcontrol import RGA

G = np.matrix([[16.8, 30.5, 4.30],
               [-16.7, 31.0, -1.41],
               [1.27, 54.1, 5.4]])

# Define 6 alternate pairings
I1 = np.asmatrix(np.eye(3))
I2 = np.matrix([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
I3 = np.matrix([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
I4 = np.matrix([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
I5 = np.matrix([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
I6 = np.matrix([[0, 1, 0], [1, 0, 0], [0, 0, 1]])

for I in I1, I2, I3, I4, I5, I6:
    print('Pairing', '\n', I, 'RGA Number =', np.sum(np.abs(RGA(G) - I)))
# Pairing of diagonal matrix I1 provides the smallest RGA number and is
# is therefore preferred
