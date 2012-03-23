import numpy as np

"""Define functions"""
G_3_29 = np.matrix([[1, 2], [-3, 4]])
G_3_30 = np.matrix([[0, 100], [0, 0]])

"""Function to calculate norms"""
def norms(G):
    spectrad = np.matrix.max(np.matrix(np.linalg.eigvals(G)))
    Frob_norm = np.linalg.norm(G, 'fro')
    B = np.cumsum(abs(G))
    Sum_norm = B[0, G.size-1]
    Columnsum = np.matrix.max(np.matrix(sum(abs(G))))
    Rowsum = np.matrix.max(np.matrix(sum(np.matrix.transpose(abs(G)))))
    [U, S, V] = np.linalg.svd(G)
    Max_singularval = np.matrix.max(np.matrix(S))

    print 'Spectral radius = ' + str(spectrad)
    print 'Frobenius norm = ' + str(Frob_norm)
    print 'Sum norm = ' + str(Sum_norm)
    print 'Maximum column sum = ' + str(Columnsum)
    print 'Maximum row sum = ' + str(Rowsum)
    print 'Maximum singular value = ' + str(Max_singularval)

norms(G_3_29)
