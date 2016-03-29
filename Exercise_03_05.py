import numpy as np

# Define functions
G_3_29 = np.matrix([[5, 4], [3, 2]])
G_3_30 = np.matrix([[0, 100], [0, 0]])

def norms(G, Matrix_input):
    """Function to calculate norms"""
    spectrad = max(np.linalg.eigvals(G))
    Frob_norm = np.linalg.norm(G, 'fro')
    B = np.cumsum(abs(G))
    Sum_norm = B[0, G.size-1]
    Columnsum = np.max(np.sum(abs(G),axis=0))
    Rowsum = np.max(np.sum(abs(G),axis=1))
    [U, S, V] = np.linalg.svd(G)
    Max_singularval = max(S)

    print Matrix_input
    print 'Spectral radius =', np.round(spectrad,2)
    print 'Frobenius norm =', np.round(Frob_norm,2)
    print 'Sum norm =', np.round(Sum_norm,2)
    print 'Maximum column sum =', np.round(Columnsum,2)
    print 'Maximum row sum =', np.round(Rowsum,2)
    print 'Maximum singular value = \n', np.round(Max_singularval,2)

norms(G_3_29, "Matrix 3.29")
norms(G_3_30, "Matrix 3.30")



