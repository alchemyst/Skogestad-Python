import numpy as np 
import scipy.linalg as sc_lin 


def generating(mat, position_row, position_column): 
    
    
    return matrix 

def construct(mat,position_row,position_column):
    import numpy as np
    if (np.shape(mat)[0]==len(position_row)) and (len(position_row)==len(position_column)):
        for i in range(len(position_row)): 
            for j in range(len(position_row)): 
                if position_row[i]==position_row[j]: 
                    if np.shape(mat[i])[0]==np.shape(mat[j])[0]: 
                        continue 
                    else: 
                        print 'matrix sizes not compatible'
        for i in range(len(position_column)):
            for j in range(len(position_column)): 
                if position_column[i]==position_column[j]: 
                    if np.shape(mat(i))[1]==np.shape(mat[j])[1]:
                        continue 
                    else: 
                        print 'matrix sizes not compatible'
                         
    if (np.shape(mat)[0]!=len(position_row)) and (len(position_row)!=len(position_column)):
       print 'not correct amount of information'
       
                         