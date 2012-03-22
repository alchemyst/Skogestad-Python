import numpy as np 
import scipy as sc
import scipy.linalg as sc_lin
import scipy.optimize as sc_opt
import scipy.signal as scs
import matplotlib.pyplot as plt 


def G(s):
    """ give the transfer matrix of the system"""
    G       = np.matrix([[ 1/s+1 , 1 ],[1/(s+2)**2 , (s+1)/(s-2)]])
    return G 

def Gd(s):
    Gd      = np.matrix([[1/(s+1)],[1/(s+20)]])
    return Gd

def Gms(s):
    """ stable, minimum phase system of G and Gd"""
    G_ms    = [[]]
    Gd_ms   = [[]]
    
    G_s     = [[]]
    Gd_s    = [[]]
    return G_ms, Gd_ms ,G_s , Gd_s

def Zeros_Poles_RHP():
    """ Give a vector with all the RHP zeros and poles
    RHP zeros and poles are calulated from sage program"""
    
    Zeros_G     =[1]
    Poles_G     =[-2]
    Zeros_Gd    =[]
    Poles_Gd    =[]
    return Zeros_G , Poles_G , Zeros_Gd , Poles_Gd 



def deadtime(s):
    """ vector of the deadtime of the system""" 
    #individual time delays 
    dead_G      = []
    dead_Gd     = []

    return dead_G, dead_Gd 




def PEAK_MIMO(w_start,w_end,error_poles_direction,R,wr):
    """ this function is for multivariable system analyses of controlability
    gives:
    minimum peak values on S and T with or without deadtime
    R is the expected worst case reference change
    wr is the freqeuncy up to where reference tracking is required""" 
    

    
    #importing most of the zeros and poles data 
    
    [Zeros_G,Poles_G,Zeros_Gd,Poles_Gd ]=Zeros_Poles_RHP()
    
    #just to save uneccasary calculations that is not needed
    if np.sum(Zeros_G)!=0:
        if np.sum(Poles_G)!=0:   
            #sensitivty peak of closed loop. eq 6-8 pg 224 skogestad 
            
            #two matrices to save all the RHP zeros and poles directions
            yz_direction = np.matrix(np.zeros([G(0.001).shape[0],len(Zeros_G)]))
            yp_direction = np.matrix(np.zeros([G(0.001).shape[0],len(Poles_G)]))
           
            
            for i in range(len(Zeros_G)):
               
                [U,S,V]              = np.linalg.svd(G(Zeros_G[i]+error_poles_direction))
                yz_direction[:,i]    = U[:,-1]

            
            for i in range(len(Poles_G)):
                #error_poles_direction is to to prevent the numerical method from breaking
                [U,S,V]              = np.linalg.svd(G(Poles_G[i]+error_poles_direction))
                yp_direction[:,i]    = U[:,0]
            
            
            yz_mat1     = np.matrix(np.diag(Zeros_G))*np.matrix(np.ones([len(Zeros_G),len(Zeros_G)]))
            yz_mat2     = np.transpose(yz_mat1)
            
            Qz          = (np.transpose(np.conjugate(yz_direction))*yz_direction)/(yz_mat1+yz_mat2)
            
            yp_mat1     = np.matrix(np.diag(Poles_G))*np.matrix(np.ones([len(Poles_G),len(Poles_G)]))
            yp_mat2     = np.transpose(yp_mat1)
        
            Qp          = (np.transpose(np.conjugate(yp_direction))*yp_direction)/(yp_mat1+yp_mat2)
            
            yzp_mat1    = np.matrix(np.diag(Zeros_G))*np.matrix(np.ones([len(Zeros_G),len(Poles_G)]))
            yzp_mat2    = np.matrix(np.ones([len(Zeros_G),len(Poles_G)]))*np.matrix(np.diag(Poles_G))
            
            Qzp         = np.transpose(np.conjugate(yz_direction))*yp_direction/(yzp_mat1-yzp_mat2)
            
            
            #this matrix is the matrix from which the SVD is going to be done to determine the final minimum peak
            pre_mat     = (np.sqrt((complex(np.linalg.inv(Qz))))*Qzp*(np.sqrt(complex(np.linalg.inv(Qp)))))
            
            #final calculation for the peak value
            Ms_min      = np.sqrt(1+(np.max(np.linalg.svd(pre_mat)[1]))**2)
            print ''
            print 'Minimum peak values on T and S'
            print 'Ms_min = Mt_min =', Ms_min
            
        else:
            print '' 
            print 'Minimum peak values on T and S' 
            print 'No limits on minimum peak values'
            
    #check for dead time 
    #dead_G=deadtime[0]
    #dead_gd=deadtime[1]
    
    #if np.sum(dead_G)!=0:
        #therefore deadtime is present in the system therefore exstra precuations need to be taken 
        #manually set up the dead time matrix
        
    #    dead_m=np.zeros([len(Poles_G),len(Poles_G)])
        
        
    #    for i in range(len(Poles_G)):
    #        for j in range(len(Poles_G))
    #            dead_m
      
    #plant with RHP zeros from 6-48
    #checking that the plant and controlled variables have the ability to reject load disturbances
    
    
     
      
    #eq 6-50 pg 240 from skogestad          
    #Checking input saturation for perfect control for disturbance rejection
    #checking for maximum disturbance just at steady state 
    
    [U_gd,S_gd,V_gd] = np.linalg.svd(Gd(0.000001))
    y_gd_max         = U_gd[:,0]
    mod_G_gd_ss      = np.max(np.linalg.inv(G(0.000001))*y_gd_max)
    
    print ''
    print 'Perfect control input saturation from disturbances'
    print 'Max Norm method'
    print 'Checking input saturation at steady state'
    print 'This is done by the worse output direction of Gd'
    print mod_G_gd_ss
    
    w    = np.logspace(w_start,w_end,1000)
    
    mod_G_gd    = np.zeros(len(w))
    
    for i in range(len(w)):
        [U_gd,S_gd,V_gd]    = np.linalg.svd(Gd(1j*w[i]))
        gd_m                = U_gd[:,0]
        mod_G_gd[i]         = np.max(np.linalg.inv(G(1j*w[i]))*gd_m)
    
    def G_gd(w):
        [U_gd,S_gd,V_gd]    = np.linalg.svd(Gd(1j*w))
        gd_m                = U_gd[:,0]
        mod_G_gd[i]         = np.max(np.linalg.inv(G(1j*w))*gd_m)-1
        return mod_G_gd
    
    #w_mod_G_gd_1    =sc_opt.fsolve(G_gd,0.001)
    
    
    #print 'Freqeuncy till which input saturation would not acure'
    #print w_mod_G_gd_1
    print 'Figure 1 is the plot of G**1 gd'
<<<<<<< HEAD
    print '' 
=======
>>>>>>> 1e81e23e5f7a94b69a049c1041958624b91e9c0a
    plt.figure(1)
    plt.xlabel('w')
    plt.ylabel('|inv(G)* gd|')
    plt.semilogx(w,mod_G_gd) 
    
    
    #checking input saturation for acceptable control  disturbance rejection 
    #equation 6-55 pg 241 in skogestad 
<<<<<<< HEAD
    #checking each singular values and the associated input vector with output direction vector of Gd
    
    store_value_input_sat=np.zeros([])
    
    for i in range(len(w)):
        
=======
    
>>>>>>> 1e81e23e5f7a94b69a049c1041958624b91e9c0a
    
    
    
    
    #checking input saturation for perfect control with reference change 
    #eq 6-53 pg 241 
 
    singular_min_G_ref_track=[np.min(np.linalg.svd(G(1j*w_i))[1]) for w_i in w]
<<<<<<< HEAD
    plt.figure(2)
    plt.loglog(w,singular_min_G_ref_track)
    plt.loglog([w[0],w[-1]],[1,1])
    plt.loglog(w[0],1.2)
    plt.loglog(w[0],0.8)
    plt.loglog([wr,wr],[np.min([0.8,np.min(singular_min_G_ref_track)]),np.max([1.2,np.max(singular_min_G_ref_track)])])
    
    
    
    print 'Figure 2 is to check input saturation for reference changes'
    print 'Shows a line of 1, where the singular value needs to be above 1'
    print 'Shows the wr up to where control is needed'
    print '' 
    
    
    #checking input saturation for accepatable control with reference change 
    
    
    #added check for controllability is the minimum and maximum singular values of system transfer function matrix 
    # as a function of frequency
    singular_min_G=[np.min(np.linalg.svd(G(1j*w_i))[1]) for w_i in w]
    singular_max_G=[np.max(np.linalg.svd(G(1j*w_i))[1]) for w_i in w]
    
    print 'Figure 3 is the maximum and minimum sigular values of G over a freqeuncy range'    
    print ''
    plt.figure(3)
=======
    

    
    
    #checking input saturation for accepatable control with reference change 
    
    
    #added check for controllability is the minimum and maximum singular values of system transfer function matrix 
    # as a function of frequency
    
    singular_min_G=np.zeros(len(w))
    singular_max_G=np.zeros(len(w))
    
    
    for i in range(len(w)):
        singular_min_G[i]= np.min(np.linalg.svd(G(1j*w[i]))[1])
        singular_max_G[i]= np.max(np.linalg.svd(G(1j*w[i]))[1])
    
    
    print 'Figure 2 is the maximum and minimum sigular values of G over a freqeuncy range'    
    plt.figure(2)
>>>>>>> 1e81e23e5f7a94b69a049c1041958624b91e9c0a
    plt.loglog(w,singular_min_G)   
    plt.loglog(w,singular_max_G)
    
        
    plt.show()
    
    return Ms_min

R=np.matrix([[1],[1]])
R=R/np.abs(R)

#just a check to make shure the R and G matrix is the correct shapes
G(1)*R

PEAK_MIMO(-4,5,0.00001,R,0.1 )
        