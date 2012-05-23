import matplotlib.pyplot as plt 
import scipy.linalg as sc_lin 
import numpy as np 

#this program contains all the important equations of chapter 6 of Skogestad pre-programmed
#the first set of function is of the specific system

def G(s):
    """ give the transfer matrix of the system"""
    G = np.matrix([[1/s+1, 1], [1/(s+2)**2, (s-1)/(s-2)]])
    return G

def Gd(s):
    Gd = np.matrix([[1/(s+1)], [1/(s+20)]])
    return Gd

def reference_change():
    """Reference change matrix/vector for use in eq 6-52 pg 242 to check input saturation"""

    R = np.matrix([[1, 0], [0, 1]])
    R = R/np.linalg.norm(R, 2)
    return R

def G_s(s):
    """ stable, minimum phase system of G and Gd
    This could be done symbolically using Sage"""

    G_s = np.matrix([[1/s+1, 1], [1/(s+2)**2, (s-1)/(s+2)]])
    return G_s

def Zeros_Poles_RHP():
    """ Give a vector with all the RHP zeros and poles
    RHP zeros and poles are calculated from sage program"""

    Zeros_G = [1, 5]
    Poles_G = [2, 3]

    return Zeros_G, Poles_G


def deadtime():
    """ vector of the deadtime of the system"""
    #individual time delays
    dead_G = np.matrix([[0, -2], [-1, -4]])
    dead_Gd = np.matrix([])

    return dead_G, dead_Gd


def Equation_6_8_output(error_poles_direction, deadtime_if=0):
    #this function will calculate the minimum peak values of S and T if the system has zeros and poles for the output
    #could also be spesified if the system has deadtime

    Zeros_G = Zeros_Poles_RHP()[0]
    Poles_G = Zeros_Poles_RHP()[1]

    #two matrices to save all the RHP zeros and poles directions
    yz_direction = np.matrix(np.zeros([G(0.001).shape[0], len(Zeros_G)]))
    yp_direction = np.matrix(np.zeros([G(0.001).shape[0], len(Poles_G)]))


    for i in range(len(Zeros_G)):

        [U, S, V] = np.linalg.svd(G(Zeros_G[i]+error_poles_direction))
        yz_direction[:, i] = U[:, -1]


    for i in range(len(Poles_G)):
        #error_poles_direction is to to prevent the numerical method from breaking
        [U, S, V] = np.linalg.svd(G(Poles_G[i]+error_poles_direction))
        yp_direction[:, i] = U[:, 0]


    yz_mat1 = np.matrix(np.diag(Zeros_G))*np.matrix(np.ones([len(Zeros_G), len(Zeros_G)]))
    yz_mat2 = yz_mat1.T

    Qz = (yz_direction.H*yz_direction)/(yz_mat1+yz_mat2)

    yp_mat1 = np.matrix(np.diag(Poles_G))*np.matrix(np.ones([len(Poles_G), len(Poles_G)]))
    yp_mat2 = yp_mat1.T

    Qp = (yp_direction.H*yp_direction)/(yp_mat1+yp_mat2)

    yzp_mat1 = np.matrix(np.diag(Zeros_G))*np.matrix(np.ones([len(Zeros_G), len(Poles_G)]))
    yzp_mat2 = np.matrix(np.ones([len(Zeros_G), len(Poles_G)]))*np.matrix(np.diag(Poles_G))

    Qzp = yz_direction.H*yp_direction/(yzp_mat1-yzp_mat2)

    if deadtime_if==0:
        #this matrix is the matrix from which the SVD is going to be done to determine the final minimum peak
        pre_mat = (sc_lin.sqrtm((np.linalg.inv(Qz)))*Qzp*(sc_lin.sqrtm(np.linalg.inv(Qp))))

        #final calculation for the peak value
        Ms_min = np.sqrt(1+(np.max(np.linalg.svd(pre_mat)[1]))**2)
        print ''
        print 'Minimum peak values on T and S without deadtime'
        print 'Ms_min = Mt_min = ', Ms_min
        print ''

    #skogestad eq 6-16 pg 226 using maximum deadtime per output channel to give tightest lowest bounds
    if deadtime_if == 1:
        #create vector to be used for the diagonal deadtime matrix containing each outputs' maximum dead time
        #this would ensure tighter bounds on T and S
        #the minimum function is used because all stable systems has dead time with a negative sign

        dead_time_vec_max_row = np.zeros(deadtime()[0].shape[0])

        for i in range(deadtime()[0].shape[0]):
            dead_time_vec_max_row[i] = np.max(deadtime()[0][i, :])


        def Dead_time_matrix(s, dead_time_vec_max_row):

            dead_time_matrix = np.diag(np.exp(np.multiply(dead_time_vec_max_row, s)))
            return dead_time_matrix

        Q_dead = np.zeros([G(0.0001).shape[0], G(0.0001).shape[0]])

        for j in range(len(Poles_G)):
            for j in range(len(Poles_G)):
                denominator_mat= np.transpose(np.conjugate(yp_direction[:, i]))*Dead_time_matrix(Poles_G[i], dead_time_vec_max_row)*Dead_time_matrix(Poles_G[j], dead_time_vec_max_row)*yp_direction[:, j]
                numerator_mat = Poles_G[i]+Poles_G[i]

                Q_dead[i, j] = denominator_mat/numerator_mat

        #calculating the Mt_min with dead time
        lambda_mat = sc_lin.sqrtm(np.linalg.pinv(Q_dead))*(Qp+Qzp*np.linalg.pinv(Qz)*(np.transpose(np.conjugate(Qzp))))*sc_lin.sqrtm(np.linalg.pinv(Q_dead))

        Ms_min=np.real(np.max(np.linalg.eig(lambda_mat)[0]))
        print ''
        print 'Minimum peak values on T and S without dead time'
        print 'Dead time per output channel is for the worst case dead time in that channel'
        print 'Ms_min = Mt_min = ', Ms_min
        print ''
    
    return Ms_min

def Equation_6_8_input(error_poles_direction, deadtime_if=0):
    #this function will calculate the minimum peak values of S and T if the system has zeros and poles for the input
    #could also be specified if the system has deadtime

    [Zeros_G, Poles_G, Zeros_Gd, Poles_Gd] = Zeros_Poles_RHP()
    
    #two matrices to save all the RHP zeros and poles directions
    uz_direction = np.matrix(np.zeros([G(0.001).shape[0], len(Zeros_G)]))
    up_direction = np.matrix(np.zeros([G(0.001).shape[0], len(Poles_G)]))


    for i in range(len(Zeros_G)):

        [U, S, V] = np.linalg.svd(G(Zeros_G[i]+error_poles_direction))
        uz_direction[:, i] = V[:, -1]


    for i in range(len(Poles_G)):
        #error_poles_direction is to to prevent the numerical method from breaking
        [U, S, V] = np.linalg.svd(G(Poles_G[i]+error_poles_direction))
        up_direction[:, i] = V[:, 0]
    

    yz_mat1 = np.matrix(np.diag(Zeros_G))*np.matrix(np.ones([len(Zeros_G), len(Zeros_G)]))
    yz_mat2 = yz_mat1.T

    Qz = (uz_direction.H*uz_direction)/(yz_mat1+yz_mat2)

    yp_mat1 = np.matrix(np.diag(Poles_G))*np.matrix(np.ones([len(Poles_G), len(Poles_G)]))
    yp_mat2 = yp_mat1.T

    Qp = (up_direction.H*up_direction)/(yp_mat1+yp_mat2)

    yzp_mat1 = np.matrix(np.diag(Zeros_G))*np.matrix(np.ones([len(Zeros_G), len(Poles_G)]))
    yzp_mat2 = np.matrix(np.ones([len(Zeros_G), len(Poles_G)]))*np.matrix(np.diag(Poles_G))

    Qzp = uz_direction.H*up_direction/(yzp_mat1-yzp_mat2)

    if deadtime_if==0:
        #this matrix is the matrix from which the SVD is going to be done to determine the final minimum peak
        pre_mat = (sc_lin.sqrtm((np.linalg.inv(Qz)))*Qzp*(sc_lin.sqrtm(np.linalg.inv(Qp))))

        #final calculation for the peak value
        Ms_min = np.sqrt(1+(np.max(np.linalg.svd(pre_mat)[1]))**2)
        print ''
        print 'Minimum peak values on T and S without deadtime'
        print 'Ms_min = Mt_min = ', Ms_min
        print ''

    #skogestad eq 6-16 pg 226 using maximum deadtime per output channel to give tightest lowest bounds
    if deadtime_if == 1:
        #create vector to be used for the diagonal deadtime matrix containing each outputs' maximum dead time
        #this would ensure tighter bounds on T and S
        #the minimum function is used because all stable systems has dead time with a negative sign

        dead_time_vec_max_row = np.zeros(deadtime()[0].shape[0])

        for i in range(deadtime()[0].shape[0]):
            dead_time_vec_max_row[i] = np.max(deadtime()[0][i, :])


        def Dead_time_matrix(s, dead_time_vec_max_row):

            dead_time_matrix = np.diag(np.exp(np.multiply(dead_time_vec_max_row, s)))
            return dead_time_matrix

        Q_dead = np.zeros([G(0.0001).shape[0], G(0.0001).shape[0]])

        for j in range(len(Poles_G)):
            for j in range(len(Poles_G)):
                denominator_mat= np.transpose(np.conjugate(up_direction[:, i]))*Dead_time_matrix(Poles_G[i], dead_time_vec_max_row)*Dead_time_matrix(Poles_G[j], dead_time_vec_max_row)*up_direction[:, j]
                numerator_mat = Poles_G[i]+Poles_G[i]

                Q_dead[i, j] = denominator_mat/numerator_mat

        #calculating the Mt_min with dead time
        lambda_mat = sc_lin.sqrtm(np.linalg.pinv(Q_dead))*(Qp+Qzp*np.linalg.pinv(Qz)*(np.transpose(np.conjugate(Qzp))))*sc_lin.sqrtm(np.linalg.pinv(Q_dead))

        Ms_min=np.real(np.max(np.linalg.eig(lambda_mat)[0]))
        print ''
        print 'Minimum peak values on T and S without dead time'
        print 'Dead time per output channel is for the worst case dead time in that channel'
        print 'Ms_min = Mt_min = ', Ms_min
        print ''
    
    return Ms_min

def Equation_6_24(error_poles_direction):
    #this function if for equation 6-24 in Skogestad
    #this calculate the peak value for KS transfer function using the stable version of the plant
    #only done for a plant that has poles

    Poles_G = Zeros_Poles_RHP()[1]

    KS_PEAK = [np.linalg.norm(np.linalg.svd(G_s(RHP_p+error_poles_direction))[2][:, 0].H*np.linalg.pinv(G_s(RHP_p+error_poles_direction)), 2) for RHP_p in Poles_G]
    KS_max = np.max(KS_PEAK)

    print 'Lower bound on K'
    print 'KS needs to larger than ', KS_max
    print ''
    return KS_max

def Equation_6_43(w_start, w_end):
    #this function is equation 6-43 from Skogestad
    #this function calculated the distrunbance condition number 
    #this number is ideally close to one 
    #if this number is close to the condition number of the plant means that the 
    #disturbances has a greater ability to effect the output

    w=np.logspace(w_start, w_end, 1000)

    dist_condition_num = [np.linalg.svd(G(1j*w_i))[1][0]*np.linalg.svd(np.linalg.pinv(G(1j*w_i))[1][0]*np.linalg.svd(Gd(1j*w_i))[1][0]*np.linalg.svd(Gd(1j*w_i))[0][:, 0])[1][0] for w_i in w]

    condition_num = [np.linalg.svd(G(1j*w_i))[1][0]*np.linalg.svd(np.linalg.pinv(G(1j*w_i)))[1][0] for w_i in w]

    plt.figure(1)
    plt.loglog(w, dist_condition_num, 'r-')
    plt.loglog(w, condition_num, 'b')
    plt.show()

    return dist_condition_num, condition_num

def Equation_6_48(error_zeros_direction):
    #equation 6-48 from Skogestad 
    #checking system's zeros alignment with the disturbacne matrix

    Zeros_G = Zeros_Poles_RHP()[0]

    RHP_alignment = [np.abs(np.linalg.svd(G(RHP_Z+error_zeros_direction))[0][:, 0].H*np.linalg.svd(Gd(RHP_Z+error_zeros_direction))[1][0]*np.linalg.svd(Gd(RHP_Z+error_zeros_direction))[0][:, 0]) for RHP_Z in Zeros_G]

    print 'Checking alignment of process output zeros to disturbances'
    print 'These values should be less than 1'
    print RHP_alignment
    print ''

def Equation_6_50(w_star, w_end):
    #equation 6-50 from Skogestad 
    #checking input saturation for disturbance 
    #the one is for simltaneous disturbances 
    #the other is for a single disturbance
    
    [rows columns]= np.shape(Gd(0.0001))
    
    for column in columns: 
        