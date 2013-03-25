import matplotlib.pyplot as plt
import scipy.linalg as sc_lin
import numpy as np

#this program contains all the important equations of chapter 6 of Skogestad pre-programmed
#the first set of function is of the specific system

def G(s):
    """ give the transfer matrix of the system"""
    G = np.matrix([[100,102],[100, 100]])
    return G

def Gd(s):
    Gd = 1/(s+1)*np.matrix([[10],[10]])
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

    Zeros_G = [0.6861, 2.0000]
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

    dist_condition_num = [np.linalg.svd(G(1j*w_i))[1][0]*np.linalg.svd(np.linalg.pinv(G(1j*w_i))*np.linalg.svd(Gd(1j*w_i))[1][0]*np.linalg.svd(Gd(1j*w_i))[0][:, 0])[1][0] for w_i in w]
    condition_num = [np.linalg.svd(G(1j*w_i))[1][0]*np.linalg.svd(np.linalg.pinv(G(1j*w_i)))[1][0] for w_i in w]

    plt.figure(1)
    plt.loglog(w, dist_condition_num, 'r-')
    plt.loglog(w, condition_num, 'b')
    plt.show()

    return dist_condition_num, condition_num

#Equation_6_43(-2, 2)

def Equation_6_48(error_zeros_direction):
    #equation 6-48 from Skogestad 
    #checking system's zeros alignment with the disturbacne matrix

    Zeros_G = Zeros_Poles_RHP()[0]
    [columns, rows]=np.shape(Gd(0.001))
    print 'Checking alignment of process output zeros to disturbances'
    print 'These values should be less than 1'
    RHP_alignment = np.array([np.abs(np.linalg.svd(G(RHP_Z+error_zeros_direction))[0][:, -1].H*np.linalg.svd(Gd(RHP_Z+error_zeros_direction)[:,c])[1][0]*np.linalg.svd(Gd(RHP_Z+error_zeros_direction)[:,c])[0][:, 0]) for RHP_Z in Zeros_G for c in range(columns)])
    print RHP_alignment
    print ''

#Equation_6_48(0.00001)

def Equation_6_50(w_start, w_end, matrix='single'):
    #if matrix is equal to single then just a single disturbance in Gd is considered
    #if matrix is equal to multiple then the full Gd matrix is considered

    #equation 6-50 from Skogestad 
    #checking input saturation for disturbance 
    #the one is for simltaneous disturbances 
    #the other is for a single disturbance

    print 'The blue line needs to be smaller than 1'
    print 'This is for perfect control with disturbance rejection'

    w=np.logspace(w_start, w_end, 10)

    if matrix =='single':
        columns= np.shape(Gd(0.0001))[1]

        count = 1
        for column in range(columns):
            mod_invG_gd=[np.max(np.linalg.pinv(G(1j*w_i))*Gd(1j*w_i)[:, column]) for w_i in w]
            plt.figure(count)
            plt.semilogx(w, mod_invG_gd, 'b')
            plt.semilogx([w[0], w[-1]], [1, 1], 'r')
            count=count+1


    if matrix =='multiple':
        mod_invG_Gd=[np.max(np.linalg.pinv(G(1j*w_i))*Gd(1j*w_i)) for w_i in w]
        plt.semilogx(w, mod_invG_Gd,'b')
        plt.semilogx([w[0], w[-1]], [1, 1], 'r')
    plt.show()

def Equation_6_52(w_start, w_end, R, wr, type_eq='minimal'):
    #this is an function is from equation 6-52 and 6-53 from skogestad
    #the one is the minimal requirement for input saturation check in terms of set point tracking 
    #the other is the more tighter bounds and check 
    #type is to spesify if the minimal requiremant wants to be check 
    #tighter for the more tighter bound for set point tracking
    
    print 'All the plots in this function needs to be larger than 1'
    print 'If the values on the plot is not larger than 1 then '
    print 'input saturation would occur'

    w=np.logspace(w_start, w_end, 1000)

    if type_eq=='minimal':
        min_singular = [np.linalg.svd(G(1j*w_i))[1][-1] for w_i in w]
        plt.loglog(w, min_singular, 'b')
        plt.loglog([w[0], w[-1]], [1, 1], 'r')
        plt.loglog(w[0], 1.1)
        plt.loglog(w[0], 0.8)

    if type_eq=='tighter':
        min_singular_invR_G = np.array([np.linalg.svd(np.linalg.pinv(R)*G(1j*w_i))[1][-1] for w_i in w])
        plt.loglog(w, min_singular_invR_G, 'b')
        plt.loglog([w[0], w[-1]], [1, 1], 'r')
        plt.loglog([wr, wr], [0.8*np.min(min_singular_invR_G), 1.2*np.max(min_singular_invR_G)], 'g')
        plt.loglog(w[0], 1.1)
        plt.loglog(w[0], 0.8)

    plt.show()

#Equation_6_52(-4, 4, reference_change(), 0.001, 'tighter')

def Equation_6_55(w_start, w_end):
    #this equation is from Skogestad 6-55
    #this is for acceptable control with disturbance rejection

    print 'For the graphs thats generated the blue line needs to be above the red line'
    print 'If the red line is above, input saturation is going to occur for acceptable control with load rejection'
    w=np.logspace(w_start, w_end, 1000)

    columns_Gd= np.shape(Gd(0.0001))[1]
    columns_G=np.shape(G(0.0001))[1]

    count_G=0
    count_Gd = 0

    for column in range(columns_Gd):
        count_Gd=count_Gd+1
        count_G=0

        for column_G in range(columns_G):
            lhs_eq = np.zeros([len(w), 1])
            rhs_eq = np.zeros([len(w), 1])

            for i in range(len(w)):
                lhs_eq[i, :] = np.abs(np.linalg.svd(G(1j*w[i]))[2][:, column_G].H*Gd(1j*w[i])[:, column])-1
                rhs_eq[i, :] = np.linalg.svd(G(1j*w[i]))[1][column_G]

            count_G = count_G+1
            plt.figure(count_Gd)
            plt.subplot(columns_G, 1, count_G)
            plt.semilogx(w, lhs_eq, 'r')
            plt.semilogx(w, rhs_eq, 'b')

    plt.show()


#Equation_6_55(-4, 4)

