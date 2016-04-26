from __future__ import print_function
import numpy as np
import scipy.linalg as sc_lin
import matplotlib.pyplot as plt

from utils import poles, zeros
from utilsplot import plot_freq_subplot, ref_perfect_const_plot


# TODO redefine this function with utils and utilsplot functions
def PEAK_MIMO(w_start, w_end, error_poles_direction, wr, deadtime_if=0):
    '''
    This function is for multivariable system analysis of controllability.
    gives:
    minimum peak values on S and T with or without deadtime
    R is the expected worst case reference change, with condition that ||R||2<= 2
    wr is the frequency up to where reference tracking is required
    enter value of 1 in deadtime_if if system has dead time
    
    Parameters
    ----------
    var : type
        Description (optional).

    Returns
    -------
    var : type
        Description.
    '''

    # TODO use mimotf functions
    Zeros_G = zeros(G)
    Poles_G = poles(G)
    print('Poles: ' , Zeros_G)
    print('Zeros: ' , Poles_G)

    #just to save unnecessary calculations that is not needed
    #sensitivity peak of closed loop. eq 6-8 pg 224 skogestad

    if np.sum(Zeros_G)!= 0:
        if np.sum(Poles_G)!= 0:

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
                print('')
                print('Minimum peak values on T and S without deadtime')
                print('Ms_min = Mt_min = ', Ms_min)
                print('')

            #Skogestad eq 6-16 pg 226 using maximum deadtime per output channel to give tightest lowest bounds
            if deadtime_if == 1:
                #create vector to be used for the diagonal deadtime matrix containing each outputs' maximum dead time
                #this would ensure tighter bounds on T and S
                #the minimum function is used because all stable systems have dead time with a negative sign

                dead_time_vec_max_row = np.zeros(deadtime()[0].shape[0])

                for i in range(deadtime()[0].shape[0]):
                    dead_time_vec_max_row[i] = np.max(deadtime()[0][i, :])


                def Dead_time_matrix(s, dead_time_vec_max_row):

                    dead_time_matrix = np.diag(np.exp(np.multiply(dead_time_vec_max_row, s)))
                    return dead_time_matrix

                Q_dead = np.zeros([G(0.0001).shape[0], G(0.0001).shape[0]])

                for i in range(len(Poles_G)):
                    for j in range(len(Poles_G)):
                        denominator_mat= np.transpose(np.conjugate(yp_direction[:, i]))*Dead_time_matrix(Poles_G[i], dead_time_vec_max_row)*Dead_time_matrix(Poles_G[j], dead_time_vec_max_row)*yp_direction[:, j]
                        numerator_mat = Poles_G[i]+Poles_G[i]

                        Q_dead[i, j] = denominator_mat/numerator_mat

                #calculating the Mt_min with dead time
                lambda_mat = sc_lin.sqrtm(np.linalg.pinv(Q_dead))*(Qp+Qzp*np.linalg.pinv(Qz)*(np.transpose(np.conjugate(Qzp))))*sc_lin.sqrtm(np.linalg.pinv(Q_dead))

                Ms_min=np.real(np.max(np.linalg.eig(lambda_mat)[0]))
                print('')
                print('Minimum peak values on T and S without dead time')
                print('Dead time per output channel is for the worst case dead time in that channel')
                print('Ms_min = Mt_min = ', Ms_min)
                print('')

        else:
            print('')
            print('Minimum peak values on T and S')
            print('No limits on minimum peak values')
            print('')

    #check for dead time
    #dead_G = deadtime[0]
    #dead_gd = deadtime[1]

    #if np.sum(dead_G)!= 0:
        #therefore deadtime is present in the system therefore extra precautions need to be taken
        #manually set up the dead time matrix

    #    dead_m = np.zeros([len(Poles_G), len(Poles_G)])


    #    for i in range(len(Poles_G)):
    #        for j in range(len(Poles_G))
    #            dead_m

    #eq 6-48 pg 239 for plant with RHP zeros
    #checking alignment of disturbances and RHP zeros
    RHP_alignment = [np.abs(np.linalg.svd(G(RHP_Z+error_poles_direction))[0][:, 0].H*np.linalg.svd(Gd(RHP_Z+error_poles_direction))[1][0]*np.linalg.svd(Gd(RHP_Z+error_poles_direction))[0][:, 0]) for RHP_Z in Zeros_G]

    print('Checking alignment of process output zeros to disturbances')
    print('These values should be less than 1')
    print(RHP_alignment)
    print('')

    #checking peak values of KS eq 6-24 pg 229 np.linalg.svd(A)[2][:, 0]
    #done with less tight lower bounds
    KS_PEAK = [np.linalg.norm(np.linalg.svd(G(RHP_p+error_poles_direction))[2][:, 0].H*np.linalg.pinv(G(RHP_p+error_poles_direction)), 2) for RHP_p in Poles_G]
    KS_max = np.max(KS_PEAK)

    print('Lower bound on K')
    print('KS needs to larger than ', KS_max)
    print('')

    #eq 6-50 pg 240 from Skogestad
    #eg 6-50 pg 240 from Skogestad for simultanious disturbance matrix
    #Checking input saturation for perfect control for disturbance rejection
    #checking for maximum disturbance just at steady state

    [U_gd, S_gd, V_gd] = np.linalg.svd(Gd(0.000001))
    y_gd_max = np.max(S_gd)*U_gd[:, 0]
    mod_G_gd_ss = np.max(np.linalg.inv(G(0.000001))*y_gd_max)



    print('Perfect control input saturation from disturbances')
    print('Needs to be less than 1 ')
    print('Max Norm method')
    print('Checking input saturation at steady state')
    print('This is done by the worse output direction of Gd')
    print(mod_G_gd_ss)
    print('')

    #
    #
    #

    print('Figure 1 is for perfect control for simultaneous disturbances')
    print('All values on each of the graphs should be smaller than 1')
    print('')

    print('Figure 2 is the plot of G**1 gd')
    print('The values of this plot needs to be smaller or equal to 1')
    print('')


    w = np.logspace(w_start, w_end, 100)



    mod_G_gd = np.zeros(len(w))
    mod_G_Gd = np.zeros([np.shape(G(0.0001))[0], len(w)])

    for i in range(len(w)):
        [U_gd, S_gd, V_gd] = np.linalg.svd(Gd(1j*w[i]))
        gd_m = np.max(S_gd)*U_gd[:, 0]
        mod_G_gd[i] = np.max(np.linalg.pinv(G(1j*w[i]))*gd_m)

        mat_G_Gd = np.linalg.pinv(G(w[i]))*Gd(w[i])
        for j in range(np.shape(mat_G_Gd)[0]):
            mod_G_Gd[j, i] = np.max(mat_G_Gd[j, :])

    #def for subplotting all the possible variations of mod_G_Gd


    plot_freq_subplot(plt, w, np.ones([2, len(w)]), 'Perfect control Gd', 'r', 1)
    plot_freq_subplot(plt, w, mod_G_Gd, 'Perfect control Gd', 'b', 1)

    plt.figure(2)
    plt.title('Input Saturation for perfect control |inv(G)*gd|<= 1')
    plt.xlabel('w')
    plt.ylabel('|inv(G)* gd|')
    plt.semilogx(w, mod_G_gd)
    plt.semilogx([w[0], w[-1]], [1, 1])
    plt.semilogx(w[0], 1.1)

    #def G_gd(w):
    #    [U_gd, S_gd, V_gd] = np.linalg.svd(Gd(1j*w))
    #    gd_m = U_gd[:, 0]
    #    mod_G_gd[i] = np.max(np.linalg.inv(G(1j*w))*gd_m)-1
    #    return mod_G_gd

    #w_mod_G_gd_1 = sc_opt.fsolve(G_gd, 0.001)


    #print 'frequencies up to which input saturation would not occur'
    #print w_mod_G_gd_1


    print('Figure 3 is disturbance condition number')
    print('A large number indicates that the disturbance is in a bad direction')
    print('')
    #eq 6-43 pg 238 disturbance condition number
    #this in done over a frequency range to see if there are possible problems at higher frequencies
    #finding yd

    dist_condition_num = [np.linalg.svd(G(w_i))[1][0]*np.linalg.svd(np.linalg.pinv(G(w_i))[1][0]*np.linalg.svd(Gd(w_i))[1][0]*np.linalg.svd(Gd(w_i))[0][:, 0])[1][0] for w_i in w]

    plt.figure(3)
    plt.title('yd Condition number')
    plt.ylabel('condition number')
    plt.xlabel('w')
    plt.loglog(w, dist_condition_num)

    #
    #
    #

    print('Figure 4 is the singular value of an specific output with input and disturbance direction vector')
    print('The solid blue line needs to be large than the red line')
    print('This only needs to be checked up to frequencies where |u**H gd| >1')
    print('')

    #checking input saturation for acceptable control  disturbance rejection
    #equation 6-55 pg 241 in Skogestad
    #checking each singular values and the associated input vector with output direction vector of Gd
    #just for square systems for now

    #revised method including all the possibilities of outputs i
    store_rhs_eq = np.zeros([np.shape(G(0.0001))[0], len(w)])
    store_lhs_eq = np.zeros([np.shape(G(0.0001))[0], len(w)])

    for i in range(len(w)):
        for j in range(np.shape(G(0.0001))[0]):
            store_rhs_eq[j, i] = np.abs(np.linalg.svd(G(w[i]))[2][:, j].H*np.max(np.linalg.svd(Gd(w[i]))[1])*np.linalg.svd(Gd(w[i]))[0][:, 0])-1
            store_lhs_eq[j, i] = sc_lin.svd(G(w[i]))[1][j]

    plot_freq_subplot(plt, w, store_rhs_eq, 'Acceptable control eq6-55', 'r', 4)
    plot_freq_subplot(plt, w, store_lhs_eq, 'Acceptable control eq6-55', 'b', 4)

    #
    #
    #

    print('Figure 5 is to check input saturation for reference changes')
    print('Red line in both graphs needs to be larger than the blue line for values w < wr')
    print('Shows the wr up to where control is needed')
    print('')

    #checking input saturation for perfect control with reference change
    #eq 6-52 pg 241

    #checking input saturation for perfect control with reference change
    #another equation for checking input saturation with reference change
    #eq 6-53 pg 241

    plt.figure(5)
    ref_perfect_const_plot(G, reference_change(), 0.01, w_start, w_end)

    print('Figure 6 is the maximum and minimum singular values of G over a frequency range')
    print('Figure 6 is also the maximum and minimum singular values of Gd over a frequency range')
    print('Blue is the minimum values and Red is the maximum singular values')
    print('Plot of Gd should be smaller than 1 else control is needed at frequencies where Gd is bigger than 1')
    print('')

    #checking input saturation for acceptable control with reference change
    #added check for controllability is the minimum and maximum singular values of system transfer function matrix
    #as a function of frequency
    #condition number added to check for how prone the system would be to uncertainty

    singular_min_G = [np.min(np.linalg.svd(G(1j*w_i))[1]) for w_i in w]
    singular_max_G = [np.max(np.linalg.svd(G(1j*w_i))[1]) for w_i in w]
    singular_min_Gd = [np.min(np.linalg.svd(Gd(1j*w_i))[1]) for w_i in w]
    singular_max_Gd = [np.max(np.linalg.svd(Gd(1j*w_i))[1]) for w_i in w]
    condition_num_G = [np.max(np.linalg.svd(G(1j*w_i))[1])/np.min(np.linalg.svd(G(1j*w_i))[1]) for w_i in w]

    plt.figure(6)
    plt.subplot(311)
    plt.title('min_S(G(jw)) and max_S(G(jw))')
    plt.loglog(w, singular_min_G, 'b')
    plt.loglog(w, singular_max_G, 'r')

    plt.subplot(312)
    plt.title('Condition number of G')
    plt.loglog(w, condition_num_G)

    plt.subplot(313)
    plt.title('min_S(Gd(jw)) and max_S(Gd(jw))')
    plt.loglog(w, singular_min_Gd, 'b')
    plt.loglog(w, singular_max_Gd, 'r')
    plt.loglog([w[0], w[-1]], [1, 1])


    plt.show()

    return Ms_min


if __name__ == '__main__': # only executed when called directly, not executed when imported

    def G(s):
        """ give the transfer matrix of the system"""
        return np.matrix([[1 / (s + 1), 1],
                          [1 / (s + 2) ** 2, (s - 1) / (s + 2)]])

    def Gd(s):
        return np.matrix([[1 / (s + 1)],
                          [1 / (s + 20)]])

    def reference_change():
        """Reference change matrix/vector for use in eq 6-52 pg 242 to check input saturation"""
        R = np.matrix([[1, 0], [0, 1]])
        return R/np.linalg.norm(R, 2)


    def deadtime():
        """ vector of the deadtime of the system"""
        #individual time delays
        dead_G = np.matrix([[0, -2], [-1, -4]])
        dead_Gd = np.matrix([])
        return dead_G, dead_Gd


    PEAK_MIMO(-4, 5, 0.00001, 0.1, 1)
