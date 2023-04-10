import numpy as np
from numpy.linalg import *
import sys

MAX_SYSTEM_NUMBER = sys.float_info.max
TOLERANCE_MIN = 1e-10       # You can change this value

def conjugate_eliminator(vector):
    '''
    This function elimante one of the complex conjugates from a vector
    INPUT:
        vector: Contains the elements some of them are complex
                and come in pairs of conjugates.
    OTUPUT:
        inde2delete: Contains the indices of the conjugates that will
                    be deleted.
    '''

    index2delete = []
    complex_conjugate_happens = []
    i = 0
    while i < len(vector):
        # Imaginary part is 0:
            # Real part is 0:
                # Yes: append[i] i = i + 1 continue

                # NO: i = i +1

        # Imaginary part not 0.
                    # Is there additional elements above
                            # Are those numbers complex conjugates?
                                # Yes: apend[i] i = i +2
                                # No: i = i + 1 ->Numbers are not compl. conj.

                    # There is not additional elements above.
                            # i = i +1

        if abs(np.real(vector[i])) < TOLERANCE_MIN and abs(np.imag(vector[i])) < TOLERANCE_MIN:
            index2delete.append(i)
            i = i + 1
            continue

        elif abs(np.real(vector[i])) < TOLERANCE_MIN and abs(np.imag(vector[i])) > TOLERANCE_MIN:
            i = i + 1
            continue

        elif abs(np.real(vector[i])) > TOLERANCE_MIN and abs(np.imag(vector[i])) < TOLERANCE_MIN:
            i = i + 1
            continue

        elif abs(np.real(vector[i])) > TOLERANCE_MIN and abs(np.imag(vector[i])) > TOLERANCE_MIN:
            if i + 1 < len(vector): # There are elements above
                if abs(np.real(vector[i]) - np.real(vector[i + 1])) < TOLERANCE_MIN:
                    complex_conjugate_happens.append(i)
                    index2delete.append(i+1)
                    i = i + 2

                else:
                    i = i + 1

            else:
                break


    # It will be transformed into a np array. It was not done before
    # because it was useful the use of append method for lists.
    index2delete = np.array(index2delete)

    return index2delete, complex_conjugate_happens

def hankel_matrix_creator(signal, order, dimension = None):

    '''
    Generates a Hankel matrix of order: "order" from the measurements on
    on "signal"

    INPUT:
        signal: Measurements to generate the Hankel matrix H
        order: Order of the Hankel matrix.
        dimension: Dimension of the Hankel Matrix

    OUTPUT:
        H: Generated Hankel Matrix
    '''

    if dimension == None:
        dimension = len(signal)//2

    H = np.zeros([dimension, dimension])
    for i in range(dimension):
        H[i, :] = signal[i+order: i + order + dimension]

    return H

# Lets build a functions that gets the powers of complex numbers.
def power_constructor(Z, N):
    '''
    This function takes every row of Z and gets the ith power.

    IMPORTANT:  This code just replace "inf" values with the maximum value accepted for the PC.
                It is not considered "nan" values to be replaced or analyzed and I even consider this values
                will never happen.

    INPUT:
        Z: Matrix you are going to get the powers from
        powers: list of integers starting from 0

    OUTPUT:

    '''

    n_rows = N
    n_columns = np.shape(Z)[0]

    Z_powers = np.zeros([n_rows, n_columns], dtype = complex)

    for i in np.arange(N):
        Z_powers[i, :] = np.exp(i*np.log(Z))

    bool_real_infs = abs(np.real(Z_powers)) == np.inf
    bool_imag_infs = abs(np.imag(Z_powers)) == np.inf

    Z_powers.real[bool_real_infs] = np.sign(np.real(Z_powers[bool_real_infs]))*sys.float_info.max
    Z_powers.imag[bool_imag_infs] = np.sign(np.imag(Z_powers[bool_imag_infs]))*sys.float_info.max

    return Z_powers

def power_constructor_direct(Z, N, p):
    Z_powers = np.zeros([N, p], dtype = complex)


    for i in range(N):
        Z_powers[i:] = np.float_power(Z, np.float64(i))


        for j in range(len(Z_powers[i])):
            if np.real(Z_powers[i, j]) == np.inf:
                print(Z[j], Z_powers[i][j], i)



    return Z_powers



def least_squares_direct(Z_powers, y):
    # LEAST SQUARES METHOD 1 TO FIND GAMMA: FORMULA DIRECT APPLICATION.
    gamma = np.dot(np.dot(inv(np.dot(np.transpose(Z_powers), Z_powers)),
                   np.transpose(Z_powers)), np.transpose(y))
    return gamma

def least_squares_qr(Z_powers, y):
    # LEAST SQUARES METHOD 2 TO FIND GAMMA: QR DECOMPOSITION:
    q, r = qr(Z_powers)
    gamma = np.dot(inv(r), np.dot(q.T, np.transpose(y)))

    return gamma

def least_squares_lstsq(Z_powers, signal):
    # LEAST SQUARES METHOD 4 TO FIND GAMMA: lstsq PYTHON FUNCTION
    gamma = lstsq(Z_powers, np.transpose(signal))[0]

    return gamma

def prony_method(signal, fm, **kwargs):
    '''
    INPUT:
       signal:
       fm: Sampling frequency
   '''
    delta_t = 1.0/fm

    if len(kwargs) == 0:
        t = np.arange(0, len(signal))*delta_t

    else: t = np.array(list(kwargs.values()))[0]

    p = len(signal)//2
    N = len(signal)

    H0 = hankel_matrix_creator(signal, 0, p)
    H1 = hankel_matrix_creator(signal, 1, p)

    [U, S, Vt] = svd(H0)
    V = np.transpose(Vt)
    S_diag_power = np.diag(np.power(S, -1.0/2))

    A = np.dot(S_diag_power,
          np.dot(np.transpose(U),
                np.dot(H1,
                      np.dot(V,S_diag_power))))

    [Z, V] = eig(A)

    # Z VALUES REVISION.
    Z.imag[abs(np.imag(Z)) < TOLERANCE_MIN] = 0.0
    Z.real[abs(np.real(Z)) < TOLERANCE_MIN] = 0.0

    lmbds = np.log(Z)/delta_t

    # lmbds VALUES REVISION.
    lmbds.real[abs(np.real(lmbds)) < TOLERANCE_MIN] = 0.0
    lmbds.imag[abs(np.imag(lmbds)) < TOLERANCE_MIN] = 0.0

    [index2delete_lmbds, complex_conjugate_happen] = conjugate_eliminator(lmbds)
    not_repeated_lmbds = np.delete(lmbds, index2delete_lmbds)

    # THIS PART WILL BE SEPARATED LIKE COMMENTS
    # numeric_alpha = np.zeros(len(not_repeated_lmbds))
    # numeric_f = np.zeros(len(not_repeated_lmbds))
    # for i in range(len(not_repeated_lmbds)):
    #    numeric_alpha[i] = -np.real(not_repeated_lmbds[i])
    #    numeric_f[i] = np.imag(not_repeated_lmbds[i])/(2*np.pi)

    Z_powers = power_constructor(Z, N)
    #Z_powers = power_constructor_direct(Z, N, p)

    # Z_powers VALUES REVISION
    Z_powers.real[abs(np.real(Z_powers)) < TOLERANCE_MIN] = 0.0
    Z_powers.imag[abs(np.imag(Z_powers)) < TOLERANCE_MIN] = 0.0

    gamma = least_squares_direct(Z_powers, signal)
    #gamma = least_squares_lstsq(Z_powers, signal)

    # gamma: VALUES REVISION.
    gamma.imag[abs(gamma.imag) < TOLERANCE_MIN] = 0.0
    gamma.real[abs(gamma.real) < TOLERANCE_MIN] = 0.0

    gamma_two_times = gamma
    gamma_two_times[complex_conjugate_happen] = 2*gamma_two_times[complex_conjugate_happen]

    not_repeated_gamma = np.delete(gamma, index2delete_lmbds)

    # THIS PART WILL BE SEPARATED LIKE COMMENTS
    # numeric_A = np.zeros(len(not_repeated_gamma))
    # numeric_theta = np.zeros(len(not_repeated_gamma))
    # for i in range(len(not_repeated_gamma)):
    #    numeric_A[i] = abs(not_repeated_gamma[i])
    #    numeric_theta[i] = np.arctan2(np.imag(not_repeated_gamma[i]), np.real(not_repeated_gamma[i]))

    numeric_signal = np.zeros([N, 1], dtype = complex)

    for i in range(N):
        for j in range(len(not_repeated_gamma[:])):
            numeric_signal[i] = numeric_signal[i] + \
            not_repeated_gamma[j]*np.exp(not_repeated_lmbds[j]*t[i])

    return [numeric_signal, A, Z, not_repeated_lmbds, lmbds, Z_powers, not_repeated_gamma, gamma]
