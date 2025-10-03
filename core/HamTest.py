import numpy as np
from numpy import exp, pi, sqrt

from file_python.condition import pbc


def HamTNN_test(band: int, alattice: float, p: int, q: int, kx: float, ky: float, IM: dict):
    # matt, alattice, e1, e2, t0, t1, t2, t11, t12, t22 = para(argument)
    eta = 2 * p / q
    alpha = 1 / 2 * kx * alattice
    beta = sqrt(3) / 2 * ky * alattice

    E0 = IM["NN"][0]
    h1 = IM["NN"][1]
    h2 = IM["NN"][2]
    h3 = IM["NN"][3]
    h4 = IM["NN"][4]
    h5 = IM["NN"][5]
    h6 = IM["NN"][6]
    o1 = IM["TNN"][0]
    o2 = IM["TNN"][1]
    o3 = IM["TNN"][2]
    o4 = IM["TNN"][3]
    o5 = IM["TNN"][4]
    o6 = IM["TNN"][5]
    v1 = IM["NNN"][0]
    v2 = IM["NNN"][1]
    v3 = IM["NNN"][2]
    v4 = IM["NNN"][3]
    v5 = IM["NNN"][4]
    v6 = IM["NNN"][5]

    H0 = np.zeros([q, q], dtype=complex)
    H1 = np.zeros([q, q], dtype=complex)
    H1T = np.zeros([q, q], dtype=complex)
    H2 = np.zeros([q, q], dtype=complex)
    H2T = np.zeros([q, q], dtype=complex)
    H11 = np.zeros([q, q], dtype=complex)
    H22 = np.zeros([q, q], dtype=complex)
    H12 = np.zeros([q, q], dtype=complex)
    H12T = np.zeros([q, q], dtype=complex)
    H = np.zeros([3 * q, 3 * q], dtype=complex)

    for m in range(0, q):

        H0[m, m] = E0[0, 0] + h1[0, 0] * exp(2j * pi * eta * m) + h4[0, 0] * exp(-2j * pi * eta * m) + o1[0, 0] * exp(4j * pi * eta * m) + o4[0, 0] * exp(-4j * pi * eta * m)
        H0[m, pbc(m + 1, q)] = h5[0, 0] * exp(-1j * pi * eta * (m + 1 / 2)) + h6[0, 0] * exp(1j * pi * eta * (m + 1 / 2)) + v4[0, 0] * exp(-3j * pi * eta * (m + 1 / 2)) + v6[0, 0] * exp(3j * pi * eta * (m + 1 / 2))
        H0[m, pbc(m + 2, q)] = o5[0, 0] * exp(-2j * pi * eta * (m + 1)) + v5[0, 0] * exp(0) + o6[0, 0] * exp(2j * pi * eta * (m + 1))
        H0[m, pbc(m - 1, q)] = v1[0, 0] * exp(3j * pi * eta * (m - 1 / 2)) + h2[0, 0] * exp(1j * pi * eta * (m - 1 / 2)) + h3[0, 0] * exp(-1j * pi * eta * (m - 1 / 2)) + v3[0, 0] * exp(-3j * pi * eta * (m - 1 / 2))
        H0[m, pbc(m - 2, q)] = o2[0, 0] * exp(2j * pi * eta * (m - 1)) + v2[0, 0] * exp(0) + o3[0, 0] * exp(-2j * pi * eta * (m - 1))

    if band == 1:
        return H0
