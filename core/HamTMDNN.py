import numpy as np
from numpy import exp, pi, sqrt
from numpy.typing import NDArray

from core.condition import pbc


def HamTNN(alattice, p, q, k, lambd, IM) -> NDArray[np.complex128]:
    eta = p / q
    alpha = 1 / 2 * k[0] * alattice
    beta = sqrt(3) / 2 * k[1] * alattice

    hR1 = exp(1j * 2 * alpha)
    hR2 = exp(1j * (alpha - beta))
    hR3 = exp(1j * (-alpha - beta))
    hR4 = exp(-1j * 2 * alpha)
    hR5 = exp(1j * (-alpha + beta))
    hR6 = exp(1j * (alpha + beta))

    vR1 = exp(1j * (3 * alpha - beta))
    vR2 = exp(1j * (-2 * beta))
    vR3 = exp(1j * (-3 * alpha - beta))
    vR4 = exp(1j * (-3 * alpha + beta))
    vR5 = exp(1j * (2 * beta))
    vR6 = exp(1j * (3 * alpha + beta))

    oR1 = exp(1j * 4 * alpha)
    oR2 = exp(2j * (alpha - beta))
    oR3 = exp(2j * (-alpha - beta))
    oR4 = exp(-1j * 4 * alpha)
    oR5 = exp(2j * (-alpha + beta))
    oR6 = exp(2j * (alpha + beta))

    ER0 = IM["NN"][0]
    ER1 = IM["NN"][1] * hR1
    ER2 = IM["NN"][2] * hR2
    ER3 = IM["NN"][3] * hR3
    ER4 = IM["NN"][4] * hR4
    ER5 = IM["NN"][5] * hR5
    ER6 = IM["NN"][6] * hR6

    o1 = IM["TNN"][0] * oR1
    o2 = IM["TNN"][1] * oR2
    o3 = IM["TNN"][2] * oR3
    o4 = IM["TNN"][3] * oR4
    o5 = IM["TNN"][4] * oR5
    o6 = IM["TNN"][5] * oR6

    v1 = IM["NNN"][0] * vR1
    v2 = IM["NNN"][1] * vR2
    v3 = IM["NNN"][2] * vR3
    v4 = IM["NNN"][3] * vR4
    v5 = IM["NNN"][4] * vR5
    v6 = IM["NNN"][5] * vR6

    h0 = np.zeros([q, q], dtype=complex)
    h1 = np.zeros([q, q], dtype=complex)
    h1T = np.zeros([q, q], dtype=complex)
    h2 = np.zeros([q, q], dtype=complex)
    h2T = np.zeros([q, q], dtype=complex)
    h11 = np.zeros([q, q], dtype=complex)
    h22 = np.zeros([q, q], dtype=complex)
    h12 = np.zeros([q, q], dtype=complex)
    h12T = np.zeros([q, q], dtype=complex)

    for m in range(0, q):
        h0[m, m] = (
            ER0[0, 0]
            + v2[0, 0] * exp(-4j * pi * m * eta)
            + v5[0, 0] * exp(4j * pi * m * eta)
        )
        h0[m, pbc(m + 1, q)] = ER2[0, 0] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) + ER6[
            0, 0
        ] * exp(1j * 2 * pi * (m + 1 / 2) * eta)
        h0[m, pbc(m + 2, q)] = (
            ER1[0, 0]
            + o2[0, 0] * exp(-4j * pi * (m + 1) * eta)
            + o6[0, 0] * exp(4j * pi * (m + 1) * eta)
        )
        h0[m, pbc(m + 3, q)] = v1[0, 0] * exp(-1j * 2 * pi * (m + 3 / 2) * eta) + v6[
            0, 0
        ] * exp(1j * 2 * pi * (m + 3 / 2) * eta)
        h0[m, pbc(m + 4, q)] = o1[0, 0]
        h0[m, pbc(m - 1, q)] = ER5[0, 0] * exp(1j * 2 * pi * (m - 1 / 2) * eta) + ER3[
            0, 0
        ] * exp(-1j * 2 * pi * (m - 1 / 2) * eta)
        h0[m, pbc(m - 2, q)] = (
            ER4[0, 0]
            + o3[0, 0] * exp(-4j * pi * (m - 1) * eta)
            + o5[0, 0] * exp(4j * pi * (m - 1) * eta)
        )
        h0[m, pbc(m - 3, q)] = v3[0, 0] * exp(-1j * 2 * pi * (m - 3 / 2) * eta) + v4[
            0, 0
        ] * exp(1j * 2 * pi * (m - 3 / 2) * eta)
        h0[m, pbc(m - 4, q)] = o4[0, 0]

        h11[m, m] = (
            ER0[1, 1]
            + v2[1, 1] * exp(-4j * pi * m * eta)
            + v5[1, 1] * exp(4j * pi * m * eta)
        )
        h11[m, pbc(m + 1, q)] = ER2[1, 1] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) + ER6[
            1, 1
        ] * exp(1j * 2 * pi * (m + 1 / 2) * eta)
        h11[m, pbc(m + 2, q)] = (
            ER1[1, 1]
            + o2[1, 1] * exp(-4j * pi * (m + 1) * eta)
            + o6[1, 1] * exp(4j * pi * (m + 1) * eta)
        )
        h11[m, pbc(m + 3, q)] = v1[1, 1] * exp(-1j * 2 * pi * (m + 3 / 2) * eta) + v6[
            1, 1
        ] * exp(1j * 2 * pi * (m + 3 / 2) * eta)
        h11[m, pbc(m + 4, q)] = o1[1, 1]
        h11[m, pbc(m - 1, q)] = ER5[1, 1] * exp(1j * 2 * pi * (m - 1 / 2) * eta) + ER3[
            1, 1
        ] * exp(-1j * 2 * pi * (m - 1 / 2) * eta)
        h11[m, pbc(m - 2, q)] = (
            ER4[1, 1]
            + o3[1, 1] * exp(-4j * pi * (m - 1) * eta)
            + o5[1, 1] * exp(4j * pi * (m - 1) * eta)
        )
        h11[m, pbc(m - 3, q)] = v3[1, 1] * exp(-1j * 2 * pi * (m - 3 / 2) * eta) + v4[
            1, 1
        ] * exp(1j * 2 * pi * (m - 3 / 2) * eta)
        h11[m, pbc(m - 4, q)] = o4[1, 1]

        h12T[m, m] = (
            ER0[2, 1]
            + v2[2, 1] * exp(4j * pi * m * eta)
            + v5[2, 1] * exp(-4j * pi * m * eta)
        )
        h12T[m, pbc(m + 1, q)] = ER2[2, 1] * exp(1j * 2 * pi * (m + 1 / 2) * eta) + ER6[
            2, 1
        ] * exp(-1j * 2 * pi * (m + 1 / 2) * eta)
        h12T[m, pbc(m + 2, q)] = (
            ER1[2, 1]
            + o2[2, 1] * exp(4j * pi * (m + 1) * eta)
            + o6[2, 1] * exp(-4j * pi * (m + 1) * eta)
        )
        h12T[m, pbc(m + 3, q)] = v1[2, 1] * exp(1j * 2 * pi * (m + 3 / 2) * eta) + v6[
            2, 1
        ] * exp(-1j * 2 * pi * (m + 3 / 2) * eta)
        h12T[m, pbc(m + 4, q)] = o1[2, 1]
        h12T[m, pbc(m - 1, q)] = ER5[2, 1] * exp(
            -1j * 2 * pi * (m - 1 / 2) * eta
        ) + ER3[2, 1] * exp(1j * 2 * pi * (m - 1 / 2) * eta)
        h12T[m, pbc(m - 2, q)] = (
            ER4[2, 1]
            + o3[2, 1] * exp(4j * pi * (m - 1) * eta)
            + o5[2, 1] * exp(-4j * pi * (m - 1) * eta)
        )
        h12T[m, pbc(m - 3, q)] = v3[2, 1] * exp(1j * 2 * pi * (m - 3 / 2) * eta) + v4[
            2, 1
        ] * exp(-1j * 2 * pi * (m - 3 / 2) * eta)
        h12T[m, pbc(m - 4, q)] = o4[2, 1]

        h12[m, m] = (
            ER0[1, 2]
            + v2[1, 2] * exp(-4j * pi * m * eta)
            + v5[1, 2] * exp(4j * pi * m * eta)
        )
        h12[m, pbc(m + 1, q)] = ER2[1, 2] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) + ER6[
            1, 2
        ] * exp(1j * 2 * pi * (m + 1 / 2) * eta)
        h12[m, pbc(m + 2, q)] = (
            ER1[1, 2]
            + o2[1, 2] * exp(-4j * pi * (m + 1) * eta)
            + o6[1, 2] * exp(4j * pi * (m + 1) * eta)
        )
        h12[m, pbc(m + 3, q)] = v1[1, 2] * exp(-1j * 2 * pi * (m + 3 / 2) * eta) + v6[
            1, 2
        ] * exp(1j * 2 * pi * (m + 3 / 2) * eta)
        h12[m, pbc(m + 4, q)] = o1[1, 2]
        h12[m, pbc(m - 1, q)] = ER5[1, 2] * exp(1j * 2 * pi * (m - 1 / 2) * eta) + ER3[
            1, 2
        ] * exp(-1j * 2 * pi * (m - 1 / 2) * eta)
        h12[m, pbc(m - 2, q)] = (
            ER4[1, 2]
            + o3[1, 2] * exp(-4j * pi * (m - 1) * eta)
            + o5[1, 2] * exp(4j * pi * (m - 1) * eta)
        )
        h12[m, pbc(m - 3, q)] = v3[1, 2] * exp(-1j * 2 * pi * (m - 3 / 2) * eta) + v4[
            1, 2
        ] * exp(1j * 2 * pi * (m - 3 / 2) * eta)
        h12[m, pbc(m - 4, q)] = o4[1, 2]

        h1T[m, m] = (
            ER0[1, 0]
            + v2[1, 0] * exp(4j * pi * m * eta)
            + v5[1, 0] * exp(-4j * pi * m * eta)
        )
        h1T[m, pbc(m + 1, q)] = ER2[1, 0] * exp(1j * 2 * pi * (m + 1 / 2) * eta) + ER6[
            1, 0
        ] * exp(-1j * 2 * pi * (m + 1 / 2) * eta)
        h1T[m, pbc(m + 2, q)] = (
            ER1[1, 0]
            + o2[1, 0] * exp(4j * pi * (m + 1) * eta)
            + o6[1, 0] * exp(-4j * pi * (m + 1) * eta)
        )
        h1T[m, pbc(m + 3, q)] = v1[1, 0] * exp(1j * 2 * pi * (m + 3 / 2) * eta) + v6[
            1, 0
        ] * exp(-1j * 2 * pi * (m + 3 / 2) * eta)
        h1T[m, pbc(m + 4, q)] = o1[1, 0]
        h1T[m, pbc(m - 1, q)] = ER5[1, 0] * exp(-1j * 2 * pi * (m - 1 / 2) * eta) + ER3[
            1, 0
        ] * exp(1j * 2 * pi * (m - 1 / 2) * eta)
        h1T[m, pbc(m - 2, q)] = (
            ER4[1, 0]
            + o3[1, 0] * exp(4j * pi * (m - 1) * eta)
            + o5[1, 0] * exp(-4j * pi * (m - 1) * eta)
        )
        h1T[m, pbc(m - 3, q)] = v3[1, 0] * exp(1j * 2 * pi * (m - 3 / 2) * eta) + v4[
            1, 0
        ] * exp(-1j * 2 * pi * (m - 3 / 2) * eta)
        h1T[m, pbc(m - 4, q)] = o4[1, 0]

        h1[m, m] = (
            ER0[0, 1]
            + v2[0, 1] * exp(-4j * pi * m * eta)
            + v5[0, 1] * exp(4j * pi * m * eta)
        )
        h1[m, pbc(m + 1, q)] = ER2[0, 1] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) + ER6[
            0, 1
        ] * exp(1j * 2 * pi * (m + 1 / 2) * eta)
        h1[m, pbc(m + 2, q)] = (
            ER1[0, 1]
            + o2[0, 1] * exp(-4j * pi * (m + 1) * eta)
            + o6[0, 1] * exp(4j * pi * (m + 1) * eta)
        )
        h1[m, pbc(m + 3, q)] = v1[0, 1] * exp(-1j * 2 * pi * (m + 3 / 2) * eta) + v6[
            0, 1
        ] * exp(1j * 2 * pi * (m + 3 / 2) * eta)
        h1[m, pbc(m + 4, q)] = o1[0, 1]
        h1[m, pbc(m - 1, q)] = ER5[0, 1] * exp(1j * 2 * pi * (m - 1 / 2) * eta) + ER3[
            0, 1
        ] * exp(-1j * 2 * pi * (m - 1 / 2) * eta)
        h1[m, pbc(m - 2, q)] = (
            ER4[0, 1]
            + o3[0, 1] * exp(-4j * pi * (m - 1) * eta)
            + o5[0, 1] * exp(4j * pi * (m - 1) * eta)
        )
        h1[m, pbc(m - 3, q)] = v3[0, 1] * exp(-1j * 2 * pi * (m - 3 / 2) * eta) + v4[
            0, 1
        ] * exp(1j * 2 * pi * (m - 3 / 2) * eta)
        h1[m, pbc(m - 4, q)] = o4[0, 1]

        h22[m, m] = (
            ER0[2, 2]
            + v2[2, 2] * exp(-4j * pi * m * eta)
            + v5[2, 2] * exp(4j * pi * m * eta)
        )
        h22[m, pbc(m + 1, q)] = ER2[2, 2] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) + ER6[
            2, 2
        ] * exp(1j * 2 * pi * (m + 1 / 2) * eta)
        h22[m, pbc(m + 2, q)] = (
            ER1[2, 2]
            + o2[2, 2] * exp(-4j * pi * (m + 1) * eta)
            + o6[2, 2] * exp(4j * pi * (m + 1) * eta)
        )
        h22[m, pbc(m + 3, q)] = v1[2, 2] * exp(-1j * 2 * pi * (m + 3 / 2) * eta) + v6[
            2, 2
        ] * exp(1j * 2 * pi * (m + 3 / 2) * eta)
        h22[m, pbc(m + 4, q)] = o1[2, 2]
        h22[m, pbc(m - 1, q)] = ER5[2, 2] * exp(1j * 2 * pi * (m - 1 / 2) * eta) + ER3[
            2, 2
        ] * exp(-1j * 2 * pi * (m - 1 / 2) * eta)
        h22[m, pbc(m - 2, q)] = (
            ER4[2, 2]
            + o3[2, 2] * exp(-4j * pi * (m - 1) * eta)
            + o5[2, 2] * exp(4j * pi * (m - 1) * eta)
        )
        h22[m, pbc(m - 3, q)] = v3[2, 2] * exp(-1j * 2 * pi * (m - 3 / 2) * eta) + v4[
            2, 2
        ] * exp(1j * 2 * pi * (m - 3 / 2) * eta)
        h22[m, pbc(m - 4, q)] = o4[2, 2]

        h2T[m, m] = (
            ER0[2, 0]
            + v2[2, 0] * exp(4j * pi * m * eta)
            + v5[2, 0] * exp(-4j * pi * m * eta)
        )
        h2T[m, pbc(m + 1, q)] = ER2[2, 0] * exp(1j * 2 * pi * (m + 1 / 2) * eta) + ER6[
            2, 0
        ] * exp(-1j * 2 * pi * (m + 1 / 2) * eta)
        h2T[m, pbc(m + 2, q)] = (
            ER1[2, 0]
            + o2[2, 0] * exp(4j * pi * (m + 1) * eta)
            + o6[2, 0] * exp(-4j * pi * (m + 1) * eta)
        )
        h2T[m, pbc(m + 3, q)] = v1[2, 0] * exp(1j * 2 * pi * (m + 3 / 2) * eta) + v6[
            2, 0
        ] * exp(-1j * 2 * pi * (m + 3 / 2) * eta)
        h2T[m, pbc(m + 4, q)] = o1[2, 0]
        h2T[m, pbc(m - 1, q)] = ER5[2, 0] * exp(-1j * 2 * pi * (m - 1 / 2) * eta) + ER3[
            2, 0
        ] * exp(1j * 2 * pi * (m - 1 / 2) * eta)
        h2T[m, pbc(m - 2, q)] = (
            ER4[2, 0]
            + o3[2, 0] * exp(4j * pi * (m - 1) * eta)
            + o5[2, 0] * exp(-4j * pi * (m - 1) * eta)
        )
        h2T[m, pbc(m - 3, q)] = v3[2, 0] * exp(1j * 2 * pi * (m - 3 / 2) * eta) + v4[
            2, 0
        ] * exp(-1j * 2 * pi * (m - 3 / 2) * eta)
        h2T[m, pbc(m - 4, q)] = o4[2, 0]

        h2[m, m] = (
            ER0[0, 2]
            + v2[0, 2] * exp(-4j * pi * m * eta)
            + v5[0, 2] * exp(4j * pi * m * eta)
        )
        h2[m, pbc(m + 1, q)] = ER2[0, 2] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) + ER6[
            0, 2
        ] * exp(1j * 2 * pi * (m + 1 / 2) * eta)
        h2[m, pbc(m + 2, q)] = (
            ER1[0, 2]
            + o2[0, 2] * exp(-4j * pi * (m + 1) * eta)
            + o6[0, 2] * exp(4j * pi * (m + 1) * eta)
        )
        h2[m, pbc(m + 3, q)] = v1[0, 2] * exp(-1j * 2 * pi * (m + 3 / 2) * eta) + v6[
            0, 2
        ] * exp(1j * 2 * pi * (m + 3 / 2) * eta)
        h2[m, pbc(m + 4, q)] = o1[0, 2]
        h2[m, pbc(m - 1, q)] = ER5[0, 2] * exp(1j * 2 * pi * (m - 1 / 2) * eta) + ER3[
            0, 2
        ] * exp(-1j * 2 * pi * (m - 1 / 2) * eta)
        h2[m, pbc(m - 2, q)] = (
            ER4[0, 2]
            + o3[0, 2] * exp(-4j * pi * (m - 1) * eta)
            + o5[0, 2] * exp(4j * pi * (m - 1) * eta)
        )
        h2[m, pbc(m - 3, q)] = v3[0, 2] * exp(-1j * 2 * pi * (m - 3 / 2) * eta) + v4[
            0, 2
        ] * exp(1j * 2 * pi * (m - 3 / 2) * eta)
        h2[m, pbc(m - 4, q)] = o4[0, 2]

    ham = np.zeros([3 * q, 3 * q], dtype=complex)

    ham[0:q, 0:q] = h0
    ham[0:q, q : 2 * q] = h1
    ham[0:q, 2 * q : 3 * q] = h2
    ham[q : 2 * q, 0:q] = h1T
    ham[q : 2 * q, q : 2 * q] = h11
    ham[q : 2 * q, 2 * q : 3 * q] = h12
    ham[2 * q : 3 * q, 0:q] = h2T
    ham[2 * q : 3 * q, q : 2 * q] = h12T
    ham[2 * q : 3 * q, 2 * q : 3 * q] = h22

    Iq = np.eye(q, dtype=complex)
    val1 = 1j / np.sqrt(2)
    val2 = -1j / np.sqrt(2)
    val3 = 1 / np.sqrt(2)

    W_big = np.zeros([3 * q, 3 * q], dtype=complex)
    W_big[0:q, 0:q] = Iq
    W_big[q : 2 * q, q : 2 * q] = val1 * Iq
    W_big[q : 2 * q, 2 * q : 3 * q] = val2 * Iq
    W_big[2 * q : 3 * q, q : 2 * q] = val3 * Iq
    W_big[2 * q : 3 * q, 2 * q : 3 * q] = val3 * Iq

    hamu = ham.copy()
    hamu[q : 2 * q, q : 2 * q] += lambd * Iq
    hamu[2 * q : 3 * q, 2 * q : 3 * q] -= lambd * Iq

    hamd = ham.copy()
    hamd[q : 2 * q, q : 2 * q] -= lambd * Iq
    hamd[2 * q : 3 * q, 2 * q : 3 * q] += lambd * Iq

    return ham, hamu, hamd
