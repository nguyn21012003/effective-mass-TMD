import csv, os
from datetime import datetime

import numpy as np
import cupy as cp
from cupy import linalg as LA
from numpy import sqrt, exp, pi, sin, cos
from tqdm import tqdm
from numpy.typing import NDArray
import multiprocessing as mp


def paraNN(choice: int, model: str) -> dict:
    choice = int(choice)
    matt = ["MoS2", "WS2", "MoSe2", "WSe2", "MoTe2", "WTe2"]
    dataModel = {
        "LDA": {
            "alattice": [3.129, 3.132, 3.254, 3.253, 3.472, 3.476],
            "e1": [1.238, 1.355, 1.001, 1.124, 0.618, 0.623],
            "e2": [2.366, 2.569, 2.239, 2.447, 2.126, 2.251],
            "t0": [-0.218, -0.238, -0.222, -0.242, -0.202, -0.209],
            "t1": [0.444, 0.626, 0.350, 0.506, 0.254, 0.388],
            "t2": [0.533, 0.557, 0.488, 0.514, 0.423, 0.442],
            "t11": [0.250, 0.324, 0.244, 0.305, 0.241, 0.272],
            "t12": [0.360, 0.405, 0.314, 0.353, 0.263, 0.295],
            "t22": [0.047, -0.076, 0.129, 0.025, 0.269, 0.200],
        },
        "GGA": {
            "alattice": [3.190, 3.191, 3.326, 3.325, 3.357, 3.560],
            "e1": [1.046, 1.130, 0.919, 0.943, 0.605, 0.606],
            "e2": [2.104, 2.275, 2.065, 2.179, 1.972, 2.102],
            "t0": [-0.184, -0.206, -0.188, -0.207, -0.169, -0.175],
            "t1": [0.401, 0.567, 0.317, 0.457, 0.228, 0.342],
            "t2": [0.507, 0.536, 0.456, 0.486, 0.390, 0.410],
            "t11": [0.218, 0.286, 0.211, 0.263, 0.207, 0.233],
            "t12": [0.338, 0.384, 0.290, 0.329, 0.239, 0.270],
            "t22": [0.057, -0.061, 0.130, 0.034, 0.252, 0.190],
        },
    }
    e1 = dataModel[model]["e1"][choice]
    e2 = dataModel[model]["e2"][choice]
    t0 = dataModel[model]["t0"][choice]
    t1 = dataModel[model]["t1"][choice]
    t2 = dataModel[model]["t2"][choice]
    t11 = dataModel[model]["t11"][choice]
    t12 = dataModel[model]["t12"][choice]
    t22 = dataModel[model]["t22"][choice]
    alattice = dataModel[model]["alattice"][choice]

    data = {
        "material": matt[choice],
        "alattice": alattice,
        "e1": e1,
        "e2": e2,
        "t0": t0,
        "t1": t1,
        "t2": t2,
        "t11": t11,
        "t12": t12,
        "t22": t22,
    }

    return data


def paraTNN(choice: int, model: str) -> dict:
    matt = ["MoS2", "WS2", "MoSe2", "WSe2", "MoTe2", "WTe2"]

    dataModel = {
        "LDA": {
            "alattice": [3.129, 3.132, 3.254, 3.253, 3.472, 3.476],
            "e1": [0.820, 0.905, 0.715, 0.860, 0.574, 0.675],
            "e2": [1.931, 2.167, 1.687, 1.892, 1.410, 1.489],
            "t0": [-0.176, -0.175, -0.154, -0.152, -0.148, -0.124],
            "t1": [-0.101, -0.090, -0.134, -0.125, -0.173, -0.159],
            "t2": [0.531, 0.611, 0.437, 0.508, 0.333, 0.362],
            "t11": [0.084, 0.043, 0.124, 0.094, 0.203, 0.196],
            "t12": [0.169, 0.181, 0.119, 0.129, 0.186, 0.101],
            "t22": [0.070, 0.008, 0.072, 0.009, 0.127, 0.044],
            "r0": [0.070, 0.075, 0.048, 0.044, 0.007, -0.009],
            "r1": [-0.252, -0.282, -0.248, -0.278, -0.280, -0.25],
            "r2": [0.084, 0.127, 0.090, 0.129, -0.067, 0.129],
            "r11": [0.019, 0.001, 0.066, 0.059, 0.073, 0.131],
            "r12": [0.093, 0.114, 0.045, 0.058, 0.081, -0.007],
            "u0": [-0.043, -0.063, -0.067, -0.090, -0.054, -0.086],
            "u1": [0.047, 0.047, 0.041, 0.039, 0.008, 0.012],
            "u2": [0.005, 0.004, 0.005, 0.001, 0.037, -0.020],
            "u11": [0.304, 0.374, 0.327, 0.392, 0.145, 0.361],
            "u12": [-0.192, -0.224, -0.194, -0.224, -0.078, -0.193],
            "u22": [-0.162, -0.177, -0.151, -0.165, 0.035, -0.129],
        },
        "GGA": {
            "alattice": [3.190, 3.191, 3.326, 3.325, 3.357, 3.560],
            "e1": [0.683, 0.717, 0.684, 0.728, 0.588, 0.697],
            "e2": [1.707, 1.916, 1.546, 1.655, 1.303, 1.380],
            "t0": [-0.146, -0.152, -0.146, -0.146, -0.226, -0.109],
            "t1": [-0.114, -0.097, -0.130, -0.124, -0.234, -0.164],
            "t2": [0.506, 0.590, 0.432, 0.507, 0.036, 0.368],
            "t11": [0.085, 0.047, 0.144, 0.117, 0.400, 0.204],
            "t12": [0.162, 0.178, 0.117, 0.127, 0.098, 0.093],
            "t22": [0.073, 0.016, 0.075, 0.015, 0.017, 0.038],
            "r0": [0.060, 0.069, 0.039, 0.036, 0.003, -0.015],
            "r1": [-0.236, -0.261, -0.209, -0.234, -0.025, -0.209],
            "r2": [0.067, 0.107, 0.069, 0.107, -0.169, 0.107],
            "r11": [0.016, -0.003, 0.052, 0.044, 0.082, 0.115],
            "r12": [0.087, 0.109, 0.060, 0.075, 0.051, 0.009],
            "u0": [-0.038, -0.054, -0.042, -0.061, 0.057, -0.066],
            "u1": [0.046, 0.045, 0.036, 0.032, 0.103, 0.011],
            "u2": [0.001, 0.002, 0.008, 0.007, 0.187, -0.013],
            "u11": [0.266, 0.325, 0.272, 0.329, -0.045, 0.312],
            "u12": [-0.176, -0.206, -0.172, -0.202, -0.141, -0.177],
            "u22": [-0.150, -0.163, -0.150, -0.164, 0.087, -0.132],
        },
    }
    e1 = dataModel[model]["e1"][choice]
    e2 = dataModel[model]["e2"][choice]
    t0 = dataModel[model]["t0"][choice]
    t1 = dataModel[model]["t1"][choice]
    t2 = dataModel[model]["t2"][choice]
    t11 = dataModel[model]["t11"][choice]
    t12 = dataModel[model]["t12"][choice]
    t22 = dataModel[model]["t22"][choice]
    r0 = dataModel[model]["r0"][choice]
    r1 = dataModel[model]["r1"][choice]
    r2 = dataModel[model]["r2"][choice]
    r11 = dataModel[model]["r11"][choice]
    r12 = dataModel[model]["r12"][choice]
    u0 = dataModel[model]["u0"][choice]
    u1 = dataModel[model]["u1"][choice]
    u2 = dataModel[model]["u2"][choice]
    u11 = dataModel[model]["u11"][choice]
    u12 = dataModel[model]["u12"][choice]
    u22 = dataModel[model]["u22"][choice]
    a = dataModel[model]["alattice"][choice]

    data = {
        "material": matt[choice],
        "alattice": a,
        "e1": e1,
        "e2": e2,
        "t0": t0,
        "t1": t1,
        "t2": t2,
        "t11": t11,
        "t12": t12,
        "t22": t22,
        "r0": r0,
        "r1": r1,
        "r2": r2,
        "r11": r11,
        "r12": r12,
        "u0": u0,
        "u1": u1,
        "u2": u2,
        "u11": u11,
        "u12": u12,
        "u22": u22,
    }

    return data


W = cp.array(
    [
        [1, 0, 0],
        [0, 1 / sqrt(2), 1j / sqrt(2)],
        [0, 1 / sqrt(2), -1j / sqrt(2)],
    ]
)


def IRTNN_tran(data):
    u0 = data["u0"]
    u1 = data["u1"]
    u2 = data["u2"]
    u12 = data["u12"]
    u11 = data["u11"]
    u22 = data["u22"]

    D_C3 = cp.array(
        [
            [1, 0, 0],
            [0, cos(-2 * pi / 3), -sin(-2 * pi / 3)],
            [0, sin(-2 * pi / 3), cos(-2 * pi / 3)],
        ]
    )

    D_2C3 = cp.array(
        [
            [1, 0, 0],
            [0, cos(-4 * pi / 3), -sin(-4 * pi / 3)],
            [0, sin(-4 * pi / 3), cos(-4 * pi / 3)],
        ]
    )

    D_S = cp.array(
        [
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, 1],
        ]
    )

    D_S1 = cp.array(
        [
            [1, 0, 0],
            [0, 1 / 2, -sqrt(3) / 2],
            [0, -sqrt(3) / 2, -1 / 2],
        ]
    )

    D_S2 = cp.array(
        [
            [1, 0, 0],
            [0, 1 / 2, sqrt(3) / 2],
            [0, sqrt(3) / 2, -1 / 2],
        ]
    )

    E_R7 = cp.array(
        [
            [u0, u1, u2],
            [-u1, u11, u12],
            [u2, -u12, u22],
        ]
    )

    E_R8 = D_S1 @ E_R7 @ D_S1.T
    E_R9 = D_C3 @ E_R7 @ D_C3.T
    E_R10 = D_S @ E_R7 @ D_S.T
    E_R11 = D_2C3 @ E_R7 @ D_2C3.T
    E_R12 = D_S2 @ E_R7 @ D_S2.T

    E_R7 = W @ E_R7 @ cp.conjugate(W).T
    E_R8 = W @ E_R8 @ cp.conjugate(W).T
    E_R9 = W @ E_R9 @ cp.conjugate(W).T
    E_R10 = W @ E_R10 @ cp.conjugate(W).T
    E_R11 = W @ E_R11 @ cp.conjugate(W).T
    E_R12 = W @ E_R12 @ cp.conjugate(W).T

    return E_R7, E_R8, E_R9, E_R10, E_R11, E_R12


def IRNN_tran(data):
    r0 = data["r0"]
    r1 = data["r1"]
    r2 = data["r2"]
    r12 = data["r12"]
    r11 = data["r11"]

    D4 = cp.array(
        [
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, 1],
        ]
    )

    D5 = cp.array(
        [
            [1, 0, 0],
            [0, -1 / 2, -sqrt(3) / 2],
            [0, sqrt(3) / 2, -1 / 2],
        ]
    )

    v1 = cp.array(
        [
            [r0, r1, -r1 / sqrt(3)],
            [r2, r11, r12],
            [-r2 / sqrt(3), r12, r11 + 2 * sqrt(3) / 3 * r12],
        ]
    )

    v4 = cp.array(
        [
            [r0, r2, -r2 / sqrt(3)],
            [r1, r11, r12],
            [-r1 / sqrt(3), r12, r11 + 2 * sqrt(3) / 3 * r12],
        ]
    )

    v2 = D5 @ v4 @ D5.T
    v3 = D4 @ v1 @ D4.T
    v5 = D5 @ v1 @ D5.T
    v6 = D4 @ v4 @ D4.T

    v1 = W @ v1 @ cp.conjugate(W).T
    v2 = W @ v2 @ cp.conjugate(W).T
    v3 = W @ v3 @ cp.conjugate(W).T
    v4 = W @ v4 @ cp.conjugate(W).T
    v5 = W @ v5 @ cp.conjugate(W).T
    v6 = W @ v6 @ cp.conjugate(W).T

    return v1, v2, v3, v4, v5, v6


def IR_tran(data):
    e1 = data["e1"]
    e2 = data["e2"]
    t0 = data["t0"]
    t1 = data["t1"]
    t2 = data["t2"]
    t12 = data["t12"]
    t11 = data["t11"]
    t22 = data["t22"]

    D_C3 = cp.array(
        [
            [1, 0, 0],
            [0, cos(-2 * pi / 3), -sin(-2 * pi / 3)],
            [0, sin(-2 * pi / 3), cos(-2 * pi / 3)],
        ]
    )

    D_2C3 = cp.array(
        [
            [1, 0, 0],
            [0, cos(-4 * pi / 3), -sin(-4 * pi / 3)],
            [0, sin(-4 * pi / 3), cos(-4 * pi / 3)],
        ]
    )

    D_S = cp.array(
        [
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, 1],
        ]
    )

    D_S1 = cp.array(
        [
            [1, 0, 0],
            [0, 1 / 2, -sqrt(3) / 2],
            [0, -sqrt(3) / 2, -1 / 2],
        ]
    )

    D_S2 = cp.array(
        [
            [1, 0, 0],
            [0, 1 / 2, sqrt(3) / 2],
            [0, sqrt(3) / 2, -1 / 2],
        ]
    )

    E_R0 = cp.array(
        [
            [e1, 0, 0],
            [0, e2, 0],
            [0, 0, e2],
        ]
    )

    E_R1 = cp.array(
        [
            [t0, t1, t2],
            [-t1, t11, t12],
            [t2, -t12, t22],
        ]
    )

    E_R2 = D_S1 @ E_R1 @ D_S1.T
    E_R3 = D_C3 @ E_R1 @ D_C3.T
    E_R4 = D_S @ E_R1 @ D_S.T
    E_R5 = D_2C3 @ E_R1 @ D_2C3.T
    E_R6 = D_S2 @ E_R1 @ D_S2.T

    E_R0 = W @ E_R0 @ cp.conjugate(W).T
    E_R1 = W @ E_R1 @ cp.conjugate(W).T
    E_R2 = W @ E_R2 @ cp.conjugate(W).T
    E_R3 = W @ E_R3 @ cp.conjugate(W).T
    E_R4 = W @ E_R4 @ cp.conjugate(W).T
    E_R5 = W @ E_R5 @ cp.conjugate(W).T
    E_R6 = W @ E_R6 @ cp.conjugate(W).T

    return (E_R0, E_R1, E_R2, E_R3, E_R4, E_R5, E_R6)


def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b)


def pbc(i, q):
    return i % (q)


def HamTNN_ky(band: int, alattice: float, p: int, q: int, kx: float, ky: float, IM: dict) -> NDArray:
    """[Summary]

    Define a Hamiltonian for the TMD 3 band using Ref Phys.rev.B 88,085433

    Args:
        band: is the number of the band considering. Default is 3, you can choose 1.
        alattice: is the lattice constant.
        p: numerator.
        q: denomitor in the magnetic
        kx: k valley
        ky: k valley
        IM: hopping terms using group theory and Irreducible matrices.
    """

    eta = 1 * p / q
    alpha = 1 / 2 * kx * alattice
    beta = sqrt(3) / 2 * ky * alattice

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

    E0 = IM["NN"][0] * 0
    h1 = IM["NN"][1] * hR1 * (0)
    h2 = IM["NN"][2] * hR2 * (-sqrt(3) * 1j / 2)
    h3 = IM["NN"][3] * hR3 * (-sqrt(3) * 1j / 2)
    h4 = IM["NN"][4] * hR4 * (0)
    h5 = IM["NN"][5] * hR5 * (sqrt(3) * 1j / 2)
    h6 = IM["NN"][6] * hR6 * (sqrt(3) * 1j / 2)
    o1 = IM["TNN"][0] * oR1 * (0)
    o2 = IM["TNN"][1] * oR2 * (-sqrt(3) * 1j)
    o3 = IM["TNN"][2] * oR3 * (-sqrt(3) * 1j)
    o4 = IM["TNN"][3] * oR4 * (0)
    o5 = IM["TNN"][4] * oR5 * (sqrt(3) * 1j)
    o6 = IM["TNN"][5] * oR6 * (sqrt(3) * 1j)
    v1 = IM["NNN"][0] * vR1 * (-sqrt(3) * 1j / 2)
    v2 = IM["NNN"][1] * vR2 * (-sqrt(3) * 1j)
    v3 = IM["NNN"][2] * vR3 * (-sqrt(3) * 1j / 2)
    v4 = IM["NNN"][3] * vR4 * (sqrt(3) * 1j / 2)
    v5 = IM["NNN"][4] * vR5 * (sqrt(3) * 1j)
    v6 = IM["NNN"][5] * vR6 * (sqrt(3) * 1j / 2)

    H0 = cp.zeros([q, q], dtype=complex)
    H1 = cp.zeros([q, q], dtype=complex)
    H1T = cp.zeros([q, q], dtype=complex)
    H2 = cp.zeros([q, q], dtype=complex)
    H2T = cp.zeros([q, q], dtype=complex)
    H11 = cp.zeros([q, q], dtype=complex)
    H22 = cp.zeros([q, q], dtype=complex)
    H12 = cp.zeros([q, q], dtype=complex)
    H12T = cp.zeros([q, q], dtype=complex)
    H = cp.zeros([3 * q, 3 * q], dtype=complex)
    H2band = cp.zeros([2 * q, 2 * q], dtype=complex)

    for m in range(0, q):

        H0[m, m] = E0[0, 0] + v2[0, 0] * exp(-4j * pi * m * eta) + v5[0, 0] * exp(4j * pi * m * eta)
        H0[m, pbc(m + 1, q)] = h2[0, 0] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) + h6[0, 0] * exp(1j * 2 * pi * (m + 1 / 2) * eta)
        H0[m, pbc(m + 2, q)] = h1[0, 0] + o2[0, 0] * exp(-4j * pi * (m + 1) * eta) + o6[0, 0] * exp(4j * pi * (m + 1) * eta)
        H0[m, pbc(m + 3, q)] = v1[0, 0] * exp(-1j * 2 * pi * (m + 3 / 2) * eta) + v6[0, 0] * exp(1j * 2 * pi * (m + 3 / 2) * eta)
        H0[m, pbc(m + 4, q)] = o1[0, 0]
        H0[m, pbc(m - 1, q)] = h5[0, 0] * exp(1j * 2 * pi * (m - 1 / 2) * eta) + h3[0, 0] * exp(-1j * 2 * pi * (m - 1 / 2) * eta)
        H0[m, pbc(m - 2, q)] = h4[0, 0] + o3[0, 0] * exp(-4j * pi * (m - 1) * eta) + o5[0, 0] * exp(4j * pi * (m - 1) * eta)
        H0[m, pbc(m - 3, q)] = v3[0, 0] * exp(-1j * 2 * pi * (m - 3 / 2) * eta) + v4[0, 0] * exp(1j * 2 * pi * (m - 3 / 2) * eta)
        H0[m, pbc(m - 4, q)] = o4[0, 0]

        H11[m, m] = E0[1, 1] + v2[1, 1] * exp(-4j * pi * m * eta) + v5[1, 1] * exp(4j * pi * m * eta)
        H11[m, pbc(m + 1, q)] = h2[1, 1] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) + h6[1, 1] * exp(1j * 2 * pi * (m + 1 / 2) * eta)
        H11[m, pbc(m + 2, q)] = h1[1, 1] + o2[1, 1] * exp(-4j * pi * (m + 1) * eta) + o6[1, 1] * exp(4j * pi * (m + 1) * eta)
        H11[m, pbc(m + 3, q)] = v1[1, 1] * exp(-1j * 2 * pi * (m + 3 / 2) * eta) + v6[1, 1] * exp(1j * 2 * pi * (m + 3 / 2) * eta)
        H11[m, pbc(m + 4, q)] = o1[1, 1]
        H11[m, pbc(m - 1, q)] = h5[1, 1] * exp(1j * 2 * pi * (m - 1 / 2) * eta) + h3[1, 1] * exp(-1j * 2 * pi * (m - 1 / 2) * eta)
        H11[m, pbc(m - 2, q)] = h4[1, 1] + o3[1, 1] * exp(-4j * pi * (m - 1) * eta) + o5[1, 1] * exp(4j * pi * (m - 1) * eta)
        H11[m, pbc(m - 3, q)] = v3[1, 1] * exp(-1j * 2 * pi * (m - 3 / 2) * eta) + v4[1, 1] * exp(1j * 2 * pi * (m - 3 / 2) * eta)
        H11[m, pbc(m - 4, q)] = o4[1, 1]

        H12T[m, m] = E0[2, 1] + v2[2, 1] * exp(4j * pi * m * eta) + v5[2, 1] * exp(-4j * pi * m * eta)
        H12T[m, pbc(m + 1, q)] = h2[2, 1] * exp(1j * 2 * pi * (m + 1 / 2) * eta) + h6[2, 1] * exp(-1j * 2 * pi * (m + 1 / 2) * eta)
        H12T[m, pbc(m + 2, q)] = h1[2, 1] + o2[2, 1] * exp(4j * pi * (m + 1) * eta) + o6[2, 1] * exp(-4j * pi * (m + 1) * eta)
        H12T[m, pbc(m + 3, q)] = v1[2, 1] * exp(1j * 2 * pi * (m + 3 / 2) * eta) + v6[2, 1] * exp(-1j * 2 * pi * (m + 3 / 2) * eta)
        H12T[m, pbc(m + 4, q)] = o1[2, 1]
        H12T[m, pbc(m - 1, q)] = h5[2, 1] * exp(-1j * 2 * pi * (m - 1 / 2) * eta) + h3[2, 1] * exp(1j * 2 * pi * (m - 1 / 2) * eta)
        H12T[m, pbc(m - 2, q)] = h4[2, 1] + o3[2, 1] * exp(4j * pi * (m - 1) * eta) + o5[2, 1] * exp(-4j * pi * (m - 1) * eta)
        H12T[m, pbc(m - 3, q)] = v3[2, 1] * exp(1j * 2 * pi * (m - 3 / 2) * eta) + v4[2, 1] * exp(-1j * 2 * pi * (m - 3 / 2) * eta)
        H12T[m, pbc(m - 4, q)] = o4[2, 1]

        H12[m, m] = E0[1, 2] + v2[1, 2] * exp(-4j * pi * m * eta) + v5[1, 2] * exp(4j * pi * m * eta)
        H12[m, pbc(m + 1, q)] = h2[1, 2] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) + h6[1, 2] * exp(1j * 2 * pi * (m + 1 / 2) * eta)
        H12[m, pbc(m + 2, q)] = h1[1, 2] + o2[1, 2] * exp(-4j * pi * (m + 1) * eta) + o6[1, 2] * exp(4j * pi * (m + 1) * eta)
        H12[m, pbc(m + 3, q)] = v1[1, 2] * exp(-1j * 2 * pi * (m + 3 / 2) * eta) + v6[1, 2] * exp(1j * 2 * pi * (m + 3 / 2) * eta)
        H12[m, pbc(m + 4, q)] = o1[1, 2]
        H12[m, pbc(m - 1, q)] = h5[1, 2] * exp(1j * 2 * pi * (m - 1 / 2) * eta) + h3[1, 2] * exp(-1j * 2 * pi * (m - 1 / 2) * eta)
        H12[m, pbc(m - 2, q)] = h4[1, 2] + o3[1, 2] * exp(-4j * pi * (m - 1) * eta) + o5[1, 2] * exp(4j * pi * (m - 1) * eta)
        H12[m, pbc(m - 3, q)] = v3[1, 2] * exp(-1j * 2 * pi * (m - 3 / 2) * eta) + v4[1, 2] * exp(1j * 2 * pi * (m - 3 / 2) * eta)
        H12[m, pbc(m - 4, q)] = o4[1, 2]

        H1T[m, m] = E0[1, 0] + v2[1, 0] * exp(4j * pi * m * eta) + v5[1, 0] * exp(-4j * pi * m * eta)
        H1T[m, pbc(m + 1, q)] = h2[1, 0] * exp(1j * 2 * pi * (m + 1 / 2) * eta) + h6[1, 0] * exp(-1j * 2 * pi * (m + 1 / 2) * eta)
        H1T[m, pbc(m + 2, q)] = h1[1, 0] + o2[1, 0] * exp(4j * pi * (m + 1) * eta) + o6[1, 0] * exp(-4j * pi * (m + 1) * eta)
        H1T[m, pbc(m + 3, q)] = v1[1, 0] * exp(1j * 2 * pi * (m + 3 / 2) * eta) + v6[1, 0] * exp(-1j * 2 * pi * (m + 3 / 2) * eta)
        H1T[m, pbc(m + 4, q)] = o1[1, 0]
        H1T[m, pbc(m - 1, q)] = h5[1, 0] * exp(-1j * 2 * pi * (m - 1 / 2) * eta) + h3[1, 0] * exp(1j * 2 * pi * (m - 1 / 2) * eta)
        H1T[m, pbc(m - 2, q)] = h4[1, 0] + o3[1, 0] * exp(4j * pi * (m - 1) * eta) + o5[1, 0] * exp(-4j * pi * (m - 1) * eta)
        H1T[m, pbc(m - 3, q)] = v3[1, 0] * exp(1j * 2 * pi * (m - 3 / 2) * eta) + v4[1, 0] * exp(-1j * 2 * pi * (m - 3 / 2) * eta)
        H1T[m, pbc(m - 4, q)] = o4[1, 0]

        H1[m, m] = E0[0, 1] + v2[0, 1] * exp(-4j * pi * m * eta) + v5[0, 1] * exp(4j * pi * m * eta)
        H1[m, pbc(m + 1, q)] = h2[0, 1] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) + h6[0, 1] * exp(1j * 2 * pi * (m + 1 / 2) * eta)
        H1[m, pbc(m + 2, q)] = h1[0, 1] + o2[0, 1] * exp(-4j * pi * (m + 1) * eta) + o6[0, 1] * exp(4j * pi * (m + 1) * eta)
        H1[m, pbc(m + 3, q)] = v1[0, 1] * exp(-1j * 2 * pi * (m + 3 / 2) * eta) + v6[0, 1] * exp(1j * 2 * pi * (m + 3 / 2) * eta)
        H1[m, pbc(m + 4, q)] = o1[0, 1]
        H1[m, pbc(m - 1, q)] = h5[0, 1] * exp(1j * 2 * pi * (m - 1 / 2) * eta) + h3[0, 1] * exp(-1j * 2 * pi * (m - 1 / 2) * eta)
        H1[m, pbc(m - 2, q)] = h4[0, 1] + o3[0, 1] * exp(-4j * pi * (m - 1) * eta) + o5[0, 1] * exp(4j * pi * (m - 1) * eta)
        H1[m, pbc(m - 3, q)] = v3[0, 1] * exp(-1j * 2 * pi * (m - 3 / 2) * eta) + v4[0, 1] * exp(1j * 2 * pi * (m - 3 / 2) * eta)
        H1[m, pbc(m - 4, q)] = o4[0, 1]

        H22[m, m] = E0[2, 2] + v2[2, 2] * exp(-4j * pi * m * eta) + v5[2, 2] * exp(4j * pi * m * eta)
        H22[m, pbc(m + 1, q)] = h2[2, 2] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) + h6[2, 2] * exp(1j * 2 * pi * (m + 1 / 2) * eta)
        H22[m, pbc(m + 2, q)] = h1[2, 2] + o2[2, 2] * exp(-4j * pi * (m + 1) * eta) + o6[2, 2] * exp(4j * pi * (m + 1) * eta)
        H22[m, pbc(m + 3, q)] = v1[2, 2] * exp(-1j * 2 * pi * (m + 3 / 2) * eta) + v6[2, 2] * exp(1j * 2 * pi * (m + 3 / 2) * eta)
        H22[m, pbc(m + 4, q)] = o1[2, 2]
        H22[m, pbc(m - 1, q)] = h5[2, 2] * exp(1j * 2 * pi * (m - 1 / 2) * eta) + h3[2, 2] * exp(-1j * 2 * pi * (m - 1 / 2) * eta)
        H22[m, pbc(m - 2, q)] = h4[2, 2] + o3[2, 2] * exp(-4j * pi * (m - 1) * eta) + o5[2, 2] * exp(4j * pi * (m - 1) * eta)
        H22[m, pbc(m - 3, q)] = v3[2, 2] * exp(-1j * 2 * pi * (m - 3 / 2) * eta) + v4[2, 2] * exp(1j * 2 * pi * (m - 3 / 2) * eta)
        H22[m, pbc(m - 4, q)] = o4[2, 2]

        H2T[m, m] = E0[2, 0] + v2[2, 0] * exp(4j * pi * m * eta) + v5[2, 0] * exp(-4j * pi * m * eta)
        H2T[m, pbc(m + 1, q)] = h2[2, 0] * exp(1j * 2 * pi * (m + 1 / 2) * eta) + h6[2, 0] * exp(-1j * 2 * pi * (m + 1 / 2) * eta)
        H2T[m, pbc(m + 2, q)] = h1[2, 0] + o2[2, 0] * exp(4j * pi * (m + 1) * eta) + o6[2, 0] * exp(-4j * pi * (m + 1) * eta)
        H2T[m, pbc(m + 3, q)] = v1[2, 0] * exp(1j * 2 * pi * (m + 3 / 2) * eta) + v6[2, 0] * exp(-1j * 2 * pi * (m + 3 / 2) * eta)
        H2T[m, pbc(m + 4, q)] = o1[2, 0]
        H2T[m, pbc(m - 1, q)] = h5[2, 0] * exp(-1j * 2 * pi * (m - 1 / 2) * eta) + h3[2, 0] * exp(1j * 2 * pi * (m - 1 / 2) * eta)
        H2T[m, pbc(m - 2, q)] = h4[2, 0] + o3[2, 0] * exp(4j * pi * (m - 1) * eta) + o5[2, 0] * exp(-4j * pi * (m - 1) * eta)
        H2T[m, pbc(m - 3, q)] = v3[2, 0] * exp(1j * 2 * pi * (m - 3 / 2) * eta) + v4[2, 0] * exp(-1j * 2 * pi * (m - 3 / 2) * eta)
        H2T[m, pbc(m - 4, q)] = o4[2, 0]

        H2[m, m] = E0[0, 2] + v2[0, 2] * exp(-4j * pi * m * eta) + v5[0, 2] * exp(4j * pi * m * eta)
        H2[m, pbc(m + 1, q)] = h2[0, 2] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) + h6[0, 2] * exp(1j * 2 * pi * (m + 1 / 2) * eta)
        H2[m, pbc(m + 2, q)] = h1[0, 2] + o2[0, 2] * exp(-4j * pi * (m + 1) * eta) + o6[0, 2] * exp(4j * pi * (m + 1) * eta)
        H2[m, pbc(m + 3, q)] = v1[0, 2] * exp(-1j * 2 * pi * (m + 3 / 2) * eta) + v6[0, 2] * exp(1j * 2 * pi * (m + 3 / 2) * eta)
        H2[m, pbc(m + 4, q)] = o1[0, 2]
        H2[m, pbc(m - 1, q)] = h5[0, 2] * exp(1j * 2 * pi * (m - 1 / 2) * eta) + h3[0, 2] * exp(-1j * 2 * pi * (m - 1 / 2) * eta)
        H2[m, pbc(m - 2, q)] = h4[0, 2] + o3[0, 2] * exp(-4j * pi * (m - 1) * eta) + o5[0, 2] * exp(4j * pi * (m - 1) * eta)
        H2[m, pbc(m - 3, q)] = v3[0, 2] * exp(-1j * 2 * pi * (m - 3 / 2) * eta) + v4[0, 2] * exp(1j * 2 * pi * (m - 3 / 2) * eta)
        H2[m, pbc(m - 4, q)] = o4[0, 2]

    H[0:q, 0:q] = H0
    H[0:q, q : 2 * q] = H1
    H[0:q, 2 * q : 3 * q] = H2
    H[q : 2 * q, 0:q] = H1T
    H[q : 2 * q, q : 2 * q] = H11
    H[q : 2 * q, 2 * q : 3 * q] = H12
    H[2 * q : 3 * q, 0:q] = H2T
    H[2 * q : 3 * q, q : 2 * q] = H12T
    H[2 * q : 3 * q, 2 * q : 3 * q] = H22

    return H


def HamTNN_kx(band: int, alattice: float, p: int, q: int, kx: float, ky: float, IM: dict):
    """[Summary]

    Define a Hamiltonian for the TMD 3 band using Ref Phys.rev.B 88,085433

    Args:
        band: is the number of the band considering. Default is 3, you can choose 1.
        alattice: is the lattice constant.
        p: numerator.
        q: denomitor in the magnetic
        kx: k valley
        ky: k valley
        IM: hopping terms using group theory and Irreducible matrices.
    """

    eta = 1 * p / q
    alpha = 1 / 2 * kx * alattice
    beta = sqrt(3) / 2 * ky * alattice

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

    E0 = IM["NN"][0] * 0
    h1 = IM["NN"][1] * hR1 * (1j)
    h2 = IM["NN"][2] * hR2 * (1j / 2)
    h3 = IM["NN"][3] * hR3 * (-1j / 2)
    h4 = IM["NN"][4] * hR4 * (-1j)
    h5 = IM["NN"][5] * hR5 * (-1j / 2)
    h6 = IM["NN"][6] * hR6 * (1j / 2)
    o1 = IM["TNN"][0] * oR1 * (2j)
    o2 = IM["TNN"][1] * oR2 * (1j)
    o3 = IM["TNN"][2] * oR3 * (-1j)
    o4 = IM["TNN"][3] * oR4 * (-2j)
    o5 = IM["TNN"][4] * oR5 * (-1j)
    o6 = IM["TNN"][5] * oR6 * (1j)
    v1 = IM["NNN"][0] * vR1 * (3 / 2j)
    v2 = IM["NNN"][1] * vR2 * 0
    v3 = IM["NNN"][2] * vR3 * (-3 / 2j)
    v4 = IM["NNN"][3] * vR4 * (-3 / 2j)
    v5 = IM["NNN"][4] * vR5 * 0
    v6 = IM["NNN"][5] * vR6 * (3 / 2j)

    H0 = cp.zeros([q, q], dtype=complex)
    H1 = cp.zeros([q, q], dtype=complex)
    H1T = cp.zeros([q, q], dtype=complex)
    H2 = cp.zeros([q, q], dtype=complex)
    H2T = cp.zeros([q, q], dtype=complex)
    H11 = cp.zeros([q, q], dtype=complex)
    H22 = cp.zeros([q, q], dtype=complex)
    H12 = cp.zeros([q, q], dtype=complex)
    H12T = cp.zeros([q, q], dtype=complex)
    H = cp.zeros([3 * q, 3 * q], dtype=complex)
    H2band = cp.zeros([2 * q, 2 * q], dtype=complex)

    for m in range(0, q):

        H0[m, m] = E0[0, 0] + v2[0, 0] * exp(-4j * pi * m * eta) + v5[0, 0] * exp(4j * pi * m * eta)
        H0[m, pbc(m + 1, q)] = h2[0, 0] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) + h6[0, 0] * exp(1j * 2 * pi * (m + 1 / 2) * eta)
        H0[m, pbc(m + 2, q)] = h1[0, 0] + o2[0, 0] * exp(-4j * pi * (m + 1) * eta) + o6[0, 0] * exp(4j * pi * (m + 1) * eta)
        H0[m, pbc(m + 3, q)] = v1[0, 0] * exp(-1j * 2 * pi * (m + 3 / 2) * eta) + v6[0, 0] * exp(1j * 2 * pi * (m + 3 / 2) * eta)
        H0[m, pbc(m + 4, q)] = o1[0, 0]
        H0[m, pbc(m - 1, q)] = h5[0, 0] * exp(1j * 2 * pi * (m - 1 / 2) * eta) + h3[0, 0] * exp(-1j * 2 * pi * (m - 1 / 2) * eta)
        H0[m, pbc(m - 2, q)] = h4[0, 0] + o3[0, 0] * exp(-4j * pi * (m - 1) * eta) + o5[0, 0] * exp(4j * pi * (m - 1) * eta)
        H0[m, pbc(m - 3, q)] = v3[0, 0] * exp(-1j * 2 * pi * (m - 3 / 2) * eta) + v4[0, 0] * exp(1j * 2 * pi * (m - 3 / 2) * eta)
        H0[m, pbc(m - 4, q)] = o4[0, 0]

        H11[m, m] = E0[1, 1] + v2[1, 1] * exp(-4j * pi * m * eta) + v5[1, 1] * exp(4j * pi * m * eta)
        H11[m, pbc(m + 1, q)] = h2[1, 1] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) + h6[1, 1] * exp(1j * 2 * pi * (m + 1 / 2) * eta)
        H11[m, pbc(m + 2, q)] = h1[1, 1] + o2[1, 1] * exp(-4j * pi * (m + 1) * eta) + o6[1, 1] * exp(4j * pi * (m + 1) * eta)
        H11[m, pbc(m + 3, q)] = v1[1, 1] * exp(-1j * 2 * pi * (m + 3 / 2) * eta) + v6[1, 1] * exp(1j * 2 * pi * (m + 3 / 2) * eta)
        H11[m, pbc(m + 4, q)] = o1[1, 1]
        H11[m, pbc(m - 1, q)] = h5[1, 1] * exp(1j * 2 * pi * (m - 1 / 2) * eta) + h3[1, 1] * exp(-1j * 2 * pi * (m - 1 / 2) * eta)
        H11[m, pbc(m - 2, q)] = h4[1, 1] + o3[1, 1] * exp(-4j * pi * (m - 1) * eta) + o5[1, 1] * exp(4j * pi * (m - 1) * eta)
        H11[m, pbc(m - 3, q)] = v3[1, 1] * exp(-1j * 2 * pi * (m - 3 / 2) * eta) + v4[1, 1] * exp(1j * 2 * pi * (m - 3 / 2) * eta)
        H11[m, pbc(m - 4, q)] = o4[1, 1]

        H12T[m, m] = E0[2, 1] + v2[2, 1] * exp(4j * pi * m * eta) + v5[2, 1] * exp(-4j * pi * m * eta)
        H12T[m, pbc(m + 1, q)] = h2[2, 1] * exp(1j * 2 * pi * (m + 1 / 2) * eta) + h6[2, 1] * exp(-1j * 2 * pi * (m + 1 / 2) * eta)
        H12T[m, pbc(m + 2, q)] = h1[2, 1] + o2[2, 1] * exp(4j * pi * (m + 1) * eta) + o6[2, 1] * exp(-4j * pi * (m + 1) * eta)
        H12T[m, pbc(m + 3, q)] = v1[2, 1] * exp(1j * 2 * pi * (m + 3 / 2) * eta) + v6[2, 1] * exp(-1j * 2 * pi * (m + 3 / 2) * eta)
        H12T[m, pbc(m + 4, q)] = o1[2, 1]
        H12T[m, pbc(m - 1, q)] = h5[2, 1] * exp(-1j * 2 * pi * (m - 1 / 2) * eta) + h3[2, 1] * exp(1j * 2 * pi * (m - 1 / 2) * eta)
        H12T[m, pbc(m - 2, q)] = h4[2, 1] + o3[2, 1] * exp(4j * pi * (m - 1) * eta) + o5[2, 1] * exp(-4j * pi * (m - 1) * eta)
        H12T[m, pbc(m - 3, q)] = v3[2, 1] * exp(1j * 2 * pi * (m - 3 / 2) * eta) + v4[2, 1] * exp(-1j * 2 * pi * (m - 3 / 2) * eta)
        H12T[m, pbc(m - 4, q)] = o4[2, 1]

        H12[m, m] = E0[1, 2] + v2[1, 2] * exp(-4j * pi * m * eta) + v5[1, 2] * exp(4j * pi * m * eta)
        H12[m, pbc(m + 1, q)] = h2[1, 2] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) + h6[1, 2] * exp(1j * 2 * pi * (m + 1 / 2) * eta)
        H12[m, pbc(m + 2, q)] = h1[1, 2] + o2[1, 2] * exp(-4j * pi * (m + 1) * eta) + o6[1, 2] * exp(4j * pi * (m + 1) * eta)
        H12[m, pbc(m + 3, q)] = v1[1, 2] * exp(-1j * 2 * pi * (m + 3 / 2) * eta) + v6[1, 2] * exp(1j * 2 * pi * (m + 3 / 2) * eta)
        H12[m, pbc(m + 4, q)] = o1[1, 2]
        H12[m, pbc(m - 1, q)] = h5[1, 2] * exp(1j * 2 * pi * (m - 1 / 2) * eta) + h3[1, 2] * exp(-1j * 2 * pi * (m - 1 / 2) * eta)
        H12[m, pbc(m - 2, q)] = h4[1, 2] + o3[1, 2] * exp(-4j * pi * (m - 1) * eta) + o5[1, 2] * exp(4j * pi * (m - 1) * eta)
        H12[m, pbc(m - 3, q)] = v3[1, 2] * exp(-1j * 2 * pi * (m - 3 / 2) * eta) + v4[1, 2] * exp(1j * 2 * pi * (m - 3 / 2) * eta)
        H12[m, pbc(m - 4, q)] = o4[1, 2]

        H1T[m, m] = E0[1, 0] + v2[1, 0] * exp(4j * pi * m * eta) + v5[1, 0] * exp(-4j * pi * m * eta)
        H1T[m, pbc(m + 1, q)] = h2[1, 0] * exp(1j * 2 * pi * (m + 1 / 2) * eta) + h6[1, 0] * exp(-1j * 2 * pi * (m + 1 / 2) * eta)
        H1T[m, pbc(m + 2, q)] = h1[1, 0] + o2[1, 0] * exp(4j * pi * (m + 1) * eta) + o6[1, 0] * exp(-4j * pi * (m + 1) * eta)
        H1T[m, pbc(m + 3, q)] = v1[1, 0] * exp(1j * 2 * pi * (m + 3 / 2) * eta) + v6[1, 0] * exp(-1j * 2 * pi * (m + 3 / 2) * eta)
        H1T[m, pbc(m + 4, q)] = o1[1, 0]
        H1T[m, pbc(m - 1, q)] = h5[1, 0] * exp(-1j * 2 * pi * (m - 1 / 2) * eta) + h3[1, 0] * exp(1j * 2 * pi * (m - 1 / 2) * eta)
        H1T[m, pbc(m - 2, q)] = h4[1, 0] + o3[1, 0] * exp(4j * pi * (m - 1) * eta) + o5[1, 0] * exp(-4j * pi * (m - 1) * eta)
        H1T[m, pbc(m - 3, q)] = v3[1, 0] * exp(1j * 2 * pi * (m - 3 / 2) * eta) + v4[1, 0] * exp(-1j * 2 * pi * (m - 3 / 2) * eta)
        H1T[m, pbc(m - 4, q)] = o4[1, 0]

        H1[m, m] = E0[0, 1] + v2[0, 1] * exp(-4j * pi * m * eta) + v5[0, 1] * exp(4j * pi * m * eta)
        H1[m, pbc(m + 1, q)] = h2[0, 1] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) + h6[0, 1] * exp(1j * 2 * pi * (m + 1 / 2) * eta)
        H1[m, pbc(m + 2, q)] = h1[0, 1] + o2[0, 1] * exp(-4j * pi * (m + 1) * eta) + o6[0, 1] * exp(4j * pi * (m + 1) * eta)
        H1[m, pbc(m + 3, q)] = v1[0, 1] * exp(-1j * 2 * pi * (m + 3 / 2) * eta) + v6[0, 1] * exp(1j * 2 * pi * (m + 3 / 2) * eta)
        H1[m, pbc(m + 4, q)] = o1[0, 1]
        H1[m, pbc(m - 1, q)] = h5[0, 1] * exp(1j * 2 * pi * (m - 1 / 2) * eta) + h3[0, 1] * exp(-1j * 2 * pi * (m - 1 / 2) * eta)
        H1[m, pbc(m - 2, q)] = h4[0, 1] + o3[0, 1] * exp(-4j * pi * (m - 1) * eta) + o5[0, 1] * exp(4j * pi * (m - 1) * eta)
        H1[m, pbc(m - 3, q)] = v3[0, 1] * exp(-1j * 2 * pi * (m - 3 / 2) * eta) + v4[0, 1] * exp(1j * 2 * pi * (m - 3 / 2) * eta)
        H1[m, pbc(m - 4, q)] = o4[0, 1]

        H22[m, m] = E0[2, 2] + v2[2, 2] * exp(-4j * pi * m * eta) + v5[2, 2] * exp(4j * pi * m * eta)
        H22[m, pbc(m + 1, q)] = h2[2, 2] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) + h6[2, 2] * exp(1j * 2 * pi * (m + 1 / 2) * eta)
        H22[m, pbc(m + 2, q)] = h1[2, 2] + o2[2, 2] * exp(-4j * pi * (m + 1) * eta) + o6[2, 2] * exp(4j * pi * (m + 1) * eta)
        H22[m, pbc(m + 3, q)] = v1[2, 2] * exp(-1j * 2 * pi * (m + 3 / 2) * eta) + v6[2, 2] * exp(1j * 2 * pi * (m + 3 / 2) * eta)
        H22[m, pbc(m + 4, q)] = o1[2, 2]
        H22[m, pbc(m - 1, q)] = h5[2, 2] * exp(1j * 2 * pi * (m - 1 / 2) * eta) + h3[2, 2] * exp(-1j * 2 * pi * (m - 1 / 2) * eta)
        H22[m, pbc(m - 2, q)] = h4[2, 2] + o3[2, 2] * exp(-4j * pi * (m - 1) * eta) + o5[2, 2] * exp(4j * pi * (m - 1) * eta)
        H22[m, pbc(m - 3, q)] = v3[2, 2] * exp(-1j * 2 * pi * (m - 3 / 2) * eta) + v4[2, 2] * exp(1j * 2 * pi * (m - 3 / 2) * eta)
        H22[m, pbc(m - 4, q)] = o4[2, 2]

        H2T[m, m] = E0[2, 0] + v2[2, 0] * exp(4j * pi * m * eta) + v5[2, 0] * exp(-4j * pi * m * eta)
        H2T[m, pbc(m + 1, q)] = h2[2, 0] * exp(1j * 2 * pi * (m + 1 / 2) * eta) + h6[2, 0] * exp(-1j * 2 * pi * (m + 1 / 2) * eta)
        H2T[m, pbc(m + 2, q)] = h1[2, 0] + o2[2, 0] * exp(4j * pi * (m + 1) * eta) + o6[2, 0] * exp(-4j * pi * (m + 1) * eta)
        H2T[m, pbc(m + 3, q)] = v1[2, 0] * exp(1j * 2 * pi * (m + 3 / 2) * eta) + v6[2, 0] * exp(-1j * 2 * pi * (m + 3 / 2) * eta)
        H2T[m, pbc(m + 4, q)] = o1[2, 0]
        H2T[m, pbc(m - 1, q)] = h5[2, 0] * exp(-1j * 2 * pi * (m - 1 / 2) * eta) + h3[2, 0] * exp(1j * 2 * pi * (m - 1 / 2) * eta)
        H2T[m, pbc(m - 2, q)] = h4[2, 0] + o3[2, 0] * exp(4j * pi * (m - 1) * eta) + o5[2, 0] * exp(-4j * pi * (m - 1) * eta)
        H2T[m, pbc(m - 3, q)] = v3[2, 0] * exp(1j * 2 * pi * (m - 3 / 2) * eta) + v4[2, 0] * exp(-1j * 2 * pi * (m - 3 / 2) * eta)
        H2T[m, pbc(m - 4, q)] = o4[2, 0]

        H2[m, m] = E0[0, 2] + v2[0, 2] * exp(-4j * pi * m * eta) + v5[0, 2] * exp(4j * pi * m * eta)
        H2[m, pbc(m + 1, q)] = h2[0, 2] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) + h6[0, 2] * exp(1j * 2 * pi * (m + 1 / 2) * eta)
        H2[m, pbc(m + 2, q)] = h1[0, 2] + o2[0, 2] * exp(-4j * pi * (m + 1) * eta) + o6[0, 2] * exp(4j * pi * (m + 1) * eta)
        H2[m, pbc(m + 3, q)] = v1[0, 2] * exp(-1j * 2 * pi * (m + 3 / 2) * eta) + v6[0, 2] * exp(1j * 2 * pi * (m + 3 / 2) * eta)
        H2[m, pbc(m + 4, q)] = o1[0, 2]
        H2[m, pbc(m - 1, q)] = h5[0, 2] * exp(1j * 2 * pi * (m - 1 / 2) * eta) + h3[0, 2] * exp(-1j * 2 * pi * (m - 1 / 2) * eta)
        H2[m, pbc(m - 2, q)] = h4[0, 2] + o3[0, 2] * exp(-4j * pi * (m - 1) * eta) + o5[0, 2] * exp(4j * pi * (m - 1) * eta)
        H2[m, pbc(m - 3, q)] = v3[0, 2] * exp(-1j * 2 * pi * (m - 3 / 2) * eta) + v4[0, 2] * exp(1j * 2 * pi * (m - 3 / 2) * eta)
        H2[m, pbc(m - 4, q)] = o4[0, 2]

    H[0:q, 0:q] = H0
    H[0:q, q : 2 * q] = H1
    H[0:q, 2 * q : 3 * q] = H2
    H[q : 2 * q, 0:q] = H1T
    H[q : 2 * q, q : 2 * q] = H11
    H[q : 2 * q, 2 * q : 3 * q] = H12
    H[2 * q : 3 * q, 0:q] = H2T
    H[2 * q : 3 * q, q : 2 * q] = H12T
    H[2 * q : 3 * q, 2 * q : 3 * q] = H22

    return H


def HamNN(band, alattice, p, q, kx, ky, IM):
    # matt, alattice, e1, e2, t0, t1, t2, t11, t12, t22 = para(argument)
    eta = p / (1 * q)

    alpha = 1 / 2 * kx * alattice
    beta = sqrt(3) / 2 * ky * alattice

    hR1 = exp(1j * 2 * alpha)
    hR2 = exp(1j * (alpha - beta))
    hR3 = exp(1j * (-alpha - beta))
    hR4 = exp(-1j * 2 * alpha)
    hR5 = exp(1j * (-alpha + beta))
    hR6 = exp(1j * (alpha + beta))

    E_R0 = IM["NN"][0]
    E_R1 = IM["NN"][1] * hR1
    E_R2 = IM["NN"][2] * hR2
    E_R3 = IM["NN"][3] * hR3
    E_R4 = IM["NN"][4] * hR4
    E_R5 = IM["NN"][5] * hR5
    E_R6 = IM["NN"][6] * hR6

    h0 = cp.zeros([q, q], dtype=complex)
    h1 = cp.zeros([q, q], dtype=complex)
    h1T = cp.zeros([q, q], dtype=complex)
    h2 = cp.zeros([q, q], dtype=complex)
    h2T = cp.zeros([q, q], dtype=complex)
    h11 = cp.zeros([q, q], dtype=complex)
    h22 = cp.zeros([q, q], dtype=complex)
    h12 = cp.zeros([q, q], dtype=complex)
    h12T = cp.zeros([q, q], dtype=complex)
    H2band = cp.zeros([2 * q, 2 * q], dtype=complex)
    H = cp.zeros([3 * q, 3 * q], dtype=complex)

    for m in range(0, q):
        h0[m][m] = E_R0[0][0]
        h1[m][m] = E_R0[0][1]
        h2[m][m] = E_R0[0][2]
        h1T[m][m] = E_R0[1][0]
        h11[m][m] = E_R0[1][1]
        h12[m][m] = E_R0[1][2]
        h2T[m][m] = E_R0[2][0]
        h12T[m][m] = E_R0[2][1]
        h22[m][m] = E_R0[2][2]

        phaseR1 = exp(2j * alpha)
        phaseR4 = exp(-2j * alpha)
        phaseR2 = exp(1j * (alpha - beta))
        phaseR3 = exp(1j * (-alpha - beta))
        phaseR5 = exp(1j * (-alpha + beta))
        phaseR6 = exp(1j * (alpha + beta))

        h0[m, pbc(m + 1, q)] = E_R2[0][0] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) * phaseR2 + E_R6[0][0] * exp(1j * 2 * pi * (m + 1 / 2) * eta) * phaseR6
        h0[m, pbc(m + 2, q)] = E_R1[0][0] * phaseR1
        h0[m, pbc(m - 1, q)] = E_R5[0][0] * exp(1j * 2 * pi * (m - 1 / 2) * eta) * phaseR5 + E_R3[0][0] * exp(-1j * 2 * pi * (m - 1 / 2) * eta) * phaseR3
        h0[m, pbc(m - 2, q)] = E_R4[0][0] * phaseR4

        h11[m, pbc(m + 1, q)] = E_R2[1][1] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) * phaseR2 + E_R6[1][1] * exp(1j * 2 * pi * (m + 1 / 2) * eta) * phaseR6
        h11[m, pbc(m + 2, q)] = E_R1[1][1] * phaseR1
        h11[m, pbc(m - 1, q)] = E_R5[1][1] * exp(1j * 2 * pi * (m - 1 / 2) * eta) * phaseR5 + E_R3[1][1] * exp(-1j * 2 * pi * (m - 1 / 2) * eta) * phaseR3
        h11[m, pbc(m - 2, q)] = E_R4[1][1] * phaseR4

        h12T[m, pbc(m + 1, q)] = E_R2[2][1] * exp(1j * 2 * pi * (m + 1 / 2) * eta) * cp.conjugate(phaseR2) + E_R6[2][1] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) * cp.conjugate(phaseR6)
        h12T[m, pbc(m + 2, q)] = E_R1[2][1] * cp.conjugate(phaseR1)
        h12T[m, pbc(m - 1, q)] = E_R5[2][1] * exp(-1j * 2 * pi * (m - 1 / 2) * eta) * cp.conjugate(phaseR5) + E_R3[2][1] * exp(1j * 2 * pi * (m - 1 / 2) * eta) * cp.conjugate(phaseR3)
        h12T[m, pbc(m - 2, q)] = E_R4[2][1] * cp.conjugate(phaseR4)

        h12[m, pbc(m + 1, q)] = E_R2[1][2] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) * phaseR2 + E_R6[1][2] * exp(1j * 2 * pi * (m + 1 / 2) * eta) * phaseR6
        h12[m, pbc(m + 2, q)] = E_R1[1][2] * phaseR1
        h12[m, pbc(m - 1, q)] = E_R5[1][2] * exp(1j * 2 * pi * (m - 1 / 2) * eta) * phaseR5 + E_R3[1][2] * exp(-1j * 2 * pi * (m - 1 / 2) * eta) * phaseR3
        h12[m, pbc(m - 2, q)] = E_R4[1][2] * phaseR4

        h1T[m, pbc(m + 1, q)] = E_R2[1][0] * exp(1j * 2 * pi * (m + 1 / 2) * eta) * cp.conjugate(phaseR2) + E_R6[1][0] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) * cp.conjugate(phaseR6)
        h1T[m, pbc(m + 2, q)] = E_R1[1][0] * cp.conjugate(phaseR1)
        h1T[m, pbc(m - 1, q)] = E_R5[1][0] * exp(-1j * 2 * pi * (m - 1 / 2) * eta) * cp.conjugate(phaseR5) + E_R3[1][0] * exp(1j * 2 * pi * (m - 1 / 2) * eta) * cp.conjugate(phaseR3)
        h1T[m, pbc(m - 2, q)] = E_R4[1][0] * cp.conjugate(phaseR4)

        h1[m, pbc(m + 1, q)] = E_R2[0][1] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) * phaseR2 + E_R6[0][1] * exp(1j * 2 * pi * (m + 1 / 2) * eta) * phaseR6
        h1[m, pbc(m + 2, q)] = E_R1[0][1] * phaseR1
        h1[m, pbc(m - 1, q)] = E_R5[0][1] * exp(1j * 2 * pi * (m - 1 / 2) * eta) * phaseR5 + E_R3[0][1] * exp(-1j * 2 * pi * (m - 1 / 2) * eta) * phaseR3
        h1[m, pbc(m - 2, q)] = E_R4[0][1] * phaseR4

        h22[m, pbc(m + 1, q)] = E_R2[2][2] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) * phaseR2 + E_R6[2][2] * exp(1j * 2 * pi * (m + 1 / 2) * eta) * phaseR6
        h22[m, pbc(m + 2, q)] = E_R1[2][2] * phaseR1
        h22[m, pbc(m - 1, q)] = E_R5[2][2] * exp(1j * 2 * pi * (m - 1 / 2) * eta) + E_R3[2][2] * exp(-1j * 2 * pi * (m - 1 / 2) * eta) * phaseR3
        h22[m, pbc(m - 2, q)] = E_R4[2][2] * phaseR4

        h2T[m, pbc(m + 1, q)] = E_R2[2][0] * exp(1j * 2 * pi * (m + 1 / 2) * eta) * cp.conjugate(phaseR2) + E_R6[2][0] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) * cp.conjugate(phaseR6)
        h2T[m, pbc(m + 2, q)] = E_R1[2][0] * cp.conjugate(phaseR1)
        h2T[m, pbc(m - 1, q)] = E_R5[2][0] * exp(-1j * 2 * pi * (m - 1 / 2) * eta) * cp.conjugate(phaseR5) + E_R3[2][0] * exp(1j * 2 * pi * (m - 1 / 2) * eta) * cp.conjugate(phaseR3)
        h2T[m, pbc(m - 2, q)] = E_R4[2][0] * cp.conjugate(phaseR4)

        h2[m, pbc(m + 1, q)] = E_R2[0][2] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) * phaseR2 + E_R6[0][2] * exp(1j * 2 * pi * (m + 1 / 2) * eta) * phaseR6
        h2[m, pbc(m + 2, q)] = E_R1[0][2] * phaseR1
        h2[m, pbc(m - 1, q)] = E_R5[0][2] * exp(1j * 2 * pi * (m - 1 / 2) * eta) * phaseR5 + E_R3[0][2] * exp(-1j * 2 * pi * (m - 1 / 2) * eta) * phaseR3
        h2[m, pbc(m - 2, q)] = E_R4[0][2] * phaseR4

    if band == 1:
        return h0

    elif band == 2:
        H2band[0:q, 0:q] = h1
        H2band[0:q, q : 2 * q] = h2
        H2band[q : 2 * q, 0:q] = h1T
        H2band[q : 2 * q, q : 2 * q] = h2T
        return H2band

    elif band == 3:
        H[0:q, 0:q] = h0
        H[0:q, q : 2 * q] = h1
        H[0:q, 2 * q : 3 * q] = h2
        H[q : 2 * q, 0:q] = h1T
        H[q : 2 * q, q : 2 * q] = h11
        H[q : 2 * q, 2 * q : 3 * q] = h12
        H[2 * q : 3 * q, 0:q] = h2T
        H[2 * q : 3 * q, q : 2 * q] = h12T
        H[2 * q : 3 * q, 2 * q : 3 * q] = h22

        return H


def HamTNN(band: int, alattice: float, p: int, q: int, kx: float, ky: float, IM: dict) -> NDArray:
    """[Summary]

    Define a Hamiltonian for the TMD 3 band using Ref Phys.rev.B 88,085433

    Args:
        band: is the number of the band considering. Default is 3, you can choose 1.
        alattice: is the lattice constant.
        p: numerator.
        q: denomitor in the magnetic
        kx: k valley
        ky: k valley
        IM: hopping terms using group theory and Irreducible matrices.
    """

    eta = 1 * p / q
    alpha = 1 / 2 * kx * alattice
    beta = sqrt(3) / 2 * ky * alattice

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

    E0 = IM["NN"][0]
    h1 = IM["NN"][1] * hR1
    h2 = IM["NN"][2] * hR2
    h3 = IM["NN"][3] * hR3
    h4 = IM["NN"][4] * hR4
    h5 = IM["NN"][5] * hR5
    h6 = IM["NN"][6] * hR6
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

    H0 = cp.zeros([q, q], dtype=complex)
    H1 = cp.zeros([q, q], dtype=complex)
    H1T = cp.zeros([q, q], dtype=complex)
    H2 = cp.zeros([q, q], dtype=complex)
    H2T = cp.zeros([q, q], dtype=complex)
    H11 = cp.zeros([q, q], dtype=complex)
    H22 = cp.zeros([q, q], dtype=complex)
    H12 = cp.zeros([q, q], dtype=complex)
    H12T = cp.zeros([q, q], dtype=complex)
    H = cp.zeros([3 * q, 3 * q], dtype=complex)

    for m in range(0, q):

        H0[m, m] = E0[0, 0] + v2[0, 0] * exp(-4j * pi * m * eta) + v5[0, 0] * exp(4j * pi * m * eta)
        H0[m, pbc(m + 1, q)] = h2[0, 0] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) + h6[0, 0] * exp(1j * 2 * pi * (m + 1 / 2) * eta)
        H0[m, pbc(m + 2, q)] = h1[0, 0] + o2[0, 0] * exp(-4j * pi * (m + 1) * eta) + o6[0, 0] * exp(4j * pi * (m + 1) * eta)
        H0[m, pbc(m + 3, q)] = v1[0, 0] * exp(-1j * 2 * pi * (m + 3 / 2) * eta) + v6[0, 0] * exp(1j * 2 * pi * (m + 3 / 2) * eta)
        H0[m, pbc(m + 4, q)] = o1[0, 0]
        H0[m, pbc(m - 1, q)] = h5[0, 0] * exp(1j * 2 * pi * (m - 1 / 2) * eta) + h3[0, 0] * exp(-1j * 2 * pi * (m - 1 / 2) * eta)
        H0[m, pbc(m - 2, q)] = h4[0, 0] + o3[0, 0] * exp(-4j * pi * (m - 1) * eta) + o5[0, 0] * exp(4j * pi * (m - 1) * eta)
        H0[m, pbc(m - 3, q)] = v3[0, 0] * exp(-1j * 2 * pi * (m - 3 / 2) * eta) + v4[0, 0] * exp(1j * 2 * pi * (m - 3 / 2) * eta)
        H0[m, pbc(m - 4, q)] = o4[0, 0]

        H11[m, m] = E0[1, 1] + v2[1, 1] * exp(-4j * pi * m * eta) + v5[1, 1] * exp(4j * pi * m * eta)
        H11[m, pbc(m + 1, q)] = h2[1, 1] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) + h6[1, 1] * exp(1j * 2 * pi * (m + 1 / 2) * eta)
        H11[m, pbc(m + 2, q)] = h1[1, 1] + o2[1, 1] * exp(-4j * pi * (m + 1) * eta) + o6[1, 1] * exp(4j * pi * (m + 1) * eta)
        H11[m, pbc(m + 3, q)] = v1[1, 1] * exp(-1j * 2 * pi * (m + 3 / 2) * eta) + v6[1, 1] * exp(1j * 2 * pi * (m + 3 / 2) * eta)
        H11[m, pbc(m + 4, q)] = o1[1, 1]
        H11[m, pbc(m - 1, q)] = h5[1, 1] * exp(1j * 2 * pi * (m - 1 / 2) * eta) + h3[1, 1] * exp(-1j * 2 * pi * (m - 1 / 2) * eta)
        H11[m, pbc(m - 2, q)] = h4[1, 1] + o3[1, 1] * exp(-4j * pi * (m - 1) * eta) + o5[1, 1] * exp(4j * pi * (m - 1) * eta)
        H11[m, pbc(m - 3, q)] = v3[1, 1] * exp(-1j * 2 * pi * (m - 3 / 2) * eta) + v4[1, 1] * exp(1j * 2 * pi * (m - 3 / 2) * eta)
        H11[m, pbc(m - 4, q)] = o4[1, 1]

        H12T[m, m] = E0[2, 1] + v2[2, 1] * exp(4j * pi * m * eta) + v5[2, 1] * exp(-4j * pi * m * eta)
        H12T[m, pbc(m + 1, q)] = h2[2, 1] * exp(1j * 2 * pi * (m + 1 / 2) * eta) + h6[2, 1] * exp(-1j * 2 * pi * (m + 1 / 2) * eta)
        H12T[m, pbc(m + 2, q)] = h1[2, 1] + o2[2, 1] * exp(4j * pi * (m + 1) * eta) + o6[2, 1] * exp(-4j * pi * (m + 1) * eta)
        H12T[m, pbc(m + 3, q)] = v1[2, 1] * exp(1j * 2 * pi * (m + 3 / 2) * eta) + v6[2, 1] * exp(-1j * 2 * pi * (m + 3 / 2) * eta)
        H12T[m, pbc(m + 4, q)] = o1[2, 1]
        H12T[m, pbc(m - 1, q)] = h5[2, 1] * exp(-1j * 2 * pi * (m - 1 / 2) * eta) + h3[2, 1] * exp(1j * 2 * pi * (m - 1 / 2) * eta)
        H12T[m, pbc(m - 2, q)] = h4[2, 1] + o3[2, 1] * exp(4j * pi * (m - 1) * eta) + o5[2, 1] * exp(-4j * pi * (m - 1) * eta)
        H12T[m, pbc(m - 3, q)] = v3[2, 1] * exp(1j * 2 * pi * (m - 3 / 2) * eta) + v4[2, 1] * exp(-1j * 2 * pi * (m - 3 / 2) * eta)
        H12T[m, pbc(m - 4, q)] = o4[2, 1]

        H12[m, m] = E0[1, 2] + v2[1, 2] * exp(-4j * pi * m * eta) + v5[1, 2] * exp(4j * pi * m * eta)
        H12[m, pbc(m + 1, q)] = h2[1, 2] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) + h6[1, 2] * exp(1j * 2 * pi * (m + 1 / 2) * eta)
        H12[m, pbc(m + 2, q)] = h1[1, 2] + o2[1, 2] * exp(-4j * pi * (m + 1) * eta) + o6[1, 2] * exp(4j * pi * (m + 1) * eta)
        H12[m, pbc(m + 3, q)] = v1[1, 2] * exp(-1j * 2 * pi * (m + 3 / 2) * eta) + v6[1, 2] * exp(1j * 2 * pi * (m + 3 / 2) * eta)
        H12[m, pbc(m + 4, q)] = o1[1, 2]
        H12[m, pbc(m - 1, q)] = h5[1, 2] * exp(1j * 2 * pi * (m - 1 / 2) * eta) + h3[1, 2] * exp(-1j * 2 * pi * (m - 1 / 2) * eta)
        H12[m, pbc(m - 2, q)] = h4[1, 2] + o3[1, 2] * exp(-4j * pi * (m - 1) * eta) + o5[1, 2] * exp(4j * pi * (m - 1) * eta)
        H12[m, pbc(m - 3, q)] = v3[1, 2] * exp(-1j * 2 * pi * (m - 3 / 2) * eta) + v4[1, 2] * exp(1j * 2 * pi * (m - 3 / 2) * eta)
        H12[m, pbc(m - 4, q)] = o4[1, 2]

        H1T[m, m] = E0[1, 0] + v2[1, 0] * exp(4j * pi * m * eta) + v5[1, 0] * exp(-4j * pi * m * eta)
        H1T[m, pbc(m + 1, q)] = h2[1, 0] * exp(1j * 2 * pi * (m + 1 / 2) * eta) + h6[1, 0] * exp(-1j * 2 * pi * (m + 1 / 2) * eta)
        H1T[m, pbc(m + 2, q)] = h1[1, 0] + o2[1, 0] * exp(4j * pi * (m + 1) * eta) + o6[1, 0] * exp(-4j * pi * (m + 1) * eta)
        H1T[m, pbc(m + 3, q)] = v1[1, 0] * exp(1j * 2 * pi * (m + 3 / 2) * eta) + v6[1, 0] * exp(-1j * 2 * pi * (m + 3 / 2) * eta)
        H1T[m, pbc(m + 4, q)] = o1[1, 0]
        H1T[m, pbc(m - 1, q)] = h5[1, 0] * exp(-1j * 2 * pi * (m - 1 / 2) * eta) + h3[1, 0] * exp(1j * 2 * pi * (m - 1 / 2) * eta)
        H1T[m, pbc(m - 2, q)] = h4[1, 0] + o3[1, 0] * exp(4j * pi * (m - 1) * eta) + o5[1, 0] * exp(-4j * pi * (m - 1) * eta)
        H1T[m, pbc(m - 3, q)] = v3[1, 0] * exp(1j * 2 * pi * (m - 3 / 2) * eta) + v4[1, 0] * exp(-1j * 2 * pi * (m - 3 / 2) * eta)
        H1T[m, pbc(m - 4, q)] = o4[1, 0]

        H1[m, m] = E0[0, 1] + v2[0, 1] * exp(-4j * pi * m * eta) + v5[0, 1] * exp(4j * pi * m * eta)
        H1[m, pbc(m + 1, q)] = h2[0, 1] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) + h6[0, 1] * exp(1j * 2 * pi * (m + 1 / 2) * eta)
        H1[m, pbc(m + 2, q)] = h1[0, 1] + o2[0, 1] * exp(-4j * pi * (m + 1) * eta) + o6[0, 1] * exp(4j * pi * (m + 1) * eta)
        H1[m, pbc(m + 3, q)] = v1[0, 1] * exp(-1j * 2 * pi * (m + 3 / 2) * eta) + v6[0, 1] * exp(1j * 2 * pi * (m + 3 / 2) * eta)
        H1[m, pbc(m + 4, q)] = o1[0, 1]
        H1[m, pbc(m - 1, q)] = h5[0, 1] * exp(1j * 2 * pi * (m - 1 / 2) * eta) + h3[0, 1] * exp(-1j * 2 * pi * (m - 1 / 2) * eta)
        H1[m, pbc(m - 2, q)] = h4[0, 1] + o3[0, 1] * exp(-4j * pi * (m - 1) * eta) + o5[0, 1] * exp(4j * pi * (m - 1) * eta)
        H1[m, pbc(m - 3, q)] = v3[0, 1] * exp(-1j * 2 * pi * (m - 3 / 2) * eta) + v4[0, 1] * exp(1j * 2 * pi * (m - 3 / 2) * eta)
        H1[m, pbc(m - 4, q)] = o4[0, 1]

        H22[m, m] = E0[2, 2] + v2[2, 2] * exp(-4j * pi * m * eta) + v5[2, 2] * exp(4j * pi * m * eta)
        H22[m, pbc(m + 1, q)] = h2[2, 2] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) + h6[2, 2] * exp(1j * 2 * pi * (m + 1 / 2) * eta)
        H22[m, pbc(m + 2, q)] = h1[2, 2] + o2[2, 2] * exp(-4j * pi * (m + 1) * eta) + o6[2, 2] * exp(4j * pi * (m + 1) * eta)
        H22[m, pbc(m + 3, q)] = v1[2, 2] * exp(-1j * 2 * pi * (m + 3 / 2) * eta) + v6[2, 2] * exp(1j * 2 * pi * (m + 3 / 2) * eta)
        H22[m, pbc(m + 4, q)] = o1[2, 2]
        H22[m, pbc(m - 1, q)] = h5[2, 2] * exp(1j * 2 * pi * (m - 1 / 2) * eta) + h3[2, 2] * exp(-1j * 2 * pi * (m - 1 / 2) * eta)
        H22[m, pbc(m - 2, q)] = h4[2, 2] + o3[2, 2] * exp(-4j * pi * (m - 1) * eta) + o5[2, 2] * exp(4j * pi * (m - 1) * eta)
        H22[m, pbc(m - 3, q)] = v3[2, 2] * exp(-1j * 2 * pi * (m - 3 / 2) * eta) + v4[2, 2] * exp(1j * 2 * pi * (m - 3 / 2) * eta)
        H22[m, pbc(m - 4, q)] = o4[2, 2]

        H2T[m, m] = E0[2, 0] + v2[2, 0] * exp(4j * pi * m * eta) + v5[2, 0] * exp(-4j * pi * m * eta)
        H2T[m, pbc(m + 1, q)] = h2[2, 0] * exp(1j * 2 * pi * (m + 1 / 2) * eta) + h6[2, 0] * exp(-1j * 2 * pi * (m + 1 / 2) * eta)
        H2T[m, pbc(m + 2, q)] = h1[2, 0] + o2[2, 0] * exp(4j * pi * (m + 1) * eta) + o6[2, 0] * exp(-4j * pi * (m + 1) * eta)
        H2T[m, pbc(m + 3, q)] = v1[2, 0] * exp(1j * 2 * pi * (m + 3 / 2) * eta) + v6[2, 0] * exp(-1j * 2 * pi * (m + 3 / 2) * eta)
        H2T[m, pbc(m + 4, q)] = o1[2, 0]
        H2T[m, pbc(m - 1, q)] = h5[2, 0] * exp(-1j * 2 * pi * (m - 1 / 2) * eta) + h3[2, 0] * exp(1j * 2 * pi * (m - 1 / 2) * eta)
        H2T[m, pbc(m - 2, q)] = h4[2, 0] + o3[2, 0] * exp(4j * pi * (m - 1) * eta) + o5[2, 0] * exp(-4j * pi * (m - 1) * eta)
        H2T[m, pbc(m - 3, q)] = v3[2, 0] * exp(1j * 2 * pi * (m - 3 / 2) * eta) + v4[2, 0] * exp(-1j * 2 * pi * (m - 3 / 2) * eta)
        H2T[m, pbc(m - 4, q)] = o4[2, 0]

        H2[m, m] = E0[0, 2] + v2[0, 2] * exp(-4j * pi * m * eta) + v5[0, 2] * exp(4j * pi * m * eta)
        H2[m, pbc(m + 1, q)] = h2[0, 2] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) + h6[0, 2] * exp(1j * 2 * pi * (m + 1 / 2) * eta)
        H2[m, pbc(m + 2, q)] = h1[0, 2] + o2[0, 2] * exp(-4j * pi * (m + 1) * eta) + o6[0, 2] * exp(4j * pi * (m + 1) * eta)
        H2[m, pbc(m + 3, q)] = v1[0, 2] * exp(-1j * 2 * pi * (m + 3 / 2) * eta) + v6[0, 2] * exp(1j * 2 * pi * (m + 3 / 2) * eta)
        H2[m, pbc(m + 4, q)] = o1[0, 2]
        H2[m, pbc(m - 1, q)] = h5[0, 2] * exp(1j * 2 * pi * (m - 1 / 2) * eta) + h3[0, 2] * exp(-1j * 2 * pi * (m - 1 / 2) * eta)
        H2[m, pbc(m - 2, q)] = h4[0, 2] + o3[0, 2] * exp(-4j * pi * (m - 1) * eta) + o5[0, 2] * exp(4j * pi * (m - 1) * eta)
        H2[m, pbc(m - 3, q)] = v3[0, 2] * exp(-1j * 2 * pi * (m - 3 / 2) * eta) + v4[0, 2] * exp(1j * 2 * pi * (m - 3 / 2) * eta)
        H2[m, pbc(m - 4, q)] = o4[0, 2]

    H[0:q, 0:q] = H0
    H[0:q, q : 2 * q] = H1
    H[0:q, 2 * q : 3 * q] = H2
    H[q : 2 * q, 0:q] = H1T
    H[q : 2 * q, q : 2 * q] = H11
    H[q : 2 * q, 2 * q : 3 * q] = H12
    H[2 * q : 3 * q, 0:q] = H2T
    H[2 * q : 3 * q, q : 2 * q] = H12T
    H[2 * q : 3 * q, 2 * q : 3 * q] = H22

    return H


def process(N: int, band: int, choice: int, qmax: int, fileData: dict, model: str):

    fileEnergy = fileData["fileEnergy"]
    fileMoment = fileData["fileMoment"]

    data = paraTNN(choice, model)
    a_lattice = data["alattice"]
    E0, h1, h2, h3, h4, h5, h6 = IR(data)
    v1, v2, v3, v4, v5, v6 = IRNN(data)
    o1, o2, o3, o4, o5, o6 = IRTNN(data)
    m0 = 5.6770736 / 100
    hb = 0.658229
    irreducibleMatrix = {
        "NN": [E0, h1, h2, h3, h4, h5, h6],
        "NNN": [v1, v2, v3, v4, v5, v6],
        "TNN": [o1, o2, o3, o4, o5, o6],
    }

    p = 1

    PxArr = cp.zeros((N, N))
    PyArr = cp.zeros((N, N))
    pPlusArr = cp.zeros((N, N))
    pMinusArr = cp.zeros((N, N))
    moduloPArr = cp.zeros((N, N))
    # dHam_kx = cp.zeros((6 * qmax, 6 * qmax), dtype=complex)
    # dHam_ky = cp.zeros((6 * qmax, 6 * qmax), dtype=complex)

    arrEigen = {}
    for q in tqdm(range(6 * qmax), desc="Create array eigenvalue"):
        arrEigen[f"L_{q}"] = cp.zeros([N, N])

    akx, aky = cp.zeros((N, N)), cp.zeros((N, N))
    dk = (4 * pi / a_lattice) / (N - 1)
    for i1 in range(N):
        for j1 in range(N):
            akx[i1][j1] = (-2 * pi / a_lattice + (i1) * dk) * 1
            aky[i1][j1] = (-2 * pi / a_lattice + (j1) * dk) * 1

    for i in tqdm(range(N), desc="vong lap i"):
        for j in range(N):
            if cp.gcd(p, qmax) != 1:
                continue

            Ham = HamTNN(band, a_lattice, p, 2 * qmax, akx[i][j], aky[i][j], irreducibleMatrix)
            dHam_kx = HamTNN_kx(band, a_lattice, p, 2 * qmax, akx[i][j], aky[i][j], irreducibleMatrix)
            dHam_ky = HamTNN_ky(band, a_lattice, p, 2 * qmax, akx[i][j], aky[i][j], irreducibleMatrix)

            eigenvalue, eigenvector = LA.eigh(Ham)

            for q in range(6 * qmax):
                arrEigen[f"L_{q}"] = eigenvalue[q]

            sumpx = 0 + 0j
            sumpy = 0 + 0j
            for bandi in range(6 * qmax):
                for bandj in range(6 * qmax):

                    sumpx += cp.conjugate(eigenvector[2 * qmax][bandj]) * dHam_kx[bandi][bandj] * eigenvector[2 * qmax + 1][bandi]
                    sumpy += cp.conjugate(eigenvector[2 * qmax][bandj]) * dHam_ky[bandi][bandj] * eigenvector[2 * qmax + 1][bandi]
            px = sumpx * m0 / hb
            py = sumpy * m0 / hb
            moduloP = sqrt(abs(px) ** 2 + abs(py) ** 2)
            pPlus = px + 1j * py
            pMinus = px - 1j * py

            PxArr = px
            PyArr = py

            moduloPArr[i][j] = moduloP
            pPlusArr[i][j] = abs(pPlus)
            pMinusArr[i][j] = abs(pMinus)

    with open(fileEnergy, "w", newline="") as writefile:
        header = [
            "kx",
            "ky",
        ]
        for q in range(6 * qmax):
            header.append(list(arrEigen.keys())[q])

        writer = csv.DictWriter(writefile, fieldnames=header, delimiter=",")
        writer.writeheader()
        row = {}
        for i in range(N):
            for j in range(N):
                row["kx"] = akx[i][j] / (2 * pi / a_lattice)
                row["ky"] = aky[i][j] / (2 * pi / a_lattice)

                for k in range(6 * qmax):
                    row[f"L_{k}"] = arrEigen[f"L_{k}"]

                writer.writerow(row)
            writefile.write("\n")

    with open(fileMoment, "w", newline="") as writefile:
        print(pPlusArr)
        print(len(pPlusArr))
        header = [
            "kx",
            "ky",
            "pPlus",
            "pMinus",
            "pAbs",
        ]

        writer = csv.DictWriter(writefile, fieldnames=header, delimiter=",")
        writer.writeheader()
        row = {}
        for i in range(N):
            for j in range(N):
                row["kx"] = akx[i][j] / (2 * pi / a_lattice)
                row["ky"] = aky[i][j] / (2 * pi / a_lattice)
                row["pPlus"] = pPlusArr[i][j]
                row["pMinus"] = pMinusArr[i][j]
                row["pAbs"] = moduloPArr[i][j]

                writer.writerow(row)
            writefile.write("\n")

    return None


def solve_p(args):
    p, qmax, coeff, band, alattice, kx, ky, irreducibleMatrix, modelNeighbor, phi0, S, charge, hbar, m_e = args

    if cp.gcd(p, qmax) != 1:
        return None  # bỏ qua

    eta = p / qmax
    B = eta * phi0 / S

    # chọn Hamiltonian theo mô hình
    if modelNeighbor == "NN":
        Ham = HamNN(band, alattice, p, coeff * qmax, kx, ky, irreducibleMatrix)
    elif modelNeighbor == "TNN":
        Ham = HamTNN(band, alattice, p, coeff * qmax, kx, ky, irreducibleMatrix)
    else:
        return None

    eigenvals = LA.eigvalsh(Ham)

    E_2q = eigenvals[coeff * qmax + 8]
    E_2q1 = eigenvals[coeff * qmax + 9]
    E_2q2 = eigenvals[coeff * qmax - 17]

    En_valence = eigenvals[coeff * qmax - 17] + eigenvals[coeff * qmax - 18]
    En1_valence = eigenvals[coeff * qmax - 27] + eigenvals[coeff * qmax - 28]

    En_conduction = eigenvals[coeff * qmax]
    En1_conduction = eigenvals[coeff * qmax + 9]

    omega_valence = abs((En1_valence - En_valence) * charge / hbar)
    omega_conduction = (En1_conduction - En_conduction) * charge / hbar
    m_eff_v = charge * B / omega_valence
    m_eff_c = charge * B / omega_conduction

    m_ratio_v = m_eff_v / m_e
    m_ratio_c = m_eff_c / m_e

    # trả về toàn bộ dòng dữ liệu cho mỗi eigenvalue
    rows = []
    for i in range(coeff * band * qmax):
        rows.append(
            {
                "eta": eta,
                "B_values": B,
                "evalues": eigenvals[i],
                "E_level1": E_2q,
                "E_level2": E_2q1,
                # "E_level3": E_2q2,
                "m*_v": m_ratio_v,
                "m*_c": m_ratio_c,
                "ω_v": omega_valence,
                "ω_c": omega_conduction,
            }
        )
    return rows


def butterfly(band, choice: int, qmax: int, kpoint: str, fileData, model: dict):
    modelParameters = model["modelParameters"]
    modelNeighbor = model["modelNeighbor"]

    functionMapping = {"TNN": paraTNN, "NN": paraNN}
    data = functionMapping[modelNeighbor](choice, modelParameters)

    matt = data["material"]
    alattice = data["alattice"] * 1e-10
    E0, h1, h2, h3, h4, h5, h6 = IR_tran(data)

    v1 = v2 = v3 = v4 = v5 = v6 = 0
    o1 = o2 = o3 = o4 = o5 = o6 = 0

    h = 6.62607007e-34
    hbar = h / (2 * cp.pi)
    charge = 1.602176621e-19
    phi0 = h / charge
    S = cp.sqrt(3) * alattice**2 / 2
    m_e = 9.10938356e-31

    if modelNeighbor == "TNN":
        v1, v2, v3, v4, v5, v6 = IRNN_tran(data)
        o1, o2, o3, o4, o5, o6 = IRTNN_tran(data)

    irreducibleMatrix = {
        "NN": [E0, h1, h2, h3, h4, h5, h6],
        "NNN": [v1, v2, v3, v4, v5, v6],
        "TNN": [o1, o2, o3, o4, o5, o6],
    }

    kpoints = {
        "G": [0, 0],
        "K1": [4 * cp.pi / (3 * alattice), 0],
        "K2": [-4 * cp.pi / (3 * alattice), 0],
        "M": [cp.pi / (alattice), cp.pi / (cp.sqrt(3) * alattice)],
    }

    kx, ky = kpoints[kpoint]

    coeff = 1
    args_list = [(p, qmax, coeff, band, alattice, kx, ky, irreducibleMatrix, modelNeighbor, phi0, S, charge, hbar, m_e) for p in range(1, qmax + 1)]

    # chạy song song với 6 core
    with mp.Pool(processes=6) as pool:
        results = list(tqdm(pool.imap(solve_p, args_list), total=qmax, desc=f"{matt}", ascii=" #"))

    # ghi file CSV
    with open(fileData, "w", newline="") as writefile:
        header = ["eta", "B_values", "evalues", "E_level1", "E_level2", "m*_v", "m*_c", "ω_v", "ω_c"]
        writer = csv.DictWriter(writefile, fieldnames=header, delimiter=",")
        writer.writeheader()
        for rows in results:
            if rows is not None:
                writer.writerows(rows)

    return None


def main():
    qmax = 797
    n_levels = 8
    choice = 0
    bandNumber = 3
    bandSelector = "A"
    modelPara = "GGA"
    modelNeighbor = "NN"
    model = {"modelParameters": modelPara, "modelNeighbor": modelNeighbor}
    kpoint1 = "G"
    # kpoint2 = "K1"
    # kpoint3 = "K2"
    # kpoint4 = "M"
    # listKpoint = [kpoint1, kpoint2, kpoint3, kpoint4]
    data = paraTNN(choice, model["modelParameters"])
    matt = data["material"]

    time_run = datetime.now().strftime("%a-%m-%Y")
    dir = f"./{time_run}/{modelNeighbor}/"
    print("folder direction: ", dir)
    print("Core available: ", mp.cpu_count())
    os.makedirs(os.path.dirname(dir), exist_ok=True)

    fileButterflyK1 = f"{dir}{bandNumber}band_Lambda2q_dataHofstadterButterfly_q_{qmax}_{matt}_{modelPara}_{kpoint1}.dat"
    # fileButterflyK2 = f"{dir}{bandNumber}band_Lambda2q_dataHofstadterButterfly_q_{qmax}_{matt}_{modelPara}_{kpoint2}.dat"

    filePlotC_k1 = f"{dir}{bandNumber}band_PlotEigenVectors_q_{qmax}_{matt}_{modelPara}_{kpoint1}_vals_vecs.dat"
    filePlotC_k2 = f"{dir}{bandNumber}band_PlotEigenVectors_2q+1_{qmax}_{matt}_{modelPara}_{kpoint1}_vals_vecs.dat"
    # filePlotC_k3 = f"{dir}{bandNumber}band_PlotEigenVectors_q_{qmax}_{matt}_{modelPara}_{kpoint3}_vals_vecs.dat"
    # filePlotC_k4 = f"{dir}{bandNumber}band_PlotEigenVectors_q_{qmax}_{matt}_{modelPara}_{kpoint4}_vals_vecs.dat"

    fileMatrix = f"{bandNumber}band_Matrix_q_{qmax}{time_run}.dat"
    file_plot_Matrix_Gnu = f"{bandNumber}band_Matrix_q_{qmax}{time_run}_h0.gnuplot"

    filegnu = f"{bandNumber}band_plotHofstadterButterfly_q={qmax}.gnuplot"

    print("file data buttefly K1: ", fileButterflyK1)
    # print("file data buttefly K2: ", fileButterflyK2)
    print("file data: ", filePlotC_k1)
    print("file gnuplot: ", filegnu)
    print("file Matrix: ", fileMatrix)
    print("file Matrix GNU: ", file_plot_Matrix_Gnu)

    fileData = {
        "fileButterfly_K1": fileButterflyK1,
        "fileMatrix": fileMatrix,
        "filePlotMatrix": file_plot_Matrix_Gnu,
        "fileWriteGnu": filegnu,
        "fileG": filePlotC_k1,
    }

    butterflyK1 = butterfly(bandNumber, choice, qmax, kpoint1, fileButterflyK1, model)
    # butterflyK2 = butterfly(bandNumber, choice, qmax, kpoint2, fileData, model)

    # dataK1 = waveFunction(bandNumber, choice, qmax, kpoint1, fileData, model)
    # dataK2 = waveFunction(bandNumber, choice, qmax, kpoint2, fileData, model)
    # dataK3 = waveFunction(bandNumber, choice, qmax, kpoint3, fileData, model)
    # dataK4 = waveFunction(bandNumber, choice, qmax, kpoint4, fileData, model)

    # PlotMatrixGNU(fileMatrix, file_plot_Matrix_Gnu)
    # saveFunction(currentProg, "fileData")

    return None


if __name__ == "__main__":
    main()
