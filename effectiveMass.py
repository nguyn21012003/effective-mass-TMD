import numpy as np
from numpy import exp
from numpy import linalg as LA
from numpy import pi, sqrt
from tqdm import tqdm

from core.irrMatrix import IR, IRNN, IRTNN
from core.irrMatrixTransform import IR as IR_tran
from core.irrMatrixTransform import IRNN as IRNN_tran
from core.irrMatrixTransform import IRTNN as IRTNN_tran
from core.parameters import paraNN, paraTNN


def massTNN(material: str, model: str):
    data = paraTNN(material, model)
    # matt, a_lattice, e1, e2, t0, t1, t2, t11, t12, t22 = paraNN(argument)
    alattice = data["alattice"] * 1e-10

    h0, h1, h2, h3, h4, h5, h6 = IR_tran(data)
    o1, o2, o3, o4, o5, o6 = IRTNN_tran(data)
    v1, v2, v3, v4, v5, v6 = IRNN_tran(data)

    G = 4 * pi / (sqrt(3) * alattice)
    dk = G / 500
    w = np.zeros([3, 3, 3])
    meffx = np.zeros(3)
    meffy = np.zeros(3)

    h = 6.62607007e-34
    hbar = h / (2 * pi)
    m_e = 9.10938356e-31

    for i in tqdm(range(3)):
        for j in range(3):
            kx = 4 * pi / (3 * alattice) + (i - 1) * dk
            ky = (j - 1) * dk
            alpha = kx / 2 * alattice
            beta = sqrt(3) / 2 * ky * alattice
            ham = (
                h0
                + exp(2j * alpha) * h1
                + exp(1j * (alpha - beta)) * h2
                + exp(1j * (-alpha - beta)) * h3
                + exp(-2j * alpha) * h4
                + exp(1j * (-alpha + beta)) * h5
                + exp(1j * (alpha + beta)) * h6
                + exp(4j * alpha) * o1
                + exp(2j * (alpha - beta)) * o2
                + exp(2j * (-alpha - beta)) * o3
                + exp(-4j * alpha) * o4
                + exp(2j * (-alpha + beta)) * o5
                + exp(2j * (alpha + beta)) * o6
                + exp(1j * (3 * alpha - beta)) * v1
                + exp(1j * (-2 * beta)) * v2
                + exp(1j * (-3 * alpha - beta)) * v3
                + exp(1j * (-3 * alpha + beta)) * v4
                + exp(1j * (2 * beta)) * v5
                + exp(1j * (3 * alpha + beta)) * v6
            )

            vals = LA.eigvalsh(ham)
            for ib in range(3):
                w[ib, i, j] = vals[ib] * 1.602176634e-19

    for ib in range(3):
        d2ex = (w[ib, 2, 1] - 2 * w[ib, 1, 1] + w[ib, 0, 1]) / (dk**2)
        d2ey = (w[ib, 1, 2] - 2 * w[ib, 1, 1] + w[ib, 1, 0]) / (dk**2)
        meffx[ib] = (hbar**2) / (d2ex * m_e)
        meffy[ib] = (hbar**2) / (d2ey * m_e)

    meff_e = meffx[1]
    meff_h = abs(meffx[0])

    mr = meff_e * meff_h / (meff_e + meff_h)

    return meff_e, meff_h, mr


def massNN(material: str, model: str):
    data = paraNN(material, model)
    # matt, a_lattice, e1, e2, t0, t1, t2, t11, t12, t22 = paraNN(argument)
    alattice = data["alattice"] * 1e-10

    E0, h1, h2, h3, h4, h5, h6 = IR(data)
    G = 4 * pi / (sqrt(3) * alattice)
    dk = G / 500
    w = np.zeros([3, 3, 3])
    meffx = np.zeros(3)
    meffy = np.zeros(3)

    h = 6.62607007e-34
    hbar = h / (2 * pi)
    m_e = 9.10938356e-31

    for i in tqdm(range(3)):
        for j in range(3):
            kx = 4 * pi / (3 * alattice) + (i - 1) * dk
            ky = (j - 1) * dk
            alpha = kx / 2 * alattice
            beta = sqrt(3) / 2 * ky * alattice
            ham = (
                E0
                + exp(2j * alpha) * h1
                + exp(1j * (alpha - beta)) * h2
                + exp(1j * (-alpha - beta)) * h3
                + exp(-2j * alpha) * h4
                + exp(1j * (-alpha + beta)) * h5
                + exp(1j * (alpha + beta)) * h6
            )

            vals = LA.eigvalsh(ham)
            for ib in range(3):
                w[ib, i, j] = vals[ib] * 1.602176634e-19

    for ib in range(3):
        d2ex = (w[ib, 2, 1] - 2 * w[ib, 1, 1] + w[ib, 0, 1]) / (dk**2)
        d2ey = (w[ib, 1, 2] - 2 * w[ib, 1, 1] + w[ib, 1, 0]) / (dk**2)
        meffx[ib] = (hbar**2) / (d2ex * m_e)
        meffy[ib] = (hbar**2) / (d2ey * m_e)

    meff_e = meffx[1]
    meff_h = abs(meffx[0])

    mr = meff_e * meff_h / (meff_e + meff_h)

    return meff_e, meff_h, mr


def main():
    modelParameter = "GGA"
    material = "MoS2"
    modelNeighbor = "NN"
    if modelNeighbor == "NN":
        meff_e, meff_h, mr = massNN(material, modelParameter)
        print(round(meff_e, 4), round(meff_h, 4), round(mr, 4), "\n", material, modelNeighbor)
        meB = 0.4605923002043953
        mhB = 0.6597343154912579
        mrB = (meB * mhB) / (meB + mhB)
        por_me = (meB - meff_e) / meff_e * 100
        por_mh = (mhB - meff_h) / meff_h * 100
        por_mr = (mrB - mr) / mr * 100
        # print(por_me, por_mh, por_mr)

    elif modelNeighbor == "TNN":
        meff_e, meff_h, mr = massTNN(material, modelParameter)
        print(round(meff_e, 4), round(meff_h, 4), round(mr, 4), "\n", material, modelNeighbor)
        mhB = 0.5584958164360369
        meB = 0.4263402692114003
        mrB = (meB * mhB) / (meB + mhB)
        por_me = (meB - meff_e) / meff_e * 100
        por_mh = (mhB - meff_h) / meff_h * 100
        por_mr = (mrB - mr) / mr * 100
        # print(meB, mhB)
        # print(por_me, por_mh, por_mr)


if __name__ == "__main__":
    main()
