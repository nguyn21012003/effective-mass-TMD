import csv
import os
from datetime import datetime

import numpy as np
from numpy import cos, exp
from numpy import linalg as LA
from numpy import pi, sin, sqrt
from scipy import constants
from tqdm import tqdm

from file_python.irrMatrix import IR, IRNN, IRTNN
from file_python.irrMatrixTransform import IR as IR_tran
from file_python.irrMatrixTransform import IRNN as IRNN_tran
from file_python.irrMatrixTransform import IRTNN as IRTNN_tran
from file_python.parameters import paraNN, paraTNN


hbar = constants.hbar
B = 0
e = 9.1e-31
N = 200
N = 201


def eigenvalue(choice, fileBandStruct):
    model = "GGA"
    data = paraTNN(choice, model)
    # matt, a_lattice, e1, e2, t0, t1, t2, t11, t12, t22 = paraNN(argument)
    a_lattice = data["alattice"]

    E0, h1, h2, h3, h4, h5, h6 = IR_tran(data)
    o1, o2, o3, o4, o5, o6 = IRTNN_tran(data)
    v1, v2, v3, v4, v5, v6 = IRNN_tran(data)

    L1 = np.zeros((N, N))
    L2 = np.zeros((N, N))
    L3 = np.zeros((N, N))

    ak1 = np.zeros(N)
    ak2 = np.zeros(N)
    G = 4 * pi / (sqrt(3) * a_lattice)

    k1min = -G / 2
    k1max = G / 2
    k2min = -G / 2
    k2max = G / 2

    dk1 = (k1max - k1min) / (N - 1)
    dk2 = (k2max - k2min) / (N - 1)

    for i in range(N):
        ak1[i] = k1min + i * dk1

    for j in range(N):
        ak2[j] = k2min + j * dk2

    akx = np.zeros((len(ak1), len(ak2)))
    aky = np.zeros((len(ak1), len(ak2)))

    # kx = np.zeros((N, N))
    # ky = np.zeros((N, N))
    # dki = 4 * pi * sqrt(3) / (3 * (N - 1) * a_lattice)
    # dkj = 4 * pi * sqrt(3) / (3 * (N - 1) * a_lattice)
    for i in tqdm(range(N)):
        for j in range(N):
            akx[i][j] = sqrt(3) / 2 * (ak1[i] + ak2[j])
            aky[i][j] = -1 / 2 * (ak1[i] - ak2[j]) * 0
            alpha = akx[i][j] / 2 * a_lattice
            beta = sqrt(3) / 2 * aky[i][j] * a_lattice
            ham = (
                E0
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

            w = LA.eigvalsh(ham)
            L1[i, j] = float(w[0])
            L2[i, j] = float(w[1])
            L3[i, j] = float(w[2])

    with open(fileBandStruct, "w", newline="") as writefile:
        header = [
            "kx",
            "ky",
            "Lambda1",
            "Lambda2",
            "Lambda3",
            # "Lambda4",
            # "Lambda5",
            # "Lambda6",
        ]
        writer = csv.DictWriter(writefile, fieldnames=header)
        writer.writeheader()

        for i in range(len(L1)):
            # for j in range(len(L1)):
            writer.writerow(
                {
                    "kx": akx[i][i],
                    "ky": aky[i][i],
                    "Lambda1": L1[i][i],
                    "Lambda2": L2[i][i],
                    "Lambda3": L3[i][i],
                    # "Lambda4": L4[i][i],
                    # "Lambda5": L5[i][i],
                    # "Lambda6": L6[i][i],
                }
            )


def main():
    model = "TNN"
    time_run = datetime.now().strftime("%a-%m-%Y")
    dir = f"./bandstructure/{model}/{time_run}/"
    os.makedirs(os.path.dirname(dir), exist_ok=True)
    fileBandStruct = f"{dir}/eigenvalue.csv"
    eigenvalue(0, fileBandStruct)


if __name__ == "__main__":
    main()
