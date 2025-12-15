import csv
import os
from datetime import datetime

import numpy as np
from numpy import exp
from numpy import linalg as LA
from numpy import pi, sqrt
from tqdm import tqdm

from core.irrMatrixTransform import IR as IR_tran
from core.parameters import paraNN, paraTNN


def eigenvalue(material: str, fileBandStruct: str):
    N = 301
    model = "GGA"
    data = paraNN(material, model)
    # matt, a_lattice, e1, e2, t0, t1, t2, t11, t12, t22 = paraNN(argument)
    a_lattice = data["alattice"]

    E0, h1, h2, h3, h4, h5, h6 = IR_tran(data)

    L1 = np.zeros((N, N))
    L2 = np.zeros((N, N))
    L3 = np.zeros((N, N))

    G = 4 * pi / (sqrt(3) * a_lattice)
    ak1 = np.linspace(-G / 2, G / 2, N)
    ak2 = np.linspace(-G / 2, G / 2, N)
    akx = np.zeros((len(ak1), len(ak2)))
    aky = np.zeros((len(ak1), len(ak2)))

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
    return None


def main():
    modelNeighbor = "NN"
    material = "MoS2"
    time_run = datetime.now().strftime("%a-%m-%d")
    dir = f"./{time_run}/{modelNeighbor}/"
    os.makedirs(os.path.dirname(dir), exist_ok=True)
    fileBandStruct = f"{dir}/{material}_eigenvalue.csv"
    eigenvalue(material, fileBandStruct)


if __name__ == "__main__":
    main()
