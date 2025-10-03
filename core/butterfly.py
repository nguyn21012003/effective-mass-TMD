import csv

import numpy as np
from numpy import linalg as LA
from numpy import pi, sqrt
from tqdm import tqdm

from core.HamTMD import HamNN
from core.HamTMDNN import HamTNN


def butterfly(dataInit: dict, irreducibleMatrix, fileSave: str):
    ##### chi so dau vao
    coeff = dataInit["coeff"]
    modelNeighbor = dataInit["modelNeighbor"]
    alattice = dataInit["alattice"]
    kx, ky = dataInit["kpoint"]
    qmax = dataInit["qmax"]

    Hamiltonian = None
    h = 6.62607007e-34
    hbar = h / (2 * pi)
    charge = 1.602176621e-19
    phi0 = h / charge
    S = sqrt(3) * alattice**2 / 2
    m_e = 9.10938356e-31
    for p in range(qmax):
        eta = p / qmax
        B = eta * phi0 / S
        if modelNeighbor == "NN":
            Hamiltonian = HamNN(alattice, p, coeff * qmax, kx, ky, irreducibleMatrix)
        elif modelNeighbor == "TNN":
            Hamiltonian = HamTNN(alattice, p, coeff * qmax, kx, ky, irreducibleMatrix)
        if np.gcd(p, qmax) != 1:
            continue
        eigenvals = LA.eigvalsh(Hamiltonian)
        with open(fileSave, "w", newline="") as writefile:
            header = [
                "eta",
                "B",
                "energy",
            ]

            row = {}
            writer = csv.DictWriter(writefile, fieldnames=header, delimiter=",")
            writer.writeheader()
            for i in range(len(eigenvals)):
                row["eta"] = eta
                row["B"] = B
                row["energy"] = eigenvals[i]
                writer.writerow(row)

    return None
