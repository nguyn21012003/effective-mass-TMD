import csv

import numpy as np
from numpy import linalg as LA
from numpy import sqrt
from tqdm import tqdm

from core.HamTMD import HamNN
from core.HamTMDNN import HamTNN


def butterfly(dataInit: dict, irreducibleMatrix, fileSave: str):
    ##### chi so dau vao
    coeff = dataInit["coeff"]
    modelNeighbor = dataInit["modelNeighbor"]
    alattice = dataInit["alattice"] * 1e-10
    k = dataInit["kpoint"]
    qmax = dataInit["qmax"]
    lambd = dataInit["lambda"]

    ham, hamu, hamd = None, None, None
    h = 6.62607007e-34
    charge = 1.602176621e-19
    phi0 = h / charge
    S = sqrt(3) * alattice**2 / 2
    with open(fileSave, "w", newline="") as writefile:
        header = [
            "eta",
            "B",
            "energy",
            "up",
            "down",
        ]
        writer = csv.DictWriter(writefile, fieldnames=header, delimiter=",")
        writer.writeheader()
        for p in tqdm(range(qmax), desc="Butterfly", colour="blue"):
            if modelNeighbor == "NN":
                ham, hamu, hamd = HamNN(
                    alattice, p, coeff * qmax, k, lambd, irreducibleMatrix
                )
            elif modelNeighbor == "TNN":
                ham, hamu, hamd = HamTNN(
                    alattice, p, coeff * qmax, k, lambd, irreducibleMatrix
                )
            eta = p / qmax
            B = eta * phi0 / S
            if np.gcd(p, qmax) != 1:
                continue
            eigenvals = LA.eigvalsh(ham)
            vals_up = LA.eigvalsh(hamu)
            vals_down = LA.eigvalsh(hamd)
            row = {}
            row["eta"] = eta
            row["B"] = B
            for e, eu, ed in zip(eigenvals, vals_up, vals_down):
                # for e in eigenvals:
                row["energy"] = e
                row["up"] = eu
                row["down"] = ed
                writer.writerow(row)

    return None
