import csv

import numpy as np
from core.HamTMD import HamNN
from core.HamTMDNN import HamTNN
from numpy import linalg as LA
from numpy import pi, sqrt
from tqdm import tqdm


def calcMass(dataInit, irreducibleMatrix, fileSave):
    p = dataInit["p"]
    coeff = dataInit["coeff"]
    numberWave = dataInit["numberWaveFunction"]  # so ham song can khao sat
    modelNeighbor = dataInit["modelNeighbor"]
    alattice = dataInit["alattice"]
    kx, ky = dataInit["kpoint"]
    qmax = dataInit["qmax"]
    alattice = dataInit["alattice"] * 1e-10

    h = 6.62607007e-34
    hbar = h / (2 * pi)
    charge = 1.602176621e-19
    phi0 = h / charge
    S = sqrt(3) * alattice**2 / 2
    m_e = 9.10938356e-31

    Hamiltonian = None

    with open(fileSave, "w", newline="") as writefile:
        header = [
            "eta",
            "B_values",
            "m*_v",
            "m*_c",
            "ω_c",
            "ω_v",
        ]

        writer = csv.DictWriter(writefile, fieldnames=header, delimiter=",")
        writer.writeheader()
        for p in tqdm(range(1, qmax + 1), ascii=" #", desc=f"Solve Hamiltonian"):
            if np.gcd(p, qmax) != 1:
                continue
            eta = p / (qmax)  ## the magnetic ratio require that p and q must be co-prime
            B = eta * phi0 / S  ## the actually B which are taken from eta

            if modelNeighbor == "NN":
                Hamiltonian = HamNN(alattice, p, coeff * qmax, kx, ky, irreducibleMatrix)
            elif modelNeighbor == "TNN":
                Hamiltonian = HamTNN(alattice, p, coeff * qmax, kx, ky, irreducibleMatrix)

            eigenvals = LA.eigvalsh(Hamiltonian)

            omega_v = 0
            omega_c = 0
            meff_v = 0
            meff_c = 0
            offset = {
                3129: (27, 37),  # 15
                2346: (23, 33),  # 20
                1877: (21, 31),  # 25
                1564: (21, 31),  # 30
                1341: (21, 29),  # 35
                1173: (19, 29),  # 40
                1043: (17, 27),  # 45
                939: (17, 27),  # 50
            }

            # valuesBandLambda = {}
            # for i in range(numWave + 1):
            # valuesBandLambda[f"E_2q{i}"] = eigenvals[coeff * qmax - i]

            if qmax in offset:
                off1, off2 = offset[qmax]
                En_valence = eigenvals[coeff * qmax - off1]  ### Only K-point at Landau level n = 0
                En1_valence = eigenvals[coeff * qmax - off2]  ### Only K-point at Landau level n = 1
                omega_v = abs((En1_valence - En_valence) * charge / hbar)
                meff_v = charge * B / omega_v

            En_conduction = eigenvals[coeff * qmax + 4]
            En1_conduction = eigenvals[coeff * qmax + 8]
            omega_c = (En1_conduction - En_conduction) * charge / hbar
            meff_c = charge * B / omega_c

            m_ratio_v = meff_v / m_e
            m_ratio_c = meff_c / m_e
            print(m_ratio_v)
            print(m_ratio_c, "\n")
            row = {
                "eta": eta,
                "B_values": B,
            }
            row["m*_v"] = m_ratio_v
            row["m*_c"] = m_ratio_c
            row["ω_v"] = omega_v
            row["ω_c"] = omega_c
            # for i in range(numWave + 1):
            #     row[f"E2q{i}"] = valuesBandLambda[f"E_2q{i}"]
            # for i in range(coeff * band * qmax):
            #    row["evalues"] = eigenvals[i]
            writer.writerow(row)

    return None
