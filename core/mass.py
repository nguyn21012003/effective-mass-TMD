import csv

import numpy as np
from numpy import linalg as LA
from numpy import pi, sqrt
from tqdm import tqdm

from core.HamTMD import HamNN
from core.HamTMDNN import HamTNN


def calcMass(dataInit, irreducibleMatrix, fileSave):
    p = dataInit["p"]
    coeff = dataInit["coeff"]
    print(coeff)
    numberWave = dataInit["numberWaveFunction"]  # so ham song can khao sat
    modelNeighbor = dataInit["modelNeighbor"]
    kx, ky = dataInit["kpoint"]
    qmax = dataInit["qmax"]
    alattice = dataInit["alattice"] * 1e-10  # angstrogn sang m

    h = 6.62607007e-34  # kg m**2 / s**2
    hbar = h / (2 * pi)
    charge = 1.602176621e-19  # Coulomb
    phi0 = h / charge
    S = sqrt(3) * alattice**2 / 2
    m_e = 9.10938356e-31  # kg
    v_f = 6.65e5  # Vận tốc Fermi trong chất rắn, đơn vị là m/s

    Hamiltonian = None
    # qrange = [round(phi0 / (S * B)) for B in B_values]
    # B_values = list(range(15, 505, 5)) # đơn vị là Tesla
    qrange = [
        3129,
        2346,
        1877,
        1564,
        1341,
        1173,
        1043,
        939,
        853,
        782,
        722,
        670,
        626,
        587,
        552,
        521,
        494,
        469,
        447,
        427,
        408,
        391,
        375,
        361,
        348,
        335,
        324,
        313,
        303,
        293,
        284,
        276,
        268,
        261,
        254,
        247,
        241,
        235,
        229,
        223,
        218,
        213,
        209,
        204,
        200,
        196,
        192,
        188,
        184,
        180,
        177,
        174,
        171,
        168,
        165,
        162,
        159,
        156,
        154,
        151,
        149,
        147,
        144,
        142,
        140,
        138,
        136,
        134,
        132,
        130,
        129,
        127,
        125,
        123,
        122,
        120,
        119,
        117,
        116,
        114,
        113,
        112,
        110,
        109,
        108,
        107,
        105,
        104,
        103,
        102,
        101,
        100,
        99,
        98,
        97,
        96,
        95,
        94,
    ]
    # print(B_values,"\n")
    print(qrange)

    with open(fileSave, "w", newline="") as writefile:
        header = [
            # "eta",
            "B_values",
            # "evalues",
            "m_hK1",
            "m_hK2",
            "m_eK1",
            "m_eK2",
        ]

        for i in range(numberWave):
            header.append(f"E2q{i}")
        writer = csv.DictWriter(writefile, fieldnames=header, delimiter=",")
        writer.writeheader()
        # for qmax in qrange:
        for qmax in tqdm(qrange, ascii=" #", desc="Solve Hamiltonian", colour="blue"):
            if np.gcd(p, qmax) != 1:
                continue
            eta = p / qmax  ## the magnetic ratio require that p and q must be co-prime
            B = eta * phi0 / S  ## the actually B which are taken from eta

            if modelNeighbor == "NN":
                ham, hamu, hamd = HamNN(
                    alattice, p, coeff * qmax, kx, ky, irreducibleMatrix
                )
            elif modelNeighbor == "TNN":
                ham, hamu, ham = HamTNN(
                    alattice, p, coeff * qmax, kx, ky, irreducibleMatrix
                )

            eigenvals = LA.eigvalsh(ham)
            vals_up = LA.eigvalsh(hamu)
            vals_down = LA.eigvalsh(hamd)

            valuesBandLambda = {}
            for i in range(numberWave):
                valuesBandLambda[f"E_2q{i}"] = vals_up[coeff * qmax + i]

            row = {
                # "eta": eta,
                "B_values": round(B, 4),
            }
            for i in range(numberWave):
                row[f"E2q{i}"] = valuesBandLambda[f"E_2q{i}"]
            writer.writerow(row)

    return None
