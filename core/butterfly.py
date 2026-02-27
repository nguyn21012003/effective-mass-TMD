import csv

import numpy as np
from numpy import linalg as LA
from numpy import sqrt
from tqdm import tqdm

from core.HamTMD import HamNN
from core.HamTMDNN import HamTNN


def butterfly(dataInit, irreducibleMatrix):
    p = dataInit["p"]
    coeff = dataInit["coeff"]
    print(coeff)
    numberWave = dataInit["numberWaveFunction"]  # so ham song can khao sat
    modelNeighbor = dataInit["modelNeighbor"]
    k = dataInit["kpoint"]
    qmax = dataInit["qmax"]
    alattice = dataInit["alattice"] * 1e-10  # angstrogn sang m

    h = 6.62607007e-34  # kg m**2 / s**2
    charge = 1.602176621e-19  # Coulomb
    phi0 = h / charge
    S = sqrt(3) * alattice**2 / 2
    lambd = 0.073

    qrange = [
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
    print(qrange)

    with open("cb_up.dat", "w", newline="") as cb_up, open(
        "cb_dn.dat", "w", newline=""
    ) as cb_dn, open("cb_line_up.dat", "w", newline="") as cb_lup, open(
        "cb_line_dn.dat", "w", newline=""
    ) as cb_ldn, open(
        "vb_up.dat", "w", newline=""
    ) as vb_up, open(
        "vb_dn.dat", "w", newline=""
    ) as vb_dn, open(
        "vb_line_up.dat", "w", newline=""
    ) as vb_lup, open(
        "vb_line_dn.dat", "w", newline=""
    ) as vb_ldn:

        header_scatter = ["B", "E"]
        header_cb_line = ["B"] + [f"E{i}" for i in range(numberWave)]
        header_vb_line = ["B"]
        header_vb_line.extend(
            [
                "EKpu1",
                "EKpu2",
                "EKpu3",
                "EKpu4",
                "EKpd1",
                "EKpd2",
                "EKpd3",
                "EKpd4",
                "EKmu1",
                "EKmu2",
                "EKmu3",
                "EKmu4",
                "EKmd1",
                "EKmd2",
                "EKmd3",
                "EKmd4",
                "EGu1",
                "EGu2",
                "EGu3",
                "EGu4",
                "EGd1",
                "EGd2",
                "EGd3",
                "EGd4",
            ]
        )
        writers = {
            "cb_up": csv.DictWriter(cb_up, fieldnames=header_scatter),
            "cb_dn": csv.DictWriter(cb_dn, fieldnames=header_scatter),
            "cb_lup": csv.DictWriter(cb_lup, fieldnames=header_cb_line),
            "cb_ldn": csv.DictWriter(cb_ldn, fieldnames=header_cb_line),
            "vb_up": csv.DictWriter(vb_up, fieldnames=header_scatter),
            "vb_dn": csv.DictWriter(vb_dn, fieldnames=header_scatter),
            "vb_lup": csv.DictWriter(vb_lup, fieldnames=header_vb_line),
            "vb_ldn": csv.DictWriter(vb_ldn, fieldnames=header_vb_line),
        }

        for w in writers.values():
            w.writeheader()

        for qmax in tqdm(qrange, desc="Solve Hamiltonian", ascii=" #", colour="blue"):
            if np.gcd(p, qmax) != 1:
                continue

            eta = p / qmax
            B = eta * phi0 / S

            if modelNeighbor == "NN":
                ham, hamu, hamd = HamNN(
                    alattice, p, coeff * qmax, k, lambd, irreducibleMatrix
                )
            elif modelNeighbor == "TNN":
                ham, hamu, hamd = HamTNN(
                    alattice, p, coeff * qmax, k, lambd, irreducibleMatrix
                )

            # eigenvals = LA.eigvalsh(ham)
            vals = LA.eigvalsh(ham)
            vals_u = LA.eigvalsh(hamu)
            vals_d = LA.eigvalsh(hamd)

            E1 = vals[2 * qmax + 0]
            E2 = vals[2 * qmax + 2]
            diffE_con1 = E2 - E1

            E1 = vals[2 * qmax + 6]
            E2 = vals[2 * qmax + 10]
            diffE_con2 = E2 - E1

            print(abs(diffE_con2 - diffE_con1))

            E1 = vals[2 * qmax - 1]
            E2 = vals[2 * qmax - 7]
            diffE_con1 = E2 - E1

            E1 = vals[2 * qmax - 17]
            E2 = vals[2 * qmax - 27]
            diffE_con2 = E2 - E1
            print(abs(diffE_con2 - diffE_con1))

    return None
