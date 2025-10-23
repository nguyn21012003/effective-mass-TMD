import csv

import numpy as np
from numpy import linalg as LA
from numpy import pi, sqrt
from tqdm import tqdm

from core.HamTMD import HamNN
from core.HamTMDNN import HamTNN


def calcRadius(dataInit, irreducibleMatrix, fileSave):
    p = dataInit["p"]
    coeff = dataInit["coeff"]
    print(coeff)
    modelNeighbor = dataInit["modelNeighbor"]
    kx, ky = dataInit["kpoint"]
    qmax = dataInit["qmax"]

    alattice = dataInit["alattice"] * 1e-10  # angstrogn sang nm
    h = 6.62607007e-34  # kg m**2 / s**2
    hbar = h / (2 * pi)
    charge = 1.602176621e-19  # Coulomb
    phi0 = h / charge
    S = sqrt(3) * alattice**2 / 2
    m_e = 9.10938356e-31  # kg
    v_f = 4.19e5  # Vận tốc Fermi trong chất rắn, đơn vị là m/s

    Hamiltonian = 0
    B_values = list(range(15, 505, 5))  # đơn vị là Tesla
    print(B_values, "\n")
    qrange = [round(phi0 / (S * B)) for B in B_values]
    print(qrange)

    with open(fileSave, "w", newline="") as writefile:
        header = [
            # "eta",
            "B_values",
            "r_vK1",
            "r_vK2",
            "r_cK1",
            "r_cK2",
        ]
        writer = csv.DictWriter(writefile, fieldnames=header, delimiter=",")
        writer.writeheader()
        for B, qmax in tqdm(
            zip(B_values, qrange),
            ascii=" #",
            total=len(B_values),
            desc=f"Solve Hamiltonian",
            colour="blue",
        ):
            if np.gcd(p, qmax) != 1:
                continue
            eta = p / (qmax)  ## the magnetic ratio require that p and q must be co-prime
            if modelNeighbor == "NN":
                Hamiltonian = HamNN(alattice, p, coeff * qmax, kx, ky, irreducibleMatrix)
            elif modelNeighbor == "TNN":
                Hamiltonian = HamTNN(alattice, p, coeff * qmax, kx, ky, irreducibleMatrix)

            eigenvals = np.sort(LA.eigvalsh(Hamiltonian))

            meff_vK1, meff_cK1, meff_vK2, meff_cK2 = 0, 0, 0, 0
            if qmax in offsetK2:
                off1K1, off2K1, off3K1, off4K1 = offsetK1[qmax]
                off1K2, off2K2, off3K2, off4K2 = offsetK2[qmax]
                off1K1_con, off2K1_con, off3K1_con, off4K1_con = offsetK1_con[qmax]
                off1K2_con, off2K2_con, off3K2_con, off4K2_con = offsetK2_con[qmax]
                off1G, off2G, off3G, off4G = offsetG[qmax]

                En_valence_K1 = eigenvals[coeff * qmax - off1K1]  ### Only K-point at Landau level n = 0
                En1_valence_K1 = eigenvals[coeff * qmax - off2K1]  ### Only K-point at Landau level n = 1

                En_valence_K2 = eigenvals[coeff * qmax - off1K2]  ### Only K'-point at Landau level n = 1
                En1_valence_K2 = eigenvals[coeff * qmax - off2K2]  ### Only K'-point at Landau level n = 1

                En_conduction_K1 = eigenvals[coeff * qmax + off1K1_con]
                En1_conduction_K1 = eigenvals[coeff * qmax + off2K1_con]

                En_conduction_K2 = eigenvals[coeff * qmax + off1K2_con]
                En1_conduction_K2 = eigenvals[coeff * qmax + off2K2_con]

                ################# K
                omega_vK1 = abs((En_valence_K1 - En1_valence_K1) * charge / hbar)
                meff_vK1 = charge * B / omega_vK1
                ################# K'
                omega_vK2 = abs((En_valence_K2 - En1_valence_K2) * charge / hbar)
                meff_vK2 = charge * B / omega_vK2

                omega_cK1 = (En1_conduction_K1 - En_conduction_K1) * charge / hbar
                meff_cK1 = charge * B / omega_cK1

                omega_cK2 = (En1_conduction_K2 - En_conduction_K2) * charge / hbar
                meff_cK2 = charge * B / omega_cK2
            row = {
                "B_values": B,
            }

            row["r_vK1"] = (meff_vK1 * v_f / (charge * B)) / alattice
            row["r_vK2"] = (meff_vK2 * v_f / (charge * B)) / alattice

            row["r_cK1"] = (meff_cK2 * v_f / (charge * B)) / alattice
            row["r_cK2"] = (meff_cK1 * v_f / (charge * B)) / alattice

            writer.writerow(row)

    return None
