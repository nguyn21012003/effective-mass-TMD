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
    B_values = list(range(15, 505, 5))  # đơn vị là Tesla
    print(B_values, "\n")
    qrange = [round(phi0 / (S * B)) for B in B_values]
    print(qrange)

    with open(fileSave, "w", newline="") as writefile:
        header = [
            # "eta",
            "B_values",
            # "evalues",
            # "m_hK1",
            # "m_hK2",
            # "m_eK1",
            # "m_eK2",
            "r_vK1",
            "r_vK2",
            "r_cK1",
            "r_cK2",
        ]

        # header.extend(["EKp1", "EKp2", "EKp3", "EKp4", "EKm1", "EKm2", "EKm3", "EKm4", "EG1", "EG2", "EG3", "EG4"])
        # for i in range(numberWave):
        #    header.append(f"E2q{i}")
        writer = csv.DictWriter(writefile, fieldnames=header, delimiter=",")
        writer.writeheader()
        # for qmax in qrange:
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
            # B = eta * phi0 / S  ## the actually B which are taken from eta

            if modelNeighbor == "NN":
                Hamiltonian = HamNN(alattice, p, coeff * qmax, kx, ky, irreducibleMatrix)
            elif modelNeighbor == "TNN":
                Hamiltonian = HamTNN(alattice, p, coeff * qmax, kx, ky, irreducibleMatrix)

            eigenvals = np.sort(LA.eigvalsh(Hamiltonian))

            # valuesBandLambda = {}
            # for i in range(numberWave):
            #    valuesBandLambda[f"E_2q{i}"] = eigenvals[coeff * qmax - i - 1]

            EKp1 = EKp2 = EKp3 = EKp4 = 0
            EKm1 = EKm2 = EKm3 = EKm4 = 0
            EKp1_con = EKp2_con = EKp3_con = EKp4_con = 0
            EKm1_con = EKm2_con = EKm3_con = EKm4_con = 0
            EG1 = EG2 = EG3 = EG4 = 0
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

                ################### Plot but and wave for convenience
                EKp1 = eigenvals[coeff * qmax - off1K1]
                EKp2 = eigenvals[coeff * qmax - off2K1]
                EKp3 = eigenvals[coeff * qmax - off3K1]
                EKp4 = eigenvals[coeff * qmax - off4K1]

                EKm1 = eigenvals[coeff * qmax - off1K2]
                EKm2 = eigenvals[coeff * qmax - off2K2]
                EKm3 = eigenvals[coeff * qmax - off3K2]
                EKm4 = eigenvals[coeff * qmax - off4K2]

                EG1 = eigenvals[coeff * qmax - off1G]
                EG2 = eigenvals[coeff * qmax - off2G]
                EG3 = eigenvals[coeff * qmax - off3G]
                EG4 = eigenvals[coeff * qmax - off4G]

            m_ratio_vK1 = meff_vK1 / m_e
            m_ratio_cK1 = meff_cK1 / m_e

            m_ratio_vK2 = meff_vK2 / m_e
            m_ratio_cK2 = meff_cK2 / m_e
            # print(m_ratio_v)
            # print(m_ratio_c, "\n")
            row = {
                # "eta": eta,
                "B_values": B,
            }
            # row["EKp1"] = EKp1
            # row["EKp2"] = EKp2
            # row["EKp3"] = EKp3
            # row["EKp4"] = EKp4

            # row["EKm1"] = EKm1
            # row["EKm2"] = EKm2
            # row["EKm3"] = EKm3
            # row["EKm4"] = EKm4

            # row["EG1"] = EG1
            # row["EG2"] = EG2
            # row["EG3"] = EG3
            # row["EG4"] = EG4

            row["m_hK1"] = m_ratio_vK1
            row["m_hK2"] = m_ratio_vK2

            row["m_eK1"] = m_ratio_cK2
            row["m_eK2"] = m_ratio_cK1

            # for i in range(numberWave):
            #    row[f"E2q{i}"] = valuesBandLambda[f"E_2q{i}"]
            # for i in range(coeff * 3 * qmax):
            #    row["evalues"] = eigenvals[i]
            writer.writerow(row)

    return None
