import csv
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from numpy import pi, sqrt
from tqdm import tqdm
import torch

from file_python.HamTMD import Hamiltonian as HamNN
from file_python.HamTMDNN import HamTNN
from file_python.irrMatrix import IR, IRNN, IRTNN
from file_python.irrMatrixTransform import IR as IR_tran
from file_python.irrMatrixTransform import IRNN as IRNN_tran
from file_python.irrMatrixTransform import IRTNN as IRTNN_tran
from file_python.parameters import paraNN, paraTNN
from file_python.plotbyGNU import PlotMatrixGNU


def waveFunction(choice: int, qmax: int, kpoint: str, fileData: dict, model: dict):
    fileSave = fileData[f"file{kpoint}"]

    modelParameters = model["modelParameters"]
    modelNeighbor = model["modelNeighbor"]
    functionMapping = {"TNN": paraTNN, "NN": paraNN}

    data = functionMapping[modelNeighbor](choice, modelParameters)
    alattice = data["alattice"]

    E0, h1, h2, h3, h4, h5, h6 = IR(data)

    v1 = v2 = v3 = v4 = v5 = v6 = 0
    o1 = o2 = o3 = o4 = o5 = o6 = 0
    if modelNeighbor == "TNN":
        v1, v2, v3, v4, v5, v6 = IRNN(data)
        o1, o2, o3, o4, o5, o6 = IRTNN(data)

    # if modelNeighbor == "TNN":

    irreducibleMatrix = {
        "NN": [E0, h1, h2, h3, h4, h5, h6],
        "NNN": [v1, v2, v3, v4, v5, v6],
        "TNN": [o1, o2, o3, o4, o5, o6],
    }

    p = 1
    coeff = 2
    kpoints = {
        "G": [0, 0],
        "K1": [4 * pi / (3 * alattice), 0],
        "K2": [-4 * pi / (3 * alattice), 0],
        "M": [pi / (alattice), pi / (sqrt(3) * alattice)],
    }
    dataArr = {
        "d0_Lambda2q": [],
        "d1_Lambda2q": [],
        "d2_Lambda2q": [],
        "d0_Lambda2q1": [],
        "d1_Lambda2q1": [],
        "d2_Lambda2q1": [],
        "PositionAtoms": [],
    }

    kx = kpoints[kpoint][0]
    ky = kpoints[kpoint][1]

    psi_band2q_d0 = np.zeros(coeff * qmax, dtype=complex)
    psi_band2q_d1 = np.zeros(coeff * qmax, dtype=complex)
    psi_band2q_d2 = np.zeros(coeff * qmax, dtype=complex)

    psi_band2q1_d0 = np.zeros(coeff * qmax, dtype=complex)
    psi_band2q1_d1 = np.zeros(coeff * qmax, dtype=complex)
    psi_band2q1_d2 = np.zeros(coeff * qmax, dtype=complex)

    iArr = np.arange(coeff * qmax)
    Ham = None

    if modelNeighbor == "NN":
        Ham = HamNN(alattice, p, coeff * qmax, kx, ky, irreducibleMatrix)
    elif modelNeighbor == "TNN":
        Ham = HamTNN(alattice, p, coeff * qmax, kx, ky, irreducibleMatrix)

    if np.gcd(p, qmax) == 1:

        # HamNewBasis = Wfull @ Ham @ np.conjugate(Wfull).T

        eigenvals, eigenvecs = LA.eigh(Ham)

        band2q_d0 = eigenvecs[:, coeff * qmax + 2]
        band2q_d1 = eigenvecs[:, coeff * qmax + 2]
        band2q_d2 = eigenvecs[:, coeff * qmax + 2]

        band2q1_d0 = eigenvecs[:, coeff * qmax + 3]
        band2q1_d1 = eigenvecs[:, coeff * qmax + 3]
        band2q1_d2 = eigenvecs[:, coeff * qmax + 3]

        for i in range(coeff * qmax):
            psi_band2q_d0[i] += band2q_d0[0 * coeff * qmax + i]
            psi_band2q_d1[i] += band2q_d1[1 * coeff * qmax + i]
            psi_band2q_d2[i] += band2q_d2[2 * coeff * qmax + i]

            psi_band2q1_d0[i] += band2q1_d0[0 * coeff * qmax + i]
            psi_band2q1_d1[i] += band2q1_d1[1 * coeff * qmax + i]
            psi_band2q1_d2[i] += band2q1_d2[2 * coeff * qmax + i]

            dataArr["PositionAtoms"].append(iArr[i])

        absPsi_band2q_d0 = np.abs(psi_band2q_d0) ** 2
        absPsi_band2q_d1 = np.abs(psi_band2q_d1) ** 2
        absPsi_band2q_d2 = np.abs(psi_band2q_d2) ** 2

        absPsi_band2q1_d0 = np.abs(psi_band2q1_d0) ** 2
        absPsi_band2q1_d1 = np.abs(psi_band2q1_d1) ** 2
        absPsi_band2q1_d2 = np.abs(psi_band2q1_d2) ** 2

        dataArr["d0_Lambda2q"].append(absPsi_band2q_d0)
        dataArr["d1_Lambda2q"].append(absPsi_band2q_d1)
        dataArr["d2_Lambda2q"].append(absPsi_band2q_d2)

        dataArr["d0_Lambda2q1"].append(absPsi_band2q1_d0)
        dataArr["d1_Lambda2q1"].append(absPsi_band2q1_d1)
        dataArr["d2_Lambda2q1"].append(absPsi_band2q1_d2)

        with open(fileSave, "w", newline="") as writefile:
            header = [
                "x",
                "d0_Lambda2q",
                "d1_Lambda2q",
                "d2_Lambda2q",
                "d0_Lambda2q1",
                "d1_Lambda2q1",
                "d2_Lambda2q1",
            ]

            writer = csv.DictWriter(writefile, fieldnames=header, delimiter=",")
            writer.writeheader()
            iPosition = dataArr["PositionAtoms"]
            band_Lambda2q_d0 = dataArr["d0_Lambda2q"][0]
            band_Lambda2q_d1 = dataArr["d1_Lambda2q"][0]
            band_Lambda2q_d2 = dataArr["d2_Lambda2q"][0]

            band_Lambda2q1_d0 = dataArr["d0_Lambda2q1"][0]
            band_Lambda2q1_d1 = dataArr["d1_Lambda2q1"][0]
            band_Lambda2q1_d2 = dataArr["d2_Lambda2q1"][0]

            for q in range(coeff * qmax):
                row = {
                    "x": iPosition[q],
                    "d0_Lambda2q": band_Lambda2q_d0[q],
                    "d1_Lambda2q": band_Lambda2q_d1[q],
                    "d2_Lambda2q": band_Lambda2q_d2[q],
                    "d0_Lambda2q1": band_Lambda2q1_d0[q],
                    "d1_Lambda2q1": band_Lambda2q1_d1[q],
                    "d2_Lambda2q1": band_Lambda2q1_d2[q],
                }
                writer.writerow(row)
        # saveMatrix(Ham, fileMatrix)
        # plotMatrix(H)

    return None


def butterfly(band, choice: int, qmax: int, kpoint: str, fileData, model: dict):

    # fileSaveButterfly = fileData[f"fileButterfly_{kpoint}"]
    # fileMatrix = fileData["fileMatrix"]
    modelParameters = model["modelParameters"]
    modelNeighbor = model["modelNeighbor"]

    functionMapping = {"TNN": paraTNN, "NN": paraNN}
    data = functionMapping[modelNeighbor](choice, modelParameters)

    matt = data["material"]
    alattice = data["alattice"] * 1e-10
    E0, h1, h2, h3, h4, h5, h6 = IR_tran(data)

    v1 = v2 = v3 = v4 = v5 = v6 = 0
    o1 = o2 = o3 = o4 = o5 = o6 = 0

    h = 6.62607007e-34
    hbar = h / (2 * pi)
    charge = 1.602176621e-19
    phi0 = h / charge
    S = sqrt(3) * alattice**2 / 2
    m_e = 9.10938356e-31

    if modelNeighbor == "TNN":
        v1, v2, v3, v4, v5, v6 = IRNN_tran(data)
        o1, o2, o3, o4, o5, o6 = IRTNN_tran(data)

    irreducibleMatrix = {
        "NN": [E0, h1, h2, h3, h4, h5, h6],
        "NNN": [v1, v2, v3, v4, v5, v6],
        "TNN": [o1, o2, o3, o4, o5, o6],
    }

    kpoints = {
        "G": [0, 0],
        "K1": [4 * pi / (3 * alattice), 0],
        "K2": [-4 * pi / (3 * alattice), 0],
        "M": [pi / (alattice), pi / (sqrt(3) * alattice)],
    }

    kx = kpoints[kpoint][0]
    ky = kpoints[kpoint][1]
    Ham = None

    with open(fileData, "w", newline="") as writefile:
        header = [
            "eta",
            "B_values",
            "evalues",
            "E_level1",
            "E_level2",
            # "E_level3",
            "m*_v",
            "m*_c",
            "ω_c",
            "ω_v",
        ]

        writer = csv.DictWriter(writefile, fieldnames=header, delimiter=",")
        writer.writeheader()
        for p in tqdm(range(1, qmax + 1), ascii=" #", desc=f"{matt}"):
            if np.gcd(p, qmax) != 1:
                continue
            eta = p / (qmax)
            coeff = 1

            B = eta * phi0 / S

            if modelNeighbor == "NN":
                Ham = HamNN(alattice, p, coeff * qmax, kx, ky, irreducibleMatrix)
            elif modelNeighbor == "TNN":
                Ham = HamTNN(alattice, p, coeff * qmax, kx, ky, irreducibleMatrix)

            # HamNewBasis = Wfull @ Ham @ np.conj(Wfull).T

            eigenvals = LA.eigvalsh(Ham)
            E_2q = eigenvals[coeff * qmax + 8]
            E_2q1 = eigenvals[coeff * qmax + 9]
            E_2q2 = eigenvals[coeff * qmax - 17]

            En_valence = eigenvals[coeff * qmax - 17] + eigenvals[coeff * qmax - 18]
            En1_valence = eigenvals[coeff * qmax - 27] + eigenvals[coeff * qmax - 28]

            En_conduction = eigenvals[coeff * qmax]
            En1_conduction = eigenvals[coeff * qmax + 11]

            omega_valence = abs((En1_valence - En_valence) * charge / hbar)
            omega_conduction = (En1_conduction - En_conduction) * charge / hbar
            m_eff_v = charge * B / omega_valence
            m_eff_c = charge * B / omega_conduction

            m_ratio_v = m_eff_v / m_e
            m_ratio_c = m_eff_c / m_e

            # print("\n", En, "\n")
            # print(En1, "\n")
            # print(m_eff, "\n")
            # print("\n", m_ratio_v, "\n")
            # print(m_ratio_c, "\n")
            # print(B, "\n")
            for i in range(coeff * band * qmax):
                writer.writerow(
                    {
                        "eta": eta,
                        "B_values": B,
                        "evalues": eigenvals[i],
                        "E_level1": E_2q,
                        "E_level2": E_2q1,
                        # "E_level3": E_2q2,
                        "m*_v": m_ratio_v,
                        "m*_c": m_ratio_c,
                        "ω_v": omega_valence,
                        "ω_c": omega_conduction,
                    }
                )
            # writefile.write("\n")
            # saveMatrix(Ham, fileMatrix)
            # plotMatrix(H)

    return None


def main():
    qmax = 297
    n_levels = 8
    choice = 0
    bandNumber = 3
    bandSelector = "A"
    modelPara = "GGA"
    modelNeighbor = "TNN"
    model = {"modelParameters": modelPara, "modelNeighbor": modelNeighbor}
    kpoint1 = "G"
    # kpoint2 = "K1"
    # kpoint3 = "K2"
    # kpoint4 = "M"
    # listKpoint = [kpoint1, kpoint2, kpoint3, kpoint4]
    data = paraTNN(choice, model["modelParameters"])
    matt = data["material"]

    time_run = datetime.now().strftime("%a-%m-%d")
    dir = f"./{time_run}/{modelNeighbor}/"
    print("folder direction: ", dir)
    os.makedirs(os.path.dirname(dir), exist_ok=True)

    fileButterflyK1 = f"{dir}{bandNumber}band_Lambda2q_dataHofstadterButterfly_q_{qmax}_{matt}_{modelPara}_{kpoint1}.dat"
    # fileButterflyK2 = f"{dir}{bandNumber}band_Lambda2q_dataHofstadterButterfly_q_{qmax}_{matt}_{modelPara}_{kpoint2}.dat"

    filePlotC_k1 = f"{dir}{bandNumber}band_PlotEigenVectors_q_{qmax}_{matt}_{modelPara}_{kpoint1}_vals_vecs.dat"
    filePlotC_k2 = f"{dir}{bandNumber}band_PlotEigenVectors_2q+1_{qmax}_{matt}_{modelPara}_{kpoint1}_vals_vecs.dat"
    # filePlotC_k3 = f"{dir}{bandNumber}band_PlotEigenVectors_q_{qmax}_{matt}_{modelPara}_{kpoint3}_vals_vecs.dat"
    # filePlotC_k4 = f"{dir}{bandNumber}band_PlotEigenVectors_q_{qmax}_{matt}_{modelPara}_{kpoint4}_vals_vecs.dat"

    fileMatrix = f"{bandNumber}band_Matrix_q_{qmax}{time_run}.dat"
    file_plot_Matrix_Gnu = f"{bandNumber}band_Matrix_q_{qmax}{time_run}_h0.gnuplot"

    filegnu = f"{bandNumber}band_plotHofstadterButterfly_q={qmax}.gnuplot"

    print("file data buttefly K1: ", fileButterflyK1)
    # print("file data buttefly K2: ", fileButterflyK2)
    print("file data: ", filePlotC_k1)
    print("file gnuplot: ", filegnu)
    print("file Matrix: ", fileMatrix)
    print("file Matrix GNU: ", file_plot_Matrix_Gnu)

    fileData = {
        "fileButterfly_K1": fileButterflyK1,
        "fileMatrix": fileMatrix,
        "filePlotMatrix": file_plot_Matrix_Gnu,
        "fileWriteGnu": filegnu,
        "fileG": filePlotC_k1,
    }

    # print(torch.cuda.is_available())
    # butterflyK1 = butterfly(bandNumber, choice, qmax, kpoint1, fileButterflyK1, model)
    # # butterflyK2 = butterfly(bandNumber, choice, qmax, kpoint2, fileData, model)

    dataK1 = waveFunction(choice, qmax, kpoint1, fileData, model)
    # dataK2 = waveFunction(bandNumber, choice, qmax, kpoint2, fileData, model)
    # dataK3 = waveFunction(bandNumber, choice, qmax, kpoint3, fileData, model)
    # dataK4 = waveFunction(bandNumber, choice, qmax, kpoint4, fileData, model)

    # PlotMatrixGNU(fileMatrix, file_plot_Matrix_Gnu)
    # saveFunction(currentProg, "fileData")

    return None


if __name__ == "__main__":
    main()
