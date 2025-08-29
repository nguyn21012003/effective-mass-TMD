import csv
import os
from datetime import datetime
from time import time

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from numpy import pi, sqrt
from tqdm import tqdm

from file_python.HamTMD import Hamiltonian as HamNN
from file_python.HamTMDNN import HamTNN
from file_python.irrMatrix import IR, IRNN, IRTNN
from file_python.irrMatrixTransform import IR as IR_tran
from file_python.irrMatrixTransform import IRNN as IRNN_tran
from file_python.irrMatrixTransform import IRTNN as IRTNN_tran
from file_python.parameters import paraNN, paraTNN


def waveFunction(choice: int, qmax: int, kpoint: str, fileData: str, model: dict):
    ##### chi so dau vao
    p = 1
    coeff = 2
    numberWave = 40  # so ham song can khao sat
    fileSave = fileData
    modelParameters = model["modelParameters"]
    modelNeighbor = model["modelNeighbor"]
    functionMapping = {"TNN": paraTNN, "NN": paraNN}
    data = functionMapping[modelNeighbor](choice, modelParameters)
    alattice = data["alattice"]
    E0, h1, h2, h3, h4, h5, h6 = IR_tran(data)
    v1 = v2 = v3 = v4 = v5 = v6 = 0
    o1 = o2 = o3 = o4 = o5 = o6 = 0
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
    ##### tao array de luu ket qua
    dataArr = {"PositionAtoms": []}
    for i in range(numberWave + 1):
        dataArr[f"lambda2q_band_{i}"] = []

    #### tinh toan chi tiet
    #### bat dau bang viec khoi tao mang de chua psi va abs(psi)**2
    arrContainer = {}
    iArr = np.arange(coeff * qmax)  # chi so atom thu i, do 3 ham song thi la 3*2*qmax = 6qmax, nhung 1 ham song thi la 2qmax
    for i in range(numberWave + 1):
        #### Tinh cho d_z^2
        arrContainer[f"psi_band2q_d0_{i}"] = np.zeros(coeff * qmax, dtype=complex)
        arrContainer[f"absPsi_band2q_d0_{i}"] = np.zeros(coeff * qmax, dtype=float)
        #### Tinh cho d_-2
        arrContainer[f"psi_band2q_d1_{i}"] = np.zeros(coeff * qmax, dtype=complex)
        arrContainer[f"absPsi_band2q_d1_{i}"] = np.zeros(coeff * qmax, dtype=float)
        #### Tinh cho d_2
        arrContainer[f"psi_band2q_d2_{i}"] = np.zeros(coeff * qmax, dtype=complex)
        arrContainer[f"absPsi_band2q_d1_{i}"] = np.zeros(coeff * qmax, dtype=float)
        ##### Unused
        arrContainer[f"psi_band2q1_d0_{i}"] = np.zeros(coeff * qmax, dtype=complex)
        arrContainer[f"psi_band2q1_d1_{i}"] = np.zeros(coeff * qmax, dtype=complex)
        arrContainer[f"psi_band2q1_d2_{i}"] = np.zeros(coeff * qmax, dtype=complex)

    Hamiltonian = None

    if modelNeighbor == "NN":
        Hamiltonian = HamNN(alattice, p, coeff * qmax, kx, ky, irreducibleMatrix)
    elif modelNeighbor == "TNN":
        Hamiltonian = HamTNN(alattice, p, coeff * qmax, kx, ky, irreducibleMatrix)

    #### Tracking gia tri rieng theo ham rieng

    if np.gcd(p, qmax) == 1:

        eigenvals, eigenvecs = LA.eigh(Hamiltonian)

        # print(eigenvecs.shape)
        for i in tqdm(range(numberWave + 1), desc="Calc eigenvectors", colour="green"):
            # print(i)
            #### i la chi so 2q + i trong so band 6q
            arrContainer[f"psi_band2q_d0_{i}"][: coeff * qmax] += eigenvecs[:, coeff * qmax - i - 1][0 * coeff * qmax : 1 * coeff * qmax]
            arrContainer[f"psi_band2q_d1_{i}"][: coeff * qmax] += eigenvecs[:, coeff * qmax - i - 1][1 * coeff * qmax : 2 * coeff * qmax]
            arrContainer[f"psi_band2q_d2_{i}"][: coeff * qmax] += eigenvecs[:, coeff * qmax - i - 1][2 * coeff * qmax : 3 * coeff * qmax]

            arrContainer[f"absPsi_band2q_d0_{i}"] = np.abs(arrContainer[f"psi_band2q_d0_{i}"]) ** 2
            arrContainer[f"absPsi_band2q_d1_{i}"] = np.abs(arrContainer[f"psi_band2q_d1_{i}"]) ** 2
            arrContainer[f"absPsi_band2q_d2_{i}"] = np.abs(arrContainer[f"psi_band2q_d2_{i}"]) ** 2

        for i in range(coeff * qmax):
            dataArr["PositionAtoms"].append(iArr[i])

        with open(fileSave, "w", newline="") as writefile:
            header = ["x"]
            dArr = ["d0", "d1", "d2"]
            for d in dArr:
                for i in range(numberWave + 1):
                    header.append(f"{d}_lambda_{i}")

            writer = csv.DictWriter(writefile, fieldnames=header, delimiter=",")
            writer.writeheader()
            iPosition = dataArr["PositionAtoms"]

            for q in tqdm(range(coeff * qmax), desc="Write file", colour="blue"):
                row = {"x": iPosition[q]}
                for i in range(numberWave + 1):
                    row[f"d0_lambda_{i}"] = arrContainer[f"absPsi_band2q_d0_{i}"][q]
                    row[f"d1_lambda_{i}"] = arrContainer[f"absPsi_band2q_d1_{i}"][q]
                    row[f"d2_lambda_{i}"] = arrContainer[f"absPsi_band2q_d2_{i}"][q]
                writer.writerow(row)

    elif np.gcd(p, qmax) != 1:  # check coprime
        print("p,q pairs not co-prime!")

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
    Hamiltonian = None
    p = 1  # fixed p
    listEigens = {}
    listVectors = {}

    numWave = 80
    coeff = 2

    with open(fileData, "w", newline="") as writefile:
        header = [
            "eta",
            "B_values",
            # "evalues",
            "m*_v",
            "m*_c",
            "ω_c",
            "ω_v",
        ]

        # for i in range(numWave + 1):
        # header.append(f"E2q{i}")

        writer = csv.DictWriter(writefile, fieldnames=header, delimiter=",")
        writer.writeheader()
        for p in tqdm(range(1, qmax + 1), ascii=" #", desc=f"Solve Hamiltonian"):
            if np.gcd(p, qmax) != 1:
                continue
            eta = p / (qmax)

            B = eta * phi0 / S

            if modelNeighbor == "NN":
                Hamiltonian = HamNN(alattice, p, coeff * qmax, kx, ky, irreducibleMatrix)
            elif modelNeighbor == "TNN":
                Hamiltonian = HamTNN(alattice, p, coeff * qmax, kx, ky, irreducibleMatrix)

            eigenvals = LA.eigvalsh(Hamiltonian)

            omega_v = 0
            omega_c = 0
            meff_v = 0
            meff_c = 0
            offset = {3129: (27, 37), 2346: (23, 33), 1877: (21, 31), 1564: (21, 31), 1341: (21, 29), 1173: (19, 29), 1043: (17, 27)}

            # valuesBandLambda = {}
            # for i in range(numWave + 1):
            # valuesBandLambda[f"E_2q{i}"] = eigenvals[coeff * qmax - i]

            if qmax in offset:
                off1, off2 = offset[qmax]
                En_valence = eigenvals[coeff * qmax - off1]
                En1_valence = eigenvals[coeff * qmax - off2]
                omega_v = abs((En1_valence - En_valence) * charge / hbar)
                meff_v = charge * B / omega_v

            En_conduction = eigenvals[coeff * qmax + 4]
            En1_conduction = eigenvals[coeff * qmax + 8]
            omega_c = (En1_conduction - En_conduction) * charge / hbar
            meff_c = charge * B / omega_c

            m_ratio_v = meff_v / m_e
            print(m_ratio_v)
            m_ratio_c = meff_c / m_e
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


def main():
    qmax = 297
    qrange = [2346, 1877, 1564, 1341, 1173, 1043, 939]
    choice = 0
    bandNumber = 3
    modelPara = "GGA"
    modelNeighbor = "NN"
    model = {"modelParameters": modelPara, "modelNeighbor": modelNeighbor}
    kpoint1 = "G"
    data = paraTNN(choice, model["modelParameters"])
    matt = data["material"]

    time_run = datetime.now().strftime("%a-%m-%d")
    dir = f"./{time_run}/{modelNeighbor}/"
    os.makedirs(os.path.dirname(dir), exist_ok=True)

    print("folder direction: ", dir)

    start = time()
    for qmax in tqdm(qrange, ascii=" #", desc=f"Wave function in diff B", colour="magenta"):
        filePlotC_k1 = f"{dir}{bandNumber}band_PlotEigenVectors_q_{qmax}_{matt}_{modelPara}_{kpoint1}_vals_vecs.dat"
        fileButterflyK1 = f"{dir}{bandNumber}band_Lambda2q_dataHofstadterButterfly_q_{qmax}_{matt}_{modelPara}_{kpoint1}.dat"
        # butterflyK1 = butterfly(bandNumber, choice, qmax, kpoint1, fileButterflyK1, model)
        dataK1 = waveFunction(choice, qmax, kpoint1, filePlotC_k1, model)
    end = time()
    print(f"Time calculating wavefunction: {end - start}s")

    return None


if __name__ == "__main__":
    main()
