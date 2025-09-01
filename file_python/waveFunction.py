import numpy as np
from numpy import linalg as LA
from tqdm import tqdm
import csv

from file_python.HamTMD import HamNN
from file_python.HamTMDNN import HamTNN


def waveFunction(dataInit, irreducibleMatrix, fileSave):
    ##### chi so dau vao
    p = dataInit["p"]
    coeff = dataInit["coeff`"]
    numberWave = dataInit["numberWaveFunction"]  # so ham song can khao sat
    modelNeighbor = dataInit["modelNeighbor"]
    alattice = dataInit["alattice"]
    kx, ky = dataInit["kpoint"]
    qmax = dataInit["qmax"]

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
