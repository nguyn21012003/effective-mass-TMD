import csv

import numpy as np
from numpy import linalg as LA
from tqdm import tqdm

from core.HamTMD import HamNN
from core.HamTMDNN import HamTNN


def waveFunction(dataInit, irreducibleMatrix, fileSave):
    ##### chi so dau vao
    p = dataInit["p"]
    fileWave = fileSave["wave"]
    fileUp = fileSave["waveUp"]
    fileDown = fileSave["waveDown"]
    coeff = dataInit["coeff"]
    numberWave = dataInit["numberWaveFunction"]  # so ham song can khao sat
    modelNeighbor = dataInit["modelNeighbor"]
    alattice = dataInit["alattice"]
    k = dataInit["kpoint"]
    qmax = dataInit["qmax"]
    lambd = dataInit["lambda"]
    ##### tao array de luu ket qua
    dataArr = {"PositionAtoms": []}
    for i in range(numberWave):
        dataArr[f"lambda2q_band_{i}"] = []

    #### tinh toan chi tiet
    #### bat dau bang viec khoi tao mang de chua psi va abs(psi)**2
    waveArr = {}
    waveArrUp = {}
    waveArrDown = {}
    iArr = np.arange(
        coeff * qmax
    )  # chi so atom thu i, do 3 ham song thi la 3*2*qmax = 6qmax, nhung 1 ham song thi la 2qmax
    for i in range(numberWave):
        #### Tinh cho d_z^2
        waveArr[f"psi_band2q_d0_{i}"] = np.zeros(coeff * qmax, dtype=complex)
        waveArr[f"absPsi_band2q_d0_{i}"] = np.zeros(coeff * qmax, dtype=float)
        #### Tinh cho d_-2
        waveArr[f"psi_band2q_d1_{i}"] = np.zeros(coeff * qmax, dtype=complex)
        waveArr[f"absPsi_band2q_d1_{i}"] = np.zeros(coeff * qmax, dtype=float)
        #### Tinh cho d_2
        waveArr[f"psi_band2q_d2_{i}"] = np.zeros(coeff * qmax, dtype=complex)
        waveArr[f"absPsi_band2q_d2_{i}"] = np.zeros(coeff * qmax, dtype=float)

        ############ Spin up
        waveArrUp[f"psi_band2q_d0_{i}"] = np.zeros(coeff * qmax, dtype=complex)
        waveArrUp[f"absPsi_band2q_d0_{i}"] = np.zeros(coeff * qmax, dtype=float)
        #### Tinh cho d_-2
        waveArrUp[f"psi_band2q_d1_{i}"] = np.zeros(coeff * qmax, dtype=complex)
        waveArrUp[f"absPsi_band2q_d1_{i}"] = np.zeros(coeff * qmax, dtype=float)
        #### Tinh cho d_2
        waveArrUp[f"psi_band2q_d2_{i}"] = np.zeros(coeff * qmax, dtype=complex)
        waveArrUp[f"absPsi_band2q_d2_{i}"] = np.zeros(coeff * qmax, dtype=float)

        ############ Spin down
        waveArrDown[f"psi_band2q_d0_{i}"] = np.zeros(coeff * qmax, dtype=complex)
        waveArrDown[f"absPsi_band2q_d0_{i}"] = np.zeros(coeff * qmax, dtype=float)
        #### Tinh cho d_-2
        waveArrDown[f"psi_band2q_d1_{i}"] = np.zeros(coeff * qmax, dtype=complex)
        waveArrDown[f"absPsi_band2q_d1_{i}"] = np.zeros(coeff * qmax, dtype=float)
        #### Tinh cho d_2
        waveArrDown[f"psi_band2q_d2_{i}"] = np.zeros(coeff * qmax, dtype=complex)
        waveArrDown[f"absPsi_band2q_d2_{i}"] = np.zeros(coeff * qmax, dtype=float)
    ham, hamu, hamd = None, None, None
    if modelNeighbor == "NN":
        ham, hamu, hamd = HamNN(alattice, p, coeff * qmax, k, lambd, irreducibleMatrix)
    elif modelNeighbor == "TNN":
        ham, hamu, hamd = HamTNN(alattice, p, coeff * qmax, k, lambd, irreducibleMatrix)

    if np.gcd(p, qmax) == 1:
        # eigenvals, eigenvecs = LA.eigh(ham)
        _, vecs_u = LA.eigh(hamu)
        _, vecs_d = LA.eigh(hamd)
        # print(eigenvecs.shape)
        for i in tqdm(range(numberWave), desc="Calc eigenvectors", colour="green"):
            # print(i)
            #### i la chi so 2q + i trong so band 6q
            # waveArr[f"psi_band2q_d0_{i}"][: coeff * qmax] += eigenvecs[
            #     :, coeff * qmax + i
            # ][0 * coeff * qmax : 1 * coeff * qmax]
            # waveArr[f"psi_band2q_d1_{i}"][: coeff * qmax] += eigenvecs[
            #     :, coeff * qmax + i
            # ][1 * coeff * qmax : 2 * coeff * qmax]
            # waveArr[f"psi_band2q_d2_{i}"][: coeff * qmax] += eigenvecs[
            #     :, coeff * qmax + i
            # ][2 * coeff * qmax : 3 * coeff * qmax]
            #
            # waveArr[f"absPsi_band2q_d0_{i}"] = (
            #     np.abs(waveArr[f"psi_band2q_d0_{i}"]) ** 2
            # )
            # waveArr[f"absPsi_band2q_d1_{i}"] = (
            #     np.abs(waveArr[f"psi_band2q_d1_{i}"]) ** 2
            # )
            # waveArr[f"absPsi_band2q_d2_{i}"] = (
            #     np.abs(waveArr[f"psi_band2q_d2_{i}"]) ** 2
            # )
            ################ Spin up
            waveArrUp[f"psi_band2q_d0_{i}"][: coeff * qmax] = vecs_u[
                :, coeff * qmax + i
            ][0 * coeff * qmax : 1 * coeff * qmax]
            waveArrUp[f"psi_band2q_d1_{i}"][: coeff * qmax] = vecs_u[
                :, coeff * qmax + i
            ][1 * coeff * qmax : 2 * coeff * qmax]
            waveArrUp[f"psi_band2q_d2_{i}"][: coeff * qmax] = vecs_u[
                :, coeff * qmax + i
            ][2 * coeff * qmax : 3 * coeff * qmax]
            ################ abs Spin up
            waveArrUp[f"absPsi_band2q_d0_{i}"] = (
                np.abs(waveArrUp[f"psi_band2q_d0_{i}"]) ** 2
            )
            waveArrUp[f"absPsi_band2q_d1_{i}"] = (
                np.abs(waveArrUp[f"psi_band2q_d1_{i}"]) ** 2
            )
            waveArrUp[f"absPsi_band2q_d2_{i}"] = (
                np.abs(waveArrUp[f"psi_band2q_d2_{i}"]) ** 2
            )
            ################ Spin down
            waveArrDown[f"psi_band2q_d0_{i}"][: coeff * qmax] = vecs_d[
                :, coeff * qmax + i
            ][0 * coeff * qmax : 1 * coeff * qmax]
            waveArrDown[f"psi_band2q_d1_{i}"][: coeff * qmax] = vecs_d[
                :, coeff * qmax + i
            ][1 * coeff * qmax : 2 * coeff * qmax]
            waveArrDown[f"psi_band2q_d2_{i}"][: coeff * qmax] = vecs_d[
                :, coeff * qmax + i
            ][2 * coeff * qmax : 3 * coeff * qmax]
            ################ abs Spin down
            waveArrDown[f"absPsi_band2q_d0_{i}"] = (
                np.abs(waveArrDown[f"psi_band2q_d0_{i}"]) ** 2
            )
            waveArrDown[f"absPsi_band2q_d1_{i}"] = (
                np.abs(waveArrDown[f"psi_band2q_d1_{i}"]) ** 2
            )
            waveArrDown[f"absPsi_band2q_d2_{i}"] = (
                np.abs(waveArrDown[f"psi_band2q_d2_{i}"]) ** 2
            )

        for i in range(coeff * qmax):
            dataArr["PositionAtoms"].append(iArr[i])

        with open(fileWave, "w", newline="") as f_wave, open(
            fileUp, "w", newline=""
        ) as f_up, open(fileDown, "w", newline="") as f_down:
            header = ["x"]
            orbitals = ["d0", "d1", "d2"]
            for d in orbitals:
                for i in range(numberWave):
                    header.append(f"{d}_lambda_{i}")

            writer_wave = csv.DictWriter(f_wave, fieldnames=header)
            writer_up = csv.DictWriter(f_up, fieldnames=header)
            writer_down = csv.DictWriter(f_down, fieldnames=header)

            writer_wave.writeheader()
            writer_up.writeheader()
            writer_down.writeheader()

            iPosition = dataArr["PositionAtoms"]
            for q in range(coeff * qmax):
                row_wave = {"x": iPosition[q]}
                row_up = {"x": iPosition[q]}
                row_down = {"x": iPosition[q]}
                for i in range(numberWave):
                    row_wave[f"d0_lambda_{i}"] = waveArr[f"absPsi_band2q_d0_{i}"][q]
                    row_wave[f"d1_lambda_{i}"] = waveArr[f"absPsi_band2q_d1_{i}"][q]
                    row_wave[f"d2_lambda_{i}"] = waveArr[f"absPsi_band2q_d2_{i}"][q]

                    row_up[f"d0_lambda_{i}"] = waveArrUp[f"absPsi_band2q_d0_{i}"][q]
                    row_up[f"d1_lambda_{i}"] = waveArrUp[f"absPsi_band2q_d1_{i}"][q]
                    row_up[f"d2_lambda_{i}"] = waveArrUp[f"absPsi_band2q_d2_{i}"][q]

                    row_down[f"d0_lambda_{i}"] = waveArrDown[f"absPsi_band2q_d0_{i}"][q]
                    row_down[f"d1_lambda_{i}"] = waveArrDown[f"absPsi_band2q_d1_{i}"][q]
                    row_down[f"d2_lambda_{i}"] = waveArrDown[f"absPsi_band2q_d2_{i}"][q]

                writer_wave.writerow(row_wave)
                writer_up.writerow(row_up)
                writer_down.writerow(row_down)

    elif np.gcd(p, qmax) != 1:  # check coprime
        print("p,q pairs not co-prime!")

    return None
