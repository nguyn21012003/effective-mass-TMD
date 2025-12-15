import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv


def H(p, q, kx, ky):

    alpha = p / q

    M = np.zeros((q, q), dtype=complex)
    for i in range(0, q):
        M[i, i] = 2 * np.cos(ky - 2 * np.pi * alpha * i)
        if i == q - 1:
            M[i, i - 1] = 1
        elif i == 0:
            M[i, i + 1] = 1
        else:
            M[i, i - 1] = 1
            M[i, i + 1] = 1

    if q == 2:
        M[0, q - 1] = 1 + np.exp(-q * 1.0j * kx)
        M[q - 1, 0] = 1 + np.exp(q * 1.0j * kx)
    else:
        M[0, q - 1] = np.exp(-q * 1.0j * kx)
        M[q - 1, 0] = np.exp(q * 1.0j * kx)

    return M


def plot_butterfly(qmax):
    p = 1
    coeff = 1
    numberWave = 60
    dataArr = {"PositionAtoms": []}
    for i in range(numberWave):
        dataArr[f"lambda2q_band_{i}"] = []
    arrContainer = {}
    iArr = np.arange(coeff * qmax)  # chi so atom thu i, do 3 ham song thi la 3*2*qmax = 6qmax, nhung 1 ham song thi la 2qmax

    for i in range(numberWave):
        #### Tinh cho d_z^2
        arrContainer[f"psi_band2q_d0_{i}"] = np.zeros(coeff * qmax, dtype=complex)
        arrContainer[f"absPsi_band2q_d0_{i}"] = np.zeros(coeff * qmax, dtype=float)

    if np.gcd(p, qmax) == 1:
        _, eigenvectors = np.linalg.eigh(H(p, qmax, kx=0, ky=0))
        print(eigenvectors.size)

        for i in tqdm(range(numberWave), desc="Calc eigenvectors", colour="green"):

            arrContainer[f"psi_band2q_d0_{i}"][:qmax] += eigenvectors[:, i][0 * coeff * qmax : 1 * coeff * qmax]

            arrContainer[f"absPsi_band2q_d0_{i}"] = np.abs(arrContainer[f"psi_band2q_d0_{i}"]) ** 2

        for i in range(coeff * qmax):
            dataArr["PositionAtoms"].append(iArr[i])

        with open("test.dat", "w", newline="") as writefile:
            header = ["x"]
            dArr = ["d0"]
            for d in dArr:
                for i in range(numberWave):
                    header.append(f"{d}_lambda_{i}")

            writer = csv.DictWriter(writefile, fieldnames=header, delimiter=",")
            writer.writeheader()
            iPosition = dataArr["PositionAtoms"]

            for q in tqdm(range(coeff * qmax), desc="Write file", colour="blue"):
                row = {"x": iPosition[q]}
                for i in range(numberWave):
                    row[f"d0_lambda_{i}"] = arrContainer[f"absPsi_band2q_d0_{i}"][q]

                writer.writerow(row)

    return


plot_butterfly(97)
