import csv
import os
from datetime import datetime
from time import time

from eigenshuffle import eigenshuffle_eigh
import matplotlib.pyplot as plt
import numpy as np
import sympy as syp
from numpy import linalg as LA
from numpy import pi, shape, sqrt
from tqdm import tqdm

from file_python.HamTMD import Hamiltonian as HamNN
from file_python.HamTMDNN import HamTNN
from file_python.irrMatrix import IR, IRNN, IRTNN
from file_python.irrMatrixTransform import IR as IR_tran
from file_python.irrMatrixTransform import IRNN as IRNN_tran
from file_python.irrMatrixTransform import IRTNN as IRTNN_tran
from file_python.parameters import paraNN, paraTNN


def waveFunction(choice: int, qmax: int, kpoint: str, fileData: dict, model: dict):
    ##### chi so dau vao
    p = 1
    coeff = 2
    numberWave = 40  # so ham song can khao sat
    fileSave = fileData[f"file{kpoint}"]
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

    Ham = None
    if modelNeighbor == "NN":
        Ham = HamNN(alattice, p, coeff * qmax, kx, ky, irreducibleMatrix)
    elif modelNeighbor == "TNN":
        Ham = HamTNN(alattice, p, coeff * qmax, kx, ky, irreducibleMatrix)
    print(type(Ham))

    #### Tracking gia tri rieng theo ham rieng

    prevVecs = None
    if np.gcd(p, qmax) == 1:

        eigenvals, eigenvecs = LA.eigh(Ham)
        if prevVecs == None:
            eigenvalsSorted = eigenvals
            eigenvecsSorted = eigenvecs
            prevVecs = eigenvecs

        else:
            S = np.abs(np.conjugate(prevVecs).T @ eigenvecs)
            idxMax = np.argmax(S, axis=1)
            eigenvalsSorted = eigenvals[idxMax]
            eigenvecsSorted = eigenvecs[:, idxMax]
            prevVecs = eigenvecsSorted

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
    # p = 1  # fixed p
    qrange = [9368, 4693, 3129, 2346, 1877, 1564, 1341, 1173, 1043, 939, 853, 782, 722, 670, 626, 587, 552, 521, 494, 469]
    numWave = 80
    coeff = 1
    with open(fileData, "w", newline="") as writefile:
        header = [
            "eta",
            "B_values",
            # "evalues",
            # "m*_v",
            # "m*_c",
            # "ω_c",
            # "ω_v",
        ]

        for i in range(numWave + 1):
            header.append(f"E2q{i}")

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

            # eigenvals, eigenvecs = LA.eigh(Hamiltonian)
            eigenvals = LA.eigvalsh(Hamiltonian)
            valuesBandLambda = {}
            for i in range(numWave + 1):
                valuesBandLambda[f"E_2q{i}"] = eigenvals[coeff * qmax - i]

            # En_valence = eigenvals[coeff * qmax - 17] + eigenvals[coeff * qmax - 18]
            # En1_valence = eigenvals[coeff * qmax - 27] + eigenvals[coeff * qmax - 28]

            # En_conduction = eigenvals[coeff * qmax + 4]
            # En1_conduction = eigenvals[coeff * qmax + 8]

            # omega_valence = abs((En1_valence - En_valence) * charge / hbar)
            # omega_conduction = (En1_conduction - En_conduction) * charge / hbar
            # m_eff_v = charge * B / omega_valence
            # m_eff_c = charge * B / omega_conduction

            # m_ratio_v = m_eff_v / m_e
            # m_ratio_c = m_eff_c / m_e
            row = {
                "eta": eta,
                "B_values": B,
            }
            # row["m*_v"] = m_ratio_v
            # row["m*_c"] = m_ratio_c
            # row["ω_v"] = omega_valence
            # row["ω_c"] = omega_conduction
            for i in range(numWave + 1):
                row[f"E2q{i}"] = valuesBandLambda[f"E_2q{i}"]
            # for i in range(coeff * band * qmax):
            #    row["evalues"] = eigenvals[i]
            writer.writerow(row)
            #    writer.writerow(
            #        {
            #            "eta": eta,
            #            "B_values": B,
            #            "evalues": eigenvals[i],
            # "E_level3": E_2q2,
            # "m*_v": m_ratio_v,
            # "m*_c": m_ratio_c,
            # "ω_v": omega_valence,
            # "ω_c": omega_conduction,
            #        }
            #    )
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
    print("file data: ", filePlotC_k1)
    # print("file data buttefly K2: ", fileButterflyK2)
    # print("file gnuplot: ", filegnu)
    # print("file Matrix: ", fileMatrix)
    # print("file Matrix GNU: ", file_plot_Matrix_Gnu)

    fileData = {
        "fileButterfly_K1": fileButterflyK1,
        "fileMatrix": fileMatrix,
        "filePlotMatrix": file_plot_Matrix_Gnu,
        "fileWriteGnu": filegnu,
        "fileG": filePlotC_k1,
    }

    # print(torch.cuda.is_available())
    # # butterflyK2 = butterfly(bandNumber, choice, qmax, kpoint2, fileData, model)

    start = time()
    # dataK1 = waveFunction(choice, qmax, kpoint1, fileData, model)
    butterflyK1 = butterfly(bandNumber, choice, qmax, kpoint1, fileButterflyK1, model)
    end = time()
    print(f"Time calculating wavefunction: {end - start}s")
    # dataK2 = waveFunction(bandNumber, choice, qmax, kpoint2, fileData, model)
    # dataK3 = waveFunction(bandNumber, choice, qmax, kpoint3, fileData, model)
    # dataK4 = waveFunction(bandNumber, choice, qmax, kpoint4, fileData, model)

    # PlotMatrixGNU(fileMatrix, file_plot_Matrix_Gnu)
    # saveFunction(currentProg, "fileData")

    return None


if __name__ == "__main__":
    main()
