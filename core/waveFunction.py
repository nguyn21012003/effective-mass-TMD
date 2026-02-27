import csv
from pathlib import Path

import numpy as np
from numpy import linalg as LA
from scipy.signal import find_peaks
from tqdm import tqdm

from core.HamTMD import HamNN
from core.HamTMDNN import HamTNN


def waveFunction(dataInit, irreducibleMatrix, fileSave):
    p = dataInit["p"]
    coeff = dataInit["coeff"]
    numberWave = dataInit["numberWaveFunction"]
    modelNeighbor = dataInit["modelNeighbor"]
    alattice = dataInit["alattice"]
    k = dataInit["kpoint"]
    qmax = dataInit["qmax"]
    lambd = dataInit["lambda"]

    fileUpV = fileSave["waveUpV"]
    fileDownV = fileSave["waveDownV"]
    fileUpC = fileSave["waveUpC"]
    fileDownC = fileSave["waveDownC"]

    size = coeff * qmax
    iArr = np.arange(size)

    # ---------------- init arrays ----------------
    waveArrUpV, waveArrDownV = {}, {}
    waveArrUpC, waveArrDownC = {}, {}

    for i in range(numberWave):
        # valence
        waveArrUpV[f"psi_band2q_d0_{i}"] = np.zeros(size, complex)
        waveArrUpV[f"absPsi_band2q_d0_{i}"] = np.zeros(size, float)
        waveArrUpV[f"psi_band2q_d1_{i}"] = np.zeros(size, complex)
        waveArrUpV[f"absPsi_band2q_d1_{i}"] = np.zeros(size, float)
        waveArrUpV[f"psi_band2q_d2_{i}"] = np.zeros(size, complex)
        waveArrUpV[f"absPsi_band2q_d2_{i}"] = np.zeros(size, float)

        waveArrDownV[f"psi_band2q_d0_{i}"] = np.zeros(size, complex)
        waveArrDownV[f"absPsi_band2q_d0_{i}"] = np.zeros(size, float)
        waveArrDownV[f"psi_band2q_d1_{i}"] = np.zeros(size, complex)
        waveArrDownV[f"absPsi_band2q_d1_{i}"] = np.zeros(size, float)
        waveArrDownV[f"psi_band2q_d2_{i}"] = np.zeros(size, complex)
        waveArrDownV[f"absPsi_band2q_d2_{i}"] = np.zeros(size, float)

        # conduction
        waveArrUpC[f"psi_band2q_d0_{i}"] = np.zeros(size, complex)
        waveArrUpC[f"absPsi_band2q_d0_{i}"] = np.zeros(size, float)
        waveArrUpC[f"psi_band2q_d1_{i}"] = np.zeros(size, complex)
        waveArrUpC[f"absPsi_band2q_d1_{i}"] = np.zeros(size, float)
        waveArrUpC[f"psi_band2q_d2_{i}"] = np.zeros(size, complex)
        waveArrUpC[f"absPsi_band2q_d2_{i}"] = np.zeros(size, float)

        waveArrDownC[f"psi_band2q_d0_{i}"] = np.zeros(size, complex)
        waveArrDownC[f"absPsi_band2q_d0_{i}"] = np.zeros(size, float)
        waveArrDownC[f"psi_band2q_d1_{i}"] = np.zeros(size, complex)
        waveArrDownC[f"absPsi_band2q_d1_{i}"] = np.zeros(size, float)
        waveArrDownC[f"psi_band2q_d2_{i}"] = np.zeros(size, complex)
        waveArrDownC[f"absPsi_band2q_d2_{i}"] = np.zeros(size, float)

    # ---------------- Hamiltonian ----------------
    if modelNeighbor == "NN":
        _, hamu, hamd = HamNN(alattice, p, size, k, lambd, irreducibleMatrix)
    else:
        _, hamu, hamd = HamTNN(alattice, p, size, k, lambd, irreducibleMatrix)

    if np.gcd(p, qmax) != 1:
        return None

    _, vecs_u = LA.eigh(hamu)
    _, vecs_d = LA.eigh(hamd)

    # ---------------- eigenvectors ----------------
    for i in tqdm(range(numberWave), desc="Calc eigenvectors", colour="green"):

        idx_v = size - i - 1  # valence
        idx_c = size + i  # conduction

        # ===== SPIN UP =====
        # valence
        waveArrUpV[f"psi_band2q_d0_{i}"][:] = vecs_u[:, idx_v][0 * size : 1 * size]
        waveArrUpV[f"psi_band2q_d1_{i}"][:] = vecs_u[:, idx_v][1 * size : 2 * size]
        waveArrUpV[f"psi_band2q_d2_{i}"][:] = vecs_u[:, idx_v][2 * size : 3 * size]

        waveArrUpV[f"absPsi_band2q_d0_{i}"] = (
            np.abs(waveArrUpV[f"psi_band2q_d0_{i}"]) ** 2
        )
        waveArrUpV[f"absPsi_band2q_d1_{i}"] = (
            np.abs(waveArrUpV[f"psi_band2q_d1_{i}"]) ** 2
        )
        waveArrUpV[f"absPsi_band2q_d2_{i}"] = (
            np.abs(waveArrUpV[f"psi_band2q_d2_{i}"]) ** 2
        )

        # conduction
        waveArrUpC[f"psi_band2q_d0_{i}"][:] = vecs_u[:, idx_c][0 * size : 1 * size]
        waveArrUpC[f"psi_band2q_d1_{i}"][:] = vecs_u[:, idx_c][1 * size : 2 * size]
        waveArrUpC[f"psi_band2q_d2_{i}"][:] = vecs_u[:, idx_c][2 * size : 3 * size]

        waveArrUpC[f"absPsi_band2q_d0_{i}"] = (
            np.abs(waveArrUpC[f"psi_band2q_d0_{i}"]) ** 2
        )
        waveArrUpC[f"absPsi_band2q_d1_{i}"] = (
            np.abs(waveArrUpC[f"psi_band2q_d1_{i}"]) ** 2
        )
        waveArrUpC[f"absPsi_band2q_d2_{i}"] = (
            np.abs(waveArrUpC[f"psi_band2q_d2_{i}"]) ** 2
        )

        # ===== SPIN DOWN =====
        # valence
        waveArrDownV[f"psi_band2q_d0_{i}"][:] = vecs_d[:, idx_v][0 * size : 1 * size]
        waveArrDownV[f"psi_band2q_d1_{i}"][:] = vecs_d[:, idx_v][1 * size : 2 * size]
        waveArrDownV[f"psi_band2q_d2_{i}"][:] = vecs_d[:, idx_v][2 * size : 3 * size]

        waveArrDownV[f"absPsi_band2q_d0_{i}"] = (
            np.abs(waveArrDownV[f"psi_band2q_d0_{i}"]) ** 2
        )
        waveArrDownV[f"absPsi_band2q_d1_{i}"] = (
            np.abs(waveArrDownV[f"psi_band2q_d1_{i}"]) ** 2
        )
        waveArrDownV[f"absPsi_band2q_d2_{i}"] = (
            np.abs(waveArrDownV[f"psi_band2q_d2_{i}"]) ** 2
        )

        # conduction
        waveArrDownC[f"psi_band2q_d0_{i}"][:] = vecs_d[:, idx_c][0 * size : 1 * size]
        waveArrDownC[f"psi_band2q_d1_{i}"][:] = vecs_d[:, idx_c][1 * size : 2 * size]
        waveArrDownC[f"psi_band2q_d2_{i}"][:] = vecs_d[:, idx_c][2 * size : 3 * size]

        waveArrDownC[f"absPsi_band2q_d0_{i}"] = (
            np.abs(waveArrDownC[f"psi_band2q_d0_{i}"]) ** 2
        )
        waveArrDownC[f"absPsi_band2q_d1_{i}"] = (
            np.abs(waveArrDownC[f"psi_band2q_d1_{i}"]) ** 2
        )
        waveArrDownC[f"absPsi_band2q_d2_{i}"] = (
            np.abs(waveArrDownC[f"psi_band2q_d2_{i}"]) ** 2
        )

    offsets = {
        "Up_Conduction": find_landau_offsets(waveArrUpC, numberWave),
        "Up_Valence": find_landau_offsets(waveArrUpV, numberWave),
        "Down_Conduction": find_landau_offsets(waveArrDownC, numberWave),
        "Down_Valence": find_landau_offsets(waveArrDownV, numberWave),
    }

    print(offsets["Down_Valence"]["d1"])

    def write_offset(qmax, offset_list, file):
        path = Path(file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(file, "a") as uv:

            uv.write(f"{qmax}: {tuple(offset_list[:4])},\n")

    write_offset(qmax, offsets["Up_Conduction"]["d0"], "./offset/Up_Conduction.py")
    write_offset(qmax, offsets["Down_Conduction"]["d0"], "./offset/Down_Conduction.py")

    write_offset(qmax, offsets["Down_Valence"]["d1"], "./offset/km_d_v.py")
    write_offset(qmax, offsets["Down_Valence"]["d2"], "./offset/kp_d_v.py")

    write_offset(qmax, offsets["Up_Valence"]["d1"], "./offset/km_u_v.py")
    write_offset(qmax, offsets["Up_Valence"]["d2"], "./offset/kp_u_v.py")

    for key, data in offsets.items():
        print(f"\n[{key}]")
        for orbital in ["d0", "d1", "d2"]:
            print(f"{orbital}: {data[orbital]}")

    # ---------------- write files ----------------
    header = ["x"]
    for orbital in ["d0", "d1", "d2"]:
        for i in range(numberWave):
            header.append(f"{orbital}_lambda_{i}")

    def write_block(filename, waveDict):
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()

            for q in range(size):
                row = {"x": iArr[q]}
                for i in range(numberWave):
                    row[f"d0_lambda_{i}"] = waveDict[f"absPsi_band2q_d0_{i}"][q]
                    row[f"d1_lambda_{i}"] = waveDict[f"absPsi_band2q_d1_{i}"][q]
                    row[f"d2_lambda_{i}"] = waveDict[f"absPsi_band2q_d2_{i}"][q]
                writer.writerow(row)

    write_block(fileUpV, waveArrUpV)
    write_block(fileDownV, waveArrDownV)
    write_block(fileUpC, waveArrUpC)
    write_block(fileDownC, waveArrDownC)

    return offsets


def find_landau_offsets(
    waveArr,
    numberWave,
    orbital_keys=("d0", "d1", "d2"),
    amp_threshold=1e-2,
    prominence_ratio=0.1,
    peak_distance=10,
):
    offsets = {orb: [] for orb in orbital_keys}
    for orb in orbital_keys:
        i = 0
        offset_list = []
        while i < numberWave:
            key = f"absPsi_band2q_{orb}_{i}"
            psi_abs = waveArr[key]
            max_val = np.max(psi_abs)
            if max_val < amp_threshold:
                i += 1
                continue
            peaks, _ = find_peaks(
                psi_abs, prominence=max_val * prominence_ratio, distance=peak_distance
            )
            if 1 <= len(peaks) <= 4:
                offset_list.append(i)
                i += 2
            else:
                i += 1

        offsets[orb] = offset_list

    return offsets
