from copy import copy, deepcopy
from os import defpath

import matplotlib.pyplot as plt
import munkres
import numpy as np
import pandas as pd
from tqdm import tqdm


def track_eigenvalues(D, V, Asequence):
    # n la tong so ma tran co cung size p x p
    # size cua D la nxp
    # D la ma tran tri rieng da cheo hoa cua n Hamiltonian co size la p x p = 3q x 3q
    # V la ma tran ham rieng da cheo hoa cua n Hamiltonian co size la p x p = 3q x 3q

    Dshape = np.shape(D)

    n = Dshape[0]
    p = Dshape[-1]

    # the initial eigenvalues/vectors in nominal order
    Dseq = np.zeros([n, p])  # De luu ma tran tri rieng moi theo thu tu n voi size la p
    Vseq = np.zeros((n, p, p), dtype=complex)  # De luu ma tran vector rieng moi theo thu tu n voi size la pxp

    for i in tqdm(range(n), desc="sort index 1"):
        tags = np.argsort(D[i])

        Dseq[i] = D[i][tags]
        Vseq[i] = V[i][:, tags]

    # now, treat each eigenproblem in sequence (after the first one.)
    m = munkres.Munkres()
    for i in tqdm(range(1, n), desc="sort using munkres"):
        # compute distance between systems
        D1 = Dseq[i - 1]
        D2 = Dseq[i]
        V1 = Vseq[i - 1]
        V2 = Vseq[i]
        dist = (1 - np.abs(np.dot(np.transpose(V1), V2))) * np.sqrt(distancematrix(D1, D2) ** 2)

        # Is there a best permutation? use munkres.
        reorder = m.compute(np.transpose(dist))
        reorder = [coord[1] for coord in reorder]

        Dseq[i] = Dseq[i, reorder]
        Vseq[i] = Vseq[i][:, reorder]

        # also ensure the signs of each eigenvector pair
        # were consistent if possible
        S = np.squeeze(np.sum(Vseq[i - 1] * Vseq[i], 0).real) < 0

        Vseq[i] = Vseq[i] * (-S.astype(int) * 2 - 1)

    return Dseq, Vseq


def distancematrix(vec1, vec2):
    """simple interpoint distance matrix"""
    v1, v2 = np.meshgrid(vec1, vec2)
    return np.abs(v1 - v2)


def main():
    NNdir = "./Sun-08-24/NN/"
    TNNdir = "./Sun-08-24/TNN/"

    dataframe = pd.read_csv(NNdir + "3band_Lambda2q_dataHofstadterButterfly_q_797_MoS2_GGA_G.dat")
    dataframe = dataframe.drop(columns=["evalues"])

    numLevels = 80
    lv1 = "E2q17"
    lv2 = "E2q20"

    listAllEigen = {}
    B_arr = dataframe["B_values"].values
    for i in range(numLevels + 1):
        listAllEigen[f"E2q{i}"] = dataframe[f"E2q{i}"].values
    track_eigenvalues(listAllEigen, B_arr, lv1)

    return None


if __name__ == "__main__":
    main()
