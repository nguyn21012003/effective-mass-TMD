from copy import copy, deepcopy
from os import defpath

import matplotlib.pyplot as plt
import munkres
import numpy as np
import pandas as pd
from cupy import shape
from numpy import sqrt


def track_eigenvalues(D, V, Asequence):
    # n la tong so ma tran co cung size p x p
    # size cua D la n,p

    Dshape = np.shape(D)
    print(Dshape)
    n = Dshape[0]
    p = Dshape[-1]

    print(n, p)
    Vseq = np.zeros((n, p, p), dtype=complex)
    Dseq = np.zeros((n, p), dtype=complex)

    print(D)
    for i in range(n):
        # initial ordering is purely in decreasing order.
        # If any are complex, the sort is in terms of the
        # real part.
        tags = np.argsort(np.real(D), axis=0)[::-1]
        print(np.shape(tags))

        Dseq[i] = D[:, tags]
        Vseq[i] = V[:, tags]

    # now, treat each eigenproblem in sequence (after the first one.)
    m = munkres.Munkres()
    for i in range(1, n):
        # compute distance between systems
        D1 = Dseq[i - 1]
        D2 = Dseq[i]
        V1 = Vseq[i - 1]
        V2 = Vseq[i]
        dist = (1 - np.abs(np.dot(np.transpose(V1), V2))) * np.sqrt(distancematrix(D1.real, D2.real) ** 2 + distancematrix(D1.imag, D2.imag) ** 2)

        # Is there a best permutation? use munkres.
        reorder = m.compute(np.transpose(dist))
        reorder = [coord[1] for coord in reorder]

        Vs = Vseq[i]
        Vseq[i] = Vseq[i][:, reorder]
        Dseq[i] = Dseq[i, reorder]

        # also ensure the signs of each eigenvector pair
        # were consistent if possible
        S = np.squeeze(np.sum(Vseq[i - 1] * Vseq[i], 0).real) < 0

        Vseq[i] = Vseq[i] * (-S * 2 - 1)

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
