from os import defpath
import pandas as pd
from copy import copy, deepcopy
import numpy as np
import matplotlib.pyplot as plt


def track_eigenvalues(eigenArrInit, Barr):

    arr1 = None
    arr2 = None
    arr3 = None
    arr4 = None
    keys = []
    eigenArrNew = deepcopy(eigenArrInit)
    for key in eigenArrInit:
        keys.append(key)
    for qi in range(0, len(keys) - 3):

        key1 = keys[qi + 0]
        key2 = keys[qi + 1]
        key3 = keys[qi + 2]
        key4 = keys[qi + 3]

        arr1 = copy(eigenArrInit[key1])
        arr2 = copy(eigenArrInit[key2])
        arr3 = copy(eigenArrInit[key3])
        arr4 = copy(eigenArrInit[key4])

        for i in range(1, len(Barr) - 1):
            prev = arr1[i - 1]
            cur = arr1[i]
            next = arr1[i + 1]
            if arr1[i] > arr3[i]:
                # if (cur - prev) * (next - cur) < 0:
                print(Barr[i])
                arr1[i:], arr3[i:] = copy(arr3[i:]), copy(arr1[i:])
                arr2[i:], arr4[i:] = copy(arr4[i:]), copy(arr2[i:])
        eigenArrNew[key1] = arr1
        eigenArrNew[key2] = arr2
        eigenArrNew[key3] = arr3
        eigenArrNew[key4] = arr4

    for k in keys[:6]:
        plt.plot(Barr, eigenArrNew[k], label=k)
    plt.legend()
    plt.show()

    return None


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
    track_eigenvalues(listAllEigen, B_arr)

    return None


if __name__ == "__main__":
    main()
