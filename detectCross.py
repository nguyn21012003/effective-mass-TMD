from os import defpath
import pandas as pd
from copy import deepcopy


def sortCrossing(listEigen, level1, level2):
    arr1 = deepcopy(listEigen[level1])
    arr2 = deepcopy(listEigen[level2])
    for i in range(1, len(arr1)):
        deltaE_prev = arr1[i] - arr1[i - 1]
        deltaE_curr = arr2[i] - arr1[i]
        print(deltaE_prev)
        if deltaE_prev * deltaE_curr < 0:
            tmp1 = deepcopy(arr1[i:])
            tmp2 = deepcopy(arr2[i:])
            arr1[i:], arr2[i:] = tmp2, tmp1

    listEigen[level1] = arr1
    listEigen[level2] = arr2
    return None


def main():
    NNdir = "./Sat-08-23/NN/"
    TNNdir = "./Sat-08-23/TNN/"

    dataframe = pd.read_csv(TNNdir + "3band_Lambda2q_dataHofstadterButterfly_q_2001_MoS2_GGA_G.dat")
    dataframe = dataframe.drop(columns=["evalues"])

    numLevels = 80
    lv1 = "E2q17"
    lv2 = "E2q20"

    listEigen = {}
    for i in range(numLevels + 1):
        listEigen[f"E2q{i}"] = dataframe[dataframe.columns[i + 2]].values

    print(sortCrossing(listEigen, lv1, lv2))

    return None


if __name__ == "__main__":
    main()
