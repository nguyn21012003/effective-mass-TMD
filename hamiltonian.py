import os
from datetime import datetime
from time import time

import matplotlib.pyplot as plt

from file_python.irrMatrix import IR, IRNN, IRTNN
from file_python.irrMatrixTransform import IR as IR_tran
from file_python.irrMatrixTransform import IRNN as IRNN_tran
from file_python.irrMatrixTransform import IRTNN as IRTNN_tran
from file_python.mass import calcMass
from file_python.parameters import paraNNN, paraTNN
from file_python.waveFunction import waveFunction


def solver(qmax, material: str, model: dict, fileSave: dict):
    tran = True
    p = 1
    coeff = 2
    kpoint = [0, 0]  # Gamma
    ### the magnetic Brillouin zone now q times smaller than original Brillouin zone
    ### the K,K' points now are closed to the Gamma kpoint
    ### so we only consider the Gamma kpoint
    numberWaveFunction = 40
    modelParameters = model["modelParameters"]
    modelNeighbor = model["modelNeighbor"]

    functionMapping = {"TNN": paraTNN, "NNN": paraNNN}
    dataParameters = functionMapping[modelNeighbor](material, modelParameters)
    if tran:
        E0, h1, h2, h3, h4, h5, h6 = IR_tran(dataParameters)
        v1 = v2 = v3 = v4 = v5 = v6 = 0
        o1 = o2 = o3 = o4 = o5 = o6 = 0
        if modelNeighbor == "TNN":
            v1, v2, v3, v4, v5, v6 = IRNN_tran(dataParameters)
            o1, o2, o3, o4, o5, o6 = IRTNN_tran(dataParameters)
    else:
        E0, h1, h2, h3, h4, h5, h6 = IR(dataParameters)
        v1 = v2 = v3 = v4 = v5 = v6 = 0
        o1 = o2 = o3 = o4 = o5 = o6 = 0
        if modelNeighbor == "TNN":
            v1, v2, v3, v4, v5, v6 = IRNN(dataParameters)
            o1, o2, o3, o4, o5, o6 = IRTNN(dataParameters)

    irreducibleMatrix = {
        "NN": [E0, h1, h2, h3, h4, h5, h6],
        "NNN": [v1, v2, v3, v4, v5, v6],
        "TNN": [o1, o2, o3, o4, o5, o6],
    }
    dataInit = {}
    dataInit["numberWaveFunction"] = numberWaveFunction
    dataInit["kpoint"] = kpoint
    dataInit["qmax"] = qmax
    dataInit["coeff"] = coeff
    dataInit["modelNeighbor"] = modelNeighbor
    dataInit["p"] = p
    dataInit["alattice"] = dataParameters["alattice"]
    waveFunction(dataInit, irreducibleMatrix, fileSave["wave"])
    calcMass(dataInit, irreducibleMatrix, fileSave["mass"])


def main():
    qmax = 1564
    qrange = [2346, 1877, 1564, 1341, 1173, 1043, 939]
    material = "MoS2"
    bandNumber = 3
    modelPara = "GGA"
    modelNeighbor = "NN"
    model = {"modelParameters": modelPara, "modelNeighbor": modelNeighbor}
    kpoint1 = "G"

    time_run = datetime.now().strftime("%a-%m-%d")
    dir = f"./{time_run}/{modelNeighbor}/"
    os.makedirs(os.path.dirname(dir), exist_ok=True)

    print("folder direction: ", dir)

    filePlotWaveFunction = f"{dir}{bandNumber}band_PlotEigenVectors_q_{qmax}_{material}_{modelPara}_{kpoint1}.dat"
    fileMass = f"{dir}{bandNumber}band_dataMass_q_{qmax}_{material}_{modelPara}_{kpoint1}.dat"
    fileSave = {"wave": filePlotWaveFunction, "mass": fileMass}
    start = time()
    # for qmax in tqdm(qrange, ascii=" #", desc=f"Wave function in diff B", colour="magenta"):
    solver(qmax, material, model, fileSave)
    end = time()
    print(f"Time calculating wavefunction: {end - start}s")

    return None


if __name__ == "__main__":
    main()
