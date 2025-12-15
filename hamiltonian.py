from datetime import datetime
import os

from core.butterfly import butterfly
from core.irrMatrix import IR, IRNN, IRTNN
from core.irrMatrixTransform import IR as IR_tran
from core.irrMatrixTransform import IRNN as IRNN_tran
from core.irrMatrixTransform import IRTNN as IRTNN_tran
from core.mass import calcMass
from core.parameters import paraNN, paraTNN
from core.waveFunction import waveFunction


def solver(qmax: int, material: str, model: dict, fileSave: dict[str, str]):
    tran = True
    p = 1
    coeff = 2
    kpoint = [0, 0]  # Gamma
    ### the magnetic Brillouin zone now q times smaller than original Brillouin zone
    ### the K,K' points now are closed to the Gamma kpoint
    ### so we only consider the Gamma kpoint
    numberWaveFunction = 60
    modelParameters = model["modelParameters"]
    modelNeighbor = model["modelNeighbor"]

    functionMapping = {"TNN": paraTNN, "NN": paraNN}
    dataParameters = functionMapping[modelNeighbor](material, modelParameters)
    if tran:
        h0, h1, h2, h3, h4, h5, h6 = IR_tran(dataParameters)
        v1 = v2 = v3 = v4 = v5 = v6 = 0
        o1 = o2 = o3 = o4 = o5 = o6 = 0
        if modelNeighbor == "TNN":
            v1, v2, v3, v4, v5, v6 = IRNN_tran(dataParameters)
            o1, o2, o3, o4, o5, o6 = IRTNN_tran(dataParameters)
    else:
        h0, h1, h2, h3, h4, h5, h6 = IR(dataParameters)
        v1 = v2 = v3 = v4 = v5 = v6 = 0
        o1 = o2 = o3 = o4 = o5 = o6 = 0
        if modelNeighbor == "TNN":
            v1, v2, v3, v4, v5, v6 = IRNN(dataParameters)
            o1, o2, o3, o4, o5, o6 = IRTNN(dataParameters)

    irreducibleMatrix = {
        "NN": [h0, h1, h2, h3, h4, h5, h6],
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
    # calcMass(dataInit, irreducibleMatrix, fileSave["mass"])
    # butterfly(dataInit, irreducibleMatrix, fileSave["butterfly"])


def main():
    qmax = 93
    material = "WTe2"
    modelPara = "GGA"
    modelNeighbor = "TNN"
    model = {"modelParameters": modelPara, "modelNeighbor": modelNeighbor}

    time_run = datetime.now().strftime("%a-%m-%d")
    dir = f"./{time_run}/{modelNeighbor}/"
    os.makedirs(os.path.dirname(dir), exist_ok=True)

    print("folder direction: ", dir)
    print(material, modelNeighbor, modelPara)

    # for qmax in tqdm(qrange, ascii=" #", desc=f"Wave function in diff B", colour="magenta"):
    filePlotWaveFunction = f"{dir}WaveFunction_q_{qmax}_{material}_{modelPara}.dat"
    fileMass = f"{dir}Mass_q_{qmax}_{material}_{modelPara}.dat"
    fileBut = f"{dir}Butterfly_q_{qmax}_{material}_{modelPara}.dat"

    fileSave = {"wave": filePlotWaveFunction, "mass": fileMass, "butterfly": fileBut}
    solver(qmax, material, model, fileSave)

    return None


if __name__ == "__main__":
    main()
