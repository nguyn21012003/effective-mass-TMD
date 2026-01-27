import os
from datetime import datetime

# from core.butterfly import butterfly
from core.irrMatrix import IR, IRNN, IRTNN
from core.irrMatrixTransform import IR as IR_tran
from core.irrMatrixTransform import IRNN as IRNN_tran
from core.irrMatrixTransform import IRTNN as IRTNN_tran

# from core.mass import calcMass
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
    numberWaveFunction = 100
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
    dataInit["lambda"] = dataParameters["lambda"]
    waveFunction(dataInit, irreducibleMatrix, fileSave)
    # calcMass(dataInit, irreducibleMatrix, fileSave["mass"])
    # butterfly(dataInit, irreducibleMatrix, fileSave["butterfly"])


def main():
    qmax = 113
    print(qmax)
    material = "MoS2"
    modelPara = "GGA"
    modelNeighbor = "NN"
    model = {"modelParameters": modelPara, "modelNeighbor": modelNeighbor}

    time_run = datetime.now().strftime("%a-%m-%d")
    dir = f"./{time_run}/{modelNeighbor}"

    dir_conduction_up = f"./{time_run}/{modelNeighbor}/conduction/up/"
    dir_conduction_down = f"./{time_run}/{modelNeighbor}/conduction/down/"
    dir_valence_up = f"./{time_run}/{modelNeighbor}/valence/up/"
    dir_valence_down = f"./{time_run}/{modelNeighbor}/valence/down/"
    os.makedirs(os.path.dirname(dir), exist_ok=True)
    os.makedirs(os.path.dirname(dir_conduction_up), exist_ok=True)
    os.makedirs(os.path.dirname(dir_conduction_down), exist_ok=True)
    os.makedirs(os.path.dirname(dir_valence_up), exist_ok=True)
    os.makedirs(os.path.dirname(dir_valence_down), exist_ok=True)

    print("folder direction: ", dir)
    print(material, modelNeighbor, modelPara)

    # for qmax in tqdm(qrange, ascii=" #", desc=f"Wave function in diff B", colour="magenta"):
    wave1 = f"{dir}WaveFunction_q_{qmax}_{material}_{modelPara}.dat"
    waveUpC = f"{dir_conduction_up}/wave_up.dat"
    waveDownC = f"{dir_conduction_down}/wave_down.dat"
    waveUpV = f"{dir_valence_up}/wave_up.dat"
    waveDownV = f"{dir_valence_down}/wave_down.dat"

    fileMass = f"{dir}Mass_q_{qmax}_{material}_{modelPara}.dat"
    fileBut = f"{dir}Butterfly_q_{qmax}_{material}_{modelPara}.dat"

    fileSave = {
        "wave": wave1,
        "mass": fileMass,
        "butterfly": fileBut,
        "waveUpV": waveUpV,
        "waveDownV": waveDownV,
        "waveUpC": waveUpC,
        "waveDownC": waveDownC,
    }
    solver(qmax, material, model, fileSave)

    
    return None


if __name__ == "__main__":
    main()
