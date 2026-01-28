from numpy import exp


def Ham(alpha,beta):
    ham = (
        h0
        + exp(2j * alpha) * h1
        + exp(1j * (alpha - beta)) * h2
        + exp(1j * (-alpha - beta)) * h3
        + exp(-2j * alpha) * h4
        + exp(1j * (-alpha + beta)) * h5
        + exp(1j * (alpha + beta)) * h6
        + exp(4j * alpha) * o1
        + exp(2j * (alpha - beta)) * o2
        + exp(2j * (-alpha - beta)) * o3
        + exp(-4j * alpha) * o4
        + exp(2j * (-alpha + beta)) * o5
        + exp(2j * (alpha + beta)) * o6
        + exp(1j * (3 * alpha - beta)) * v1
        + exp(1j * (-2 * beta)) * v2
        + exp(1j * (-3 * alpha - beta)) * v3
        + exp(1j * (-3 * alpha + beta)) * v4
        + exp(1j * (2 * beta)) * v5
        + exp(1j * (3 * alpha + beta)) * v6
    )

    return ham
