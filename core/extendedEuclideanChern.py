import numpy as np
from tqdm import tqdm


def checkValidChern(a: int, b: int) -> tuple:
    """
    Compute the GCD of two integers t and s in diophantine equation
    a * p + b * q = gcd(tr,sr)

    :param a: 1st Integer
    :param b: 2nd integer
    :return:
    g: int
        gcd of a and b

    x: int
        coeff of a*x + b*y = g.
    y: int
        coeff of a*x + b*y = g.

    This algorithm give better time-consumption compared to the bruteForce one. The complexity now is O(log q), the bruteForce one is O(q^n)

    """

    # https://cp-algorithms.com/algebra/extended-euclid-algorithm.html

    if b == 0:
        return a, 1, 0
    else:
        g, x, y = checkValidChern(b, a % b)
        return g, y, x - (a // b) * y


def chern(p, q, band):
    qmax = band * q
    sr_list, tr_list = [], []
    g, x0, y0 = checkValidChern(q, p)

    if g != 1:
        raise ValueError("not coprime")

    for r in range(0, qmax + 1):
        sr = x0 * r
        tr = y0 * r
        # sr_mod = sr % p if p != 0 else sr
        # tr_mod = tr % q if q != 0 else tr

        if p != 0:
            sr_mod = (sr + p // 2) % p - p // 2
            sr_list.append(sr_mod)
        if q != 0:
            tr_mod = (tr + q // 2) % q - q // 2
            tr_list.append(tr_mod)

    Chern_list = np.diff(tr_list)

    return Chern_list.tolist(), tr_list


# def testcase():
#     qmax = 797
#     for p in tqdm(range(1, qmax + 1), desc=f"p iter"):
#         if np.gcd(p, qmax) == 1:
#             clist, tlist = chern(p, 1 * qmax, 3)
#
#     return None
#
#
# testcase()
