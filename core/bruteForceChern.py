import numpy as np

from tqdm import tqdm


def chern(p, q, band):

    qmax = band * q
    sr_list, tr_list = [], []
    for r in range(0, qmax + 1):
        for tr in range(-int(q / 2), int(q / 2) + 1):
            for sr in range(-q, q + 1):
                if r == q * sr + p * tr:
                    sr_list.append(sr)
                    tr_list.append(tr)

                    break
            else:
                continue
            break

    Chern_list = []
    for i in range(qmax):
        Chern_list.append(tr_list[i + 1] - tr_list[i])

    return Chern_list, tr_list


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
