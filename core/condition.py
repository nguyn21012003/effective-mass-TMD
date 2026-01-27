def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b)


def pbc(i, q):

    return i % (q)
