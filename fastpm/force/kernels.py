import numpy

def laplace(k, v):
    kk = sum(ki ** 2 for ki in k)
    mask = (kk == 0).nonzero()
    kk[mask] = 1
    b = v / kk
    b[mask] = 0
    return b

def gradient(dir):
    def kernel(k, v):
        mask = ~numpy.bitwise_and.reduce([(ii == 0) | (ii == ni // 2) for ii, ni in zip(v.i, v.Nmesh)])
        factor = 1j
        return v * (1j * k[dir]) * mask
    return kernel

def longrange(r_split):
    if r_split != 0:
        def kernel(k, v):
            kk = sum(ki ** 2 for ki in k)
            return v * numpy.exp(-kk * r_split**2)
    else:
        def kernel(k, v):
            return v
    return kernel
