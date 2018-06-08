import numpy

def laplace(k, v):
    kk = sum(ki ** 2 for ki in k)
    mask = (kk == 0).nonzero()
    kk[mask] = 1
    b = v / kk
    b[mask] = 0
    return b

def gradient(dir, order=0):
    if order == 0:
        def kernel(k, v):
            # clear the nyquist to ensure field is real
            mask = v.i[dir] != v.Nmesh[dir] // 2
            return v * (1j * k[dir]) * mask
    if order == 1:
        def kernel(k, v):
            cellsize = (v.BoxSize[dir] / v.Nmesh[dir])
            w = k[dir] * cellsize

            a = 1 / (6.0 * cellsize) * (8 * numpy.sin(w) - numpy.sin(2 * w))
            # a is already zero at the nyquist to ensure field is real
            return v * (1j * a)
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
