import numpy
from scipy.special import erfc
from . import kernels as FKN

def longrange(x, delta_k, split, factor):
    """ factor shall be 3 * Omega_M / 2, if delta_k is really 1 + overdensity """

    f = numpy.empty_like(x)

    pot_k = delta_k.apply(FKN.laplace) \
                  .apply(FKN.longrange(split), out=Ellipsis)

    for d in range(x.shape[1]):
        force_d = pot_k.apply(FKN.gradient(d)) \
                  .c2r(out=Ellipsis)
        force_d.readout(x, out=f[..., d])

    f[...] *= factor

    return f

def shortrange(tree1, tree2, r_split, r_cut, r_smth, factor):
    """ factor shall be G * M0 / H0** 2in order to match long range. 

        GM0 = 1.0 / (4 * pi) * (V / N)  (3 * Omega_M / 2)

        computes force for all particles in tree1 due to tree2.
    """
    X = tree1.input
    Y = tree2.input

    F = numpy.zeros_like(X)
    nd = F.shape[1]

    def shortrange_kernel(r):
        u = r / (r_split * 2)
        return erfc(u) + 2 * u / numpy.pi ** 0.5 * numpy.exp(-u**2)

    def force_kernel(r, i, j):
        mask = r > r_smth
        r = r[mask]
        i = i[mask]
        j = j[mask]

        if len(r) == 0: return

        R = X[i] - Y[j]

        imin = i.min()
        for d in range(nd):
            b = tree1.boxsize[d]
            Rd = R[:, d]
            Rd[Rd > 0.5 * b] -= b
            Rd[Rd < -0.5 * b] += b
            F1 = - 1 / r ** 3 * Rd
            s = shortrange_kernel(r)
            F1 *= s
            F1 = numpy.bincount(i - imin, weights=F1)
            F[imin:len(F1)+ imin, d] += F1

    tree1.root.enum(tree2.root, r_cut, process=force_kernel)
    return F * factor
