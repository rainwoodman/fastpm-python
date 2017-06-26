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
    """ factor shall be G * M0 / H0** 2 in order to match long range. 

        GM0 / H0 ** 2 = 1.0 / (4 * pi) * (V / N)  (3 * Omega_M / 2)

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
        r, i, j = cut(r, i, j, r_smth)
        if len(r) == 0: return

        R = X[i] - Y[j]

        s = shortrange_kernel(r)
        r3inv = 1 / r ** 3 * s
        for d in range(nd):
            b = tree1.boxsize[d]
            Rd = wrap(R[:, d], b)
            F1 = - r3inv * Rd
            numpy.add.at(F[..., d], i, F1)

    tree1.root.enum(tree2.root, r_cut, process=force_kernel)
    return F * factor

def wrap(r, b):
    Rd = r.copy()
    Rd[Rd > 0.5 * b] -= b
    Rd[Rd < -0.5 * b] += b
    return Rd

def cut(r, i, j, rmin):
    mask = r > rmin
    r = r[mask]
    i = i[mask]
    j = j[mask]
    return r, i, j

def timestep(tree, a, E, r_cut, r_smth, factor, eta=0.03):
    """ factor is GM0 / H0 ** 2 """

    X = tree.input
    h = numpy.zeros_like(X[..., 0])
    h[...] = numpy.inf

    fac = a ** 2 * E * factor ** 0.5 * (a / 2) ** 0.5 * eta

    def gettimestep(r, i, j):
        r, i, j = cut(r, i, j, r_smth)
        if len(r) == 0: return
        tau = fac * r ** 1.5
        numpy.fmin.at(h, i, tau)

    tree.root.enum(tree.root, r_cut, gettimestep)
    return h
