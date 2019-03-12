from . import kernels as FKN
import numpy

def lpt1(dlin_k, q, resampler='cic'):
    """ Run first order LPT on linear density field, returns displacements of particles
        reading out at q. The result has the same dtype as q.
    """
    basepm = dlin_k.pm

    ndim = len(basepm.Nmesh)
    delta_k = basepm.create('complex')

    layout = basepm.decompose(q)
    local_q = layout.exchange(q)

    source = numpy.zeros((len(q), ndim), dtype=q.dtype)
    for d in range(len(basepm.Nmesh)):
        disp = dlin_k.apply(FKN.laplace) \
                    .apply(FKN.gradient(d, order=1), out=Ellipsis) \
                    .c2r(out=Ellipsis)
        local_disp = disp.readout(local_q, resampler=resampler)
        source[..., d] = layout.gather(local_disp)
    return source

def lpt2source(dlin_k):
    """ Generate the second order LPT source term.  """
    source = dlin_k.pm.create('real')
    source[...] = 0
    if dlin_k.ndim != 3: # only for 3d
        return source.r2c(out=Ellipsis)

    D1 = [1, 2, 0]
    D2 = [2, 0, 1]

    phi_ii = []

    # diagnoal terms
    for d in range(dlin_k.ndim):
        phi_ii_d = dlin_k.apply(FKN.laplace) \
                     .apply(FKN.gradient(d, order=1), out=Ellipsis) \
                     .apply(FKN.gradient(d, order=1), out=Ellipsis) \
                     .c2r(out=Ellipsis)
        phi_ii.append(phi_ii_d)

    for d in range(3):
        source[...] += phi_ii[D1[d]].value * phi_ii[D2[d]].value

    # free memory
    phi_ii = []

    phi_ij = []
    # off-diag terms
    for d in range(dlin_k.ndim):
        phi_ij_d = dlin_k.apply(FKN.laplace) \
                 .apply(FKN.gradient(D1[d], order=1), out=Ellipsis) \
                 .apply(FKN.gradient(D2[d], order=1), out=Ellipsis) \
                 .c2r(out=Ellipsis)

        source[...] -= phi_ij_d[...] ** 2

    # this ensures x = x0 + dx1(t) + d2(t) for 2LPT

    source[...] *= 3.0 / 7
    return source.r2c(out=Ellipsis)



