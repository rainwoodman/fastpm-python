import numpy
from . import kernels as FKN

def gravity(x, pm, factor, f=None, return_deltak=False):
    field = pm.create(mode="real")
    layout = pm.decompose(x)
    field.paint(x, layout=layout, hold=False)

    deltak = field.r2c(out=Ellipsis)
    if f is None:
        f = numpy.empty_like(x)

    for d in range(field.ndim):
        force_d = deltak.apply(FKN.laplace) \
                  .apply(FKN.gradient(d), out=Ellipsis) \
                  .c2r(out=Ellipsis)
        force_d.readout(x, layout=layout, out=f[..., d])
    f[...] *= factor

    if return_deltak:
        rho = deltak.c2r(out=Ellipsis)
        rho = rho.readout(x, layout=layout)
        return f, deltak, rho
    else:
        return f
