import numpy

from . import core

class Solver(core.Solver):
    @property
    def nbodystep(self):
        return FastPMStep(self)

class FastPMStep(core.FastPMStep):
    def __init__(self, solver):
        core.FastPMStep.__init__(self, solver)

    def prepare_force(self, state, smoothing):
        nbar = 1.0 * state.csize / self.pm.Nmesh.prod()

        X = state.X

        layout = self.pm.decompose(X, smoothing)

        X1 = layout.exchange(X)

        rho = self.pm.paint(X1)
        rho /= nbar # 1 + delta
        return layout, X1, rho

    def Force(self, state, ai, ac, af):
        from .force.gravity import longrange

        # use the default PM support
        layout, X1, rho = self.prepare_force(state, smoothing=None)

        state.RHO[...] = layout.gather(rho.readout(X1))

        delta_k = rho.r2c(out=Ellipsis)

        delta_k = phase_space_linear_ncdm(delta_k, self.cosmology, ac)

        state.F[...] = layout.gather(longrange(X1, delta_k, split=0, factor=1.5 * self.cosmology.Om0))

        state.a['F'] = af
        return dict(delta_k=delta_k)


def phase_space_linear_ncdm(delta_k, cosmology, ac):
    z = 1 / ac - 1
    tf = cosmology.get_transfer(z=z)

    ktf = tf['k']

    d_cdm = tf['d_cdm']
    d_b = tf['d_b']
    d_ncdm = tf['d_ncdm[0]']

    Ocdm = cosmology.Omega_cdm(z)
    Ob = cosmology.Omega_b(z)
    Oncdm = cosmology.Omega_ncdm(z)

    d_m = (d_cdm * Ocdm + d_b * Ob + d_ncdm * Oncdm) / ( Ob + Ocdm + Oncdm)
    d_cb = (d_cdm * Ocdm + d_b * Ob) / ( Ob + Ocdm)

    #print(ktf)
    #print(d_m / d_cb)
    def transfer(k, v):
        k = sum(ki **2 for ki in k) ** 0.5
        r = numpy.interp(k, ktf, d_m / d_cb)
        return r * v

    delta_k = delta_k.apply(transfer, out=Ellipsis)
    return delta_k
