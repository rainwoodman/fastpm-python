from . import core
import numpy
from .core import leapfrog

from pmesh.pm import ParticleMesh
from nbodykit.cosmology import Planck15

class Solver(core.Solver):
    def __init__(self, pm, B=2):
        fpm = ParticleMesh(Nmesh=pm.Nmesh * B, BoxSize=pm.BoxSize, dtype=pm.dtype, comm=pm.comm, resampler=pm.resampler)
        self.pm = pm
        self.fpm = fpm
        self.cosmology = Planck15 # any will do

    @property
    def nbodystep(self):
        return GlassStep(self)

    def glass(self, seed, Q):
        rng = numpy.random.RandomState(seed + self.pm.comm.rank)
        nbar = 1 / (self.pm.BoxSize.prod() / self.pm.comm.allreduce(len(Q)))

        # a spread of 3 will kill most of the anisotropiness of the Q grid.
        Q = Q + 3 * (rng.uniform(size=Q.shape) -0.5) * (nbar ** -0.3333333)

        state = core.StateVector(self, Q)
        # add a uniform random displacement
        state.S[...] = 0
        state.P[...] = 0
        state.F[...] = 0
        state.a['S'] = 0
        state.a['P'] = 0

        N = 3

        # The period is 2 pi. At pi/2 we encounter the first minimium power spectrum
        # damping means after 3 periods we have almost a glass power at the minimium.
        stages = numpy.linspace(0, numpy.pi * 2 * (N + 0.25), int(4 * (N + 0.25) + 1))

        self.nbody(state, leapfrog(stages))

        # to give it some reasonable cosmology parameters.
        state.a['S'] = 1
        state.a['P'] = 1
        state.a['F'] = 1
        return state


class GlassStep(core.FastPMStep):

    # Time step with a negative dimensionless PM force and a damping term.

    def Kick(self, state, ai, ac, af):
        fac = (af - ai)
        # add a damping term.
        state.P[...] = state.P[...] + fac * (state.F[...] - state.P[...])
        state.a['P'] = af

    def Drift(self, state, ai, ac, af):
        fac = (af - ai)
        state.S[...] = (state.S[...] + fac * state.P[...]) % self.pm.BoxSize
        state.a['S'] = af

    def Force(self, state, ai, ac, af):
        from .force.gravity import longrange
        nbar = 1.0 * state.csize / self.pm.Nmesh.prod()

        X = state.X

        layout = self.pm.decompose(X, smoothing=None)

        X1 = layout.exchange(X)

        rho = self.pm.paint(X1)
        rho /= nbar # 1 + delta
        delta_k = rho.r2c(out=Ellipsis)

        # - to invert the gravity direction.
        state.F[...] = - layout.gather(longrange(X1, delta_k, split=0, factor=1.0))

        state.a['F'] = af
        return
