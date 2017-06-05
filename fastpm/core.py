import numpy

from pmesh.pm import ParticleMesh
from .background import PerturbationGrowth
from .operators import lpt1, lpt2source, gravity
from nbodykit.cosmology import Cosmology

class StateVector(object):
    def __init__(self, solver, Q, S=None, P=None, F=None):
        self.pm = solver.pm
        self.Q = Q
        self.csize = solver.pm.comm.allreduce(len(self.Q))
        self.dtype = self.Q.dtype

        if S is None: S = numpy.zeros_like(self.Q)
        if P is None: P = numpy.zeros_like(self.Q)
        if F is None: F = numpy.zeros_like(self.Q)
        self.S = S
        self.P = P
        self.F = F

    @property
    def X(self):
        return self.S + self.Q

    def to_mesh(self):
        real = self.pm.create(mode='real')
        x = self.X
        layout = self.pm.decompose(x)
        real.paint(x, layout=layout, hold=False)
        return real

class Solver(object):
    def __init__(self, pm, cosmology, B=1):
        self.pm = pm
        if not isinstance(cosmology, Cosmology):
            raise TypeError("only nbodykit.cosmology object is supported")

        self.cosmology = cosmology

        fpm = ParticleMesh(Nmesh=pm.Nmesh * B, BoxSize=pm.BoxSize, dtype=pm.dtype, comm=pm.comm, resampler=pm.resampler)
        self.boosted_pm = fpm

    def whitenoise(self, seed, unitary=False):
        return self.pm.generate_whitenoise(seed, mode='complex', unitary=unitary)

    def linear(self, whitenoise, tf, a=1.0):
        pt = PerturbationGrowth(self.cosmology, a=[a])
        return whitenoise.apply(lambda k, v:
                        pt.D1(a) * tf(sum(ki ** 2 for ki in k)**0.5) * v / v.BoxSize.prod() ** 0.5)

    def lpt(self, linear, Q, a, order=2):
        assert order in (1, 2)

        state = StateVector(self, Q)
        pt = PerturbationGrowth(self.cosmology, a=[a])
        DX1 = pt.D1(a) * lpt1(linear, Q)

        V1 = a ** 2 * pt.f1(a) * pt.E(a) * DX1
        if order == 2:
            DX2 = pt.D2(a) * lpt1(lpt2source(linear), Q)
            V2 = a ** 2 * pt.f2(a) * pt.E(a) * DX2
            state.S[...] = DX1 + DX2
            state.P[...] = V1 + V2
        else:
            state.S[...] = DX1
            state.P[...] = V1

        return state

    def nbody(self, state, stepping):
        nbar = 1.0 * state.csize / self.boosted_pm.Nmesh.prod()
        def Kick(ai, ac, af):
            pt = PerturbationGrowth(self.cosmology, a=[ai, ac, af])
            fac = 1 / (ac ** 2 * pt.E(ac)) * (pt.Gf(af) - pt.Gf(ai)) / pt.gf(ac)
            state.P[...] = state.P[...] + fac * state.F[...]

        def Drift(ai, ac, af):
            pt = PerturbationGrowth(self.cosmology, a=[ai, ac, af])
            fac = 1 / (ac ** 3 * pt.E(ac)) * (pt.Gp(af) - pt.Gp(ai)) / pt.gp(ac)
            state.S[...] = state.S[...] + fac * state.P[...]

        def Force(ai, ac, af):
            state.F[...] = gravity(state.X, self.boosted_pm, factor=1.5 * self.cosmology.Om0 / nbar)

        actions = dict(K=Kick, D=Drift, F=Force)

        for action, ai, ac, af in stepping:
            actions[action](ai, ac, af)

        return state

def leapfrog(ai, af, N):
    assert N >= 0
    if N == 0: assert ai == af
    a = numpy.linspace(ai, af, N + 1, endpoint=True)
    # first force calculation for jump starting
    yield 'F', 0, 0, ai
    x, p, f = ai, ai, ai

    for i in range(N):
        a0 = a[i]
        a1 = a[i + 1]
        ah = (a0 * a1) ** 0.5
        yield 'K', p, f, ah
        p = ah
        yield 'D', x, p, a1
        x = a1
        yield 'F', f, x, a1
        f = a1
        yield 'K', p, f, a1
        p = a1
