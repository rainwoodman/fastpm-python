from fastpm.core import leapfrog, Solver, autostages
from fastpm.background import PerturbationGrowth

from pmesh.pm import ParticleMesh
from nbodykit.cosmology import Planck15, EHPower
import numpy
from numpy.testing import assert_allclose

pm = ParticleMesh(BoxSize=128., Nmesh=[8, 8, 8])

def test_autostages():
    for knots in [
        [0.1, 1.0],
        [1.0],
        [0.5],
        [0.1, 0.2, 1.0],
        ]:

        l = autostages(knots, astart=0.1, N=12)

        assert(len(l) == 12)
        for k in knots:
            assert k in l

def test_leapfrog():
    l = list(leapfrog(numpy.linspace(0.1, 1.0, 2, endpoint=True)))
    assert len(l) == 1 + 4 * (2 - 1)

    l = list(leapfrog([1.0]))
    assert len(l) == 1

def test_solver():
    Plin = EHPower(Planck15, redshift=0)
    solver = Solver(pm, Planck15, B=1)
    Q = pm.generate_uniform_particle_grid()

    wn = solver.whitenoise(1234)
    dlin = solver.linear(wn, lambda k: Plin(k))

    state = solver.lpt(dlin, Q, a=1.0, order=2)

    dnonlin = solver.nbody(state, leapfrog([1.0]))

    dnonlin.save('nonlin')

def test_lpt():
    Plin = EHPower(Planck15, redshift=0)
    solver = Solver(pm, Planck15, B=1)
    Q = pm.generate_uniform_particle_grid()

    wn = solver.whitenoise(1234)
    dlin = solver.linear(wn, lambda k: Plin(k))

    state1 = solver.lpt(dlin, Q, a=0.01, order=1)
    state2 = solver.lpt(dlin, Q, a=1.0, order=1)

    pt = PerturbationGrowth(Planck15, a=[0.01, 1.0], a_normalize=1.0)
#    print((state2.P[...] / state1.P[...]))
    print((state2.P[...] - state1.P[...]) / state1.F[...])

    fac = 1 / (0.01 ** 2 * pt.E(0.01)) * (pt.Gf(1.0) - pt.Gf(0.01)) / pt.gf(0.01)
    assert_allclose(state2.P[...], state1.P[...] + fac * state1.F[...])
