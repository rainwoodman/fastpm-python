from fastpm.core import leapfrog, Solver, autostages

from pmesh.pm import ParticleMesh
from nbodykit.cosmology import Planck15, EHPower
import numpy
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
