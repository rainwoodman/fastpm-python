from fastpm.core import leapfrog, Solver

from pmesh.pm import ParticleMesh
from nbodykit.cosmology import Planck15, EHPower

pm = ParticleMesh(BoxSize=128., Nmesh=[8, 8, 8])

def test_leapfrog():
    l = list(leapfrog(0.1, 1.0, N=1))
    assert len(l) == 1 + 4 * 1

    l = list(leapfrog(0.1, 1.0, N=0))
    assert len(l) == 1

def test_solver():
    Plin = EHPower(Planck15, redshift=0)
    solver = Solver(pm, Planck15, B=1)
    Q = pm.generate_uniform_particle_grid()

    wn = solver.whitenoise(1234)
    dlin = solver.linear(wn, lambda k: Plin(k) ** 0.5, a=1.0)

    state = solver.lpt(dlin, Q, a=1.0, order=2)

    dnonlin = solver.nbody(state, leapfrog(1.0, 1.0, 2))

