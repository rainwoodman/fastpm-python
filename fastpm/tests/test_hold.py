from fastpm.core import leapfrog, autostages
from fastpm.hold import Solver

from pmesh.pm import ParticleMesh
from nbodykit.cosmology import Planck15, LinearPower
import numpy
pm = ParticleMesh(BoxSize=32., Nmesh=[16, 16, 16])

def test_solver():
    Plin = LinearPower(Planck15, redshift=0, transfer='EisensteinHu')
    solver = Solver(pm, Planck15, B=2)
    Q = pm.generate_uniform_particle_grid(shift=0)

    wn = solver.whitenoise(1234)
    dlin = solver.linear(wn, lambda k: Plin(k))

    state = solver.lpt(dlin, Q, a=0.3, order=2)

    dnonlin = solver.nbody(state, leapfrog([0.3, 0.35]))

    dnonlin.save('nonlin')
