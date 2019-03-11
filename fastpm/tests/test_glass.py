from fastpm.glass import leapfrog, Solver, ParticleMesh

import numpy
from numpy.testing import assert_allclose

pm = ParticleMesh(BoxSize=128., Nmesh=[8, 8, 8])

def test_solver():
    solver= Solver(pm, B=1)
    Q = pm.generate_uniform_particle_grid()
    state = solver.glass(3333, Q)
    state.save('glass')
