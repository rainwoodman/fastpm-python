from fastpm.glass import generate_glass_particle_grid, ParticleMesh

import numpy
from numpy.testing import assert_allclose

pm = ParticleMesh(BoxSize=128., Nmesh=[8, 8, 8])

def test_glass():
    X = generate_glass_particle_grid(pm, 123)

    assert pm.comm.allreduce(len(X)) == pm.Nmesh.prod()
