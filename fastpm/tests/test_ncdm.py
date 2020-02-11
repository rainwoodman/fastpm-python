from runtests.mpi import MPITest
from fastpm.core import leapfrog, autostages
from fastpm.background import PerturbationGrowth

from pmesh.pm import ParticleMesh
from nbodykit.cosmology import Planck15, LinearPower
import numpy
from numpy.testing import assert_allclose

from fastpm.ncdm import Solver


@MPITest([1, 4])
def test_ncdm(comm):
    pm = ParticleMesh(BoxSize=512., Nmesh=[8, 8, 8], comm=comm)
    Plin = LinearPower(Planck15, redshift=0, transfer='EisensteinHu')
    solver = Solver(pm, Planck15, B=1)
    Q = pm.generate_uniform_particle_grid(shift=0)

    wn = solver.whitenoise(1234)
    dlin = solver.linear(wn, lambda k: Plin(k))
    state = solver.lpt(dlin, Q, a=1.0, order=2)

    dnonlin = solver.nbody(state, leapfrog([0.1, 1.0]))

    dnonlin.save('nonlin')
