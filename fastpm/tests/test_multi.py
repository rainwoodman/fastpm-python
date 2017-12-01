from runtests.mpi import MPITest
from fastpm.core import leapfrog, autostages, Solver as CoreSolver

from fastpm.state import StateVector, Matter, Baryon, CDM, NCDM
from fastpm.multi import Solver

from pmesh.pm import ParticleMesh
from nbodykit.cosmology import Planck15, EHPower
import numpy
from numpy.testing import assert_allclose

from fastpm.multi import get_species_transfer_function_from_class

Planck15 = Planck15.clone(gauge='newtonian')
@MPITest([1, 4])
def test_solver(comm):
    pm = ParticleMesh(BoxSize=512., Nmesh=[8, 8, 8], comm=comm)
    solver = Solver(pm, Planck15, B=1)

    P_prm = Planck15.Primordial.get_pkprim

    tf = get_species_transfer_function_from_class(Planck15, 9)

    Q = pm.generate_uniform_particle_grid(shift=0)

    wn = solver.whitenoise(1234)
    prm = solver.primordial(wn, P_prm)
    ic = solver.lpt(prm, {
                '0': (Baryon, tf['d_b'], tf['dd_b']),
                '1': (CDM, tf['d_cdm'], tf['dd_cdm']),
                '4': (NCDM, tf['d_ncdm[0]'], tf['dd_ncdm[0]']),
            }, Q, a=0.1)

    print('0', ic.species['0'].S[0], ic.species['0'].P[0], ic.species['0'].Q[0])
    print('1', ic.species['1'].S[0], ic.species['1'].P[0], ic.species['1'].Q[0])
    print('4', ic.species['4'].S[0], ic.species['4'].P[0], ic.species['4'].Q[0])

    c2 = CoreSolver(pm, Planck15, B=1)
    Pk = lambda k: Planck15.get_pk(k, z=0)
    dlin = c2.linear(wn, Pk)
    ic2 = c2.lpt(dlin, Q, 0.1, order=1)
    print(ic2.S[0], ic2.P[0], ic2.Q[0])
    final2 = c2.nbody(ic2, leapfrog([0.1, 1.0]))

    final = solver.nbody(ic, leapfrog([0.1, 1.0]))
    print('0', final.species['0'].F[0])
    print('1', final.species['1'].F[0])
    print('4', final.species['4'].F[0])
    print(final2.F[0])

    final.to_catalog()
