from fastpm.state import StateVector, Matter, Baryon, CDM, NCDM
from runtests.mpi import MPITest

from nbodykit.cosmology import Planck15 as cosmo
import numpy

BoxSize = 100.
Q = numpy.zeros((100, 3))

@MPITest([1, 4])
def test_create(comm):

    matter = Matter(cosmo, BoxSize, Q, comm)

    cdm = CDM(cosmo, BoxSize, Q, comm)
    cdm.a['S'] = 1.0
    cdm.a['P'] = 1.0
    baryon = Baryon(cosmo, BoxSize, Q, comm)
    baryon.a['S'] = 1.0
    baryon.a['P'] = 1.0

    state = StateVector(cosmo, {'0': baryon, '1' : cdm}, comm)
    state.a['S'] = 1.0
    state.a['P'] = 1.0
    state.save("state")


