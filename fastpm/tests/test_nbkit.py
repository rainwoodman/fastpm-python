from __future__ import print_function
from runtests.mpi import MPITest

from numpy.testing import assert_allclose
from numpy.testing import dec
import fastpm

try:
    import nbodykit
    nbodykit.setup_logging('debug')
except ImportError:
    nbodykit = None


@MPITest([1, 4])
@dec.skipif(nbodykit is None, "nbodykit test doesn't work on travis; is not installed")
def test_nbkit(comm):
    from fastpm.nbkit import FastPMCatalogSource
    from nbodykit.lab import cosmology, FOF, LinearMesh
    cosmo = cosmology.Planck15
    power = cosmology.LinearPower(cosmo, 0)

    linear = LinearMesh(power, 256., 64, seed=400, comm=comm)
    sim = FastPMCatalogSource(linear, boost=2, Nsteps=5, cosmo=cosmo)
    fof = FOF(sim, 0.2, 8)
    sim['Labels'] = fof.labels
    sim.save('nbkit-%d' % comm.size, ['Position', 'InitialPosition', 'Displacement', 'Labels'])
    features = fof.find_features()
    features.save('nbkit-fof-%d' % comm.size, ['CMPosition', 'Length'])
    #print(features._size, features._csize)
    assert_allclose(features.csize, 500, rtol=0.1)
