from fastpm.force import kernels
from numpy.testing import assert_allclose
from pmesh.pm import ParticleMesh

pm = ParticleMesh(BoxSize=128., Nmesh=[8, 8, 8])

def test_gradient():
    # asserts the gradient is hermitian

    data = pm.generate_whitenoise(type='real', seed=333)
    for d in range(pm.ndim):
        g1 = data.r2c().apply(kernels.gradient(d))
        assert_allclose(g1.c2r().r2c(), g1, atol=1e-9)
