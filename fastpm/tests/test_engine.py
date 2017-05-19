from __future__ import print_function
from abopt.vmad2 import CodeSegment, Engine, statement, programme, ZERO, logger

from numpy.testing import assert_raises, assert_array_equal, assert_allclose
from numpy.testing.decorators import skipif
import numpy
import logging

logger.setLevel(level=logging.WARNING)

from pmesh.abopt import ParticleMesh, RealField, ComplexField, check_grad

from ..engine import FastPMEngine

from ..perturbation import PerturbationTheory

pm = ParticleMesh(BoxSize=1.0, Nmesh=(4, 4, 4), dtype='f8')

cosmo = lambda : None

cosmo.Om0 = 0.3
cosmo.Ode0 = 0.7
cosmo.Ok0 = 0.0

def pk(k):
    p = (k / 0.01) ** -3 * 80000
    return p

pt = PerturbationTheory(cosmo)

from nbodykit.source.mesh.memory import MemoryMesh
from nbodykit.algorithms.fftpower import FFTPower

def test_force():
    pm = ParticleMesh(BoxSize=8.0, Nmesh=(4, 4, 4), dtype='f8')
    engine = FastPMEngine(pm)
    code = CodeSegment(engine)
    code.create_linear_field(whitenoise='whitenoise', powerspectrum=pk, dlinear_k='dlinear_k')
    code.solve_lpt(pt=pt, dlinear_k='dlinear_k', aend=0.1, s='s', v='v', s1='s1', s2='s2')
    code.force(s='s', force='force', force_factor=1.0)

    field = engine.pm.generate_whitenoise(seed=1234).c2r()
    eps = field.cnorm() ** 0.5 * 1e-3

    from fastpm.operators import gravity
    s, force = code.compute(['s', 'force'], init={'whitenoise' : field})
    f_truth = gravity(engine.get_x(s), engine.pm, 1.0)
    assert_allclose(force, f_truth, atol=1e-8, rtol=1e-4)

    check_grad(code, 'force', 'whitenoise', init={'whitenoise': field}, eps=eps,
                rtol=1e-2)

def test_linear():
    pm = ParticleMesh(BoxSize=8.0, Nmesh=(4, 4), dtype='f8')
    engine = FastPMEngine(pm)
    code = CodeSegment(engine)
    def pk(k):
        return 0.0 * k + 1.0
    code.create_linear_field(whitenoise='whitenoise', powerspectrum=pk, dlinear_k='dlinear_k')
    code.create_whitenoise(whitenoise='whitenoise2', powerspectrum=pk, dlinear_k='dlinear_k')

    def tf1(k):
        k2 = sum(ki**2 for ki in k)
        r = (pk(k2 ** 0.5) / engine.pm.BoxSize.prod()) ** 0.5
        r[k2 == 0] = 1.0
        return r
    def tf2(k):
        k2 = sum(ki**2 for ki in k)
        r = (pk(k2 ** 0.5) / engine.pm.BoxSize.prod()) ** -0.5
        r[k2 == 0] = 1.0
        return r

    field = engine.pm.generate_whitenoise(seed=1234).c2r()
    eps = field.cnorm() ** 0.5 * 1e-3

    dlineark, whitenoise2 = code.compute(['dlinear_k', 'whitenoise2'], init={'whitenoise': field})
    dlineark2 = field.r2c().apply(lambda k, v: tf1(k) * v)
    assert_allclose(dlineark, dlineark2)
    assert_allclose(field, whitenoise2)

def test_solve_linear_displacement():
    engine = FastPMEngine(pm)
    code = CodeSegment(engine)
    def pk(k):
        p = (k / 0.1) ** -3 * .4e4
        return p

    code.create_linear_field(whitenoise='whitenoise', powerspectrum=pk, dlinear_k='dlinear_k')
    code.solve_linear_displacement(source_k='dlinear_k', s='s')

    field = pm.generate_whitenoise(seed=1234).c2r()

    eps = field.cnorm() ** 0.5 * 1e-3
    check_grad(code, 's', 'whitenoise', init={'whitenoise': field}, eps=eps,
                rtol=1e-2)
    from fastpm.operators import lpt1, lpt2source
    dlin_k, s = code.compute(['dlinear_k', 's'], init={'whitenoise' : field})

    s1_truth = lpt1(dlin_k, engine.q, resampler='cic')
    assert_allclose(s, s1_truth, rtol=1e-5)

def test_solve_lpt():
    pm = ParticleMesh(BoxSize=8.0, Nmesh=(4, 4, 4), dtype='f8')

    engine = FastPMEngine(pm, shift=0.5, B=1)

    code = CodeSegment(engine)
    code.create_linear_field(whitenoise='whitenoise', powerspectrum=pk, dlinear_k='dlinear_k')
    code.solve_lpt(pt=pt, dlinear_k='dlinear_k', aend=1.0, s='s', v='v', s1='s1', s2='s2')

    field = pm.generate_whitenoise(seed=1234).c2r()
    s1, s2 = code.compute(['s1', 's2'], init={'whitenoise' : field})
    dlin_k = code.compute('dlinear_k', init={'whitenoise' : field})

    s1, tape = code.compute('s1', init={'whitenoise' : field}, return_tape=True)

    from fastpm.operators import lpt1, lpt2source
    s1_truth = lpt1(dlin_k, engine.q, resampler='cic')
    dlin2_k = lpt2source(dlin_k)
    s2_truth = lpt1(dlin2_k, engine.q, resampler='cic')

    assert_allclose(s1, s1_truth, rtol=1e-4)
    assert_allclose(s2, s2_truth, rtol=1e-4)

    eps = field.cnorm() ** 0.5 * 1e-3
    check_grad(code, 's1', 'whitenoise', init={'whitenoise': field}, eps=eps,
                rtol=1e-2)

    check_grad(code, 's2', 'whitenoise', init={'whitenoise': field}, eps=eps,
                rtol=1e-2)

    check_grad(code, 's', 'whitenoise', init={'whitenoise': field}, eps=eps,
                rtol=1e-2)

    check_grad(code, 'v', 'whitenoise', init={'whitenoise': field}, eps=eps,
                rtol=1e-2)

def test_solve_fastpm():
    pm = ParticleMesh(BoxSize=8.0, Nmesh=(4, 4, 4), dtype='f8')

    engine = FastPMEngine(pm, shift=0.5, B=1)

    code = CodeSegment(engine)
    code.create_linear_field(whitenoise='whitenoise', powerspectrum=pk, dlinear_k='dlinear_k')
    code.solve_fastpm(pt=pt, dlinear_k='dlinear_k', asteps=[0.1, 1.0], s='s', v='v', s1='s1', s2='s2')
#    code.solve_fastpm(pt=pt, dlinear_k='dlinear_k', asteps=[1.0], s='s', v='v', s1='s1', s2='s2')
    code.paint_simple(s='s', density='density')
    field = pm.generate_whitenoise(seed=1234, unitary=True).c2r()

    eps = field.cnorm() ** 0.5 * 1e-3

    check_grad(code, 's', 'whitenoise', init={'whitenoise': field}, eps=eps,
                rtol=1e-2)
    check_grad(code, 'v', 'whitenoise', init={'whitenoise': field}, eps=eps,
                rtol=1e-2)
    check_grad(code, 'density', 'whitenoise', init={'whitenoise': field}, eps=eps,
                rtol=1e-2)

def test_solve_fastpm_linear():
    pm = ParticleMesh(BoxSize=1024.0, Nmesh=(128, 128, 128), dtype='f8')

    engine = FastPMEngine(pm)

    code = CodeSegment(engine)
    code.create_linear_field(whitenoise='whitenoise', powerspectrum=pk, dlinear_k='dlinear_k')
    code.solve_fastpm(pt=pt, dlinear_k='dlinear_k', asteps=[0.1, 0.5, 1.0], s='s', v='v', s1='s1', s2='s2')
    code.paint_simple(s='s', density='density')
    field = pm.generate_whitenoise(seed=1234, unitary=True).c2r()

    density, dlinear_k, s = code.compute(['density', 'dlinear_k', 's'], init={'whitenoise' : field})
    density_k = density.r2c()
    p_lin= FFTPower(MemoryMesh(dlinear_k), mode='1d')
    p_nonlin = FFTPower(MemoryMesh(density), mode='1d')

    # the simulation shall do a linear growth
    t1 = abs((p_nonlin.power['power'] / p_lin.power['power']) ** 0.5)
    assert_allclose(t1[1:4], 1.0, rtol=5e-2)

def test_generate_2nd_order_source():
    engine = FastPMEngine(pm)
    code = CodeSegment(engine)
    code.r2c(real='source', complex='source_k')
    code.generate_2nd_order_source(source_k='source_k', source2_k='source2_k')
    code.c2r(complex='source2_k', real='source2')
    field = pm.generate_whitenoise(seed=1234).c2r()

    from fastpm.operators import lpt1, lpt2source
    dlin2_k = lpt2source(field.r2c())
    source2_k = code.compute('source2', init={'source' : field}).r2c()

    assert_allclose(dlin2_k[...], source2_k[...], atol=1e-7, rtol=1e-4)

    check_grad(code, 'source2', 'source', init={'source': field}, eps=1e-4,
                rtol=1e-2)

