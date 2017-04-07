from __future__ import print_function
from abopt.vmad2 import CodeSegment, Engine, statement, programme, ZERO, logger

from numpy.testing import assert_raises, assert_array_equal, assert_allclose
from numpy.testing.decorators import skipif
import numpy
import logging

logger.setLevel(level=logging.WARNING)

from abopt.engines.pmesh import ParticleMesh, RealField, ComplexField, check_grad

from ..engine import FastPMEngine

from ..perturbation import PerturbationGrowth

pm = ParticleMesh(BoxSize=1.0, Nmesh=(4, 4, 4), dtype='f8')

def test_force():
    engine = FastPMEngine(pm)
    code = CodeSegment(engine)
    code.r2c(real='r', complex='c')
    code.decompose(s='s', layout='layout')
    code.force_compute(density_k='c', s='s', layout='layout', force='force', force_factor=1.0)

    s = engine.q * 0.0 + 0.2
    field = engine.fengine.pm.generate_whitenoise(seed=1234).c2r()
    check_grad(code, 'force', 's', init={'r': field, 's': s}, eps=1e-4,
                rtol=1e-2)

    eps = field.cnorm() ** 0.5 * 1e-3
    check_grad(code, 'force', 'r', init={'r': field, 's': s}, eps=eps,
                rtol=1e-3)

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

    s1_truth = lpt1(dlin_k, engine.q, method='cic')
    assert_allclose(s, s1_truth, rtol=1e-5)

def test_solve_lpt():
    pm = ParticleMesh(BoxSize=128.0, Nmesh=(4, 4, 4), dtype='f8')
    def pk(k):
        p = (k / 0.1) ** -3 * .4e4
        return p
    cosmo = lambda : None

    cosmo.Om0 = 0.3
    cosmo.Ode0 = 0.7
    cosmo.Ok0 = 0.0
    pt = PerturbationGrowth(cosmo)

    engine = FastPMEngine(pm)
    engine.q[...] += 0.5 * pm.BoxSize / pm.Nmesh

    code = CodeSegment(engine)
    code.create_linear_field(whitenoise='whitenoise', powerspectrum=pk, dlinear_k='dlinear_k')
    code.solve_lpt(pt=pt, dlinear_k='dlinear_k', aend=1.0, s='s', v='v', s1='s1', s2='s2')

    field = pm.generate_whitenoise(seed=1234).c2r()
    s1, s2 = code.compute(['s1', 's2'], init={'whitenoise' : field})
    dlin_k = code.compute('dlinear_k', init={'whitenoise' : field})

    s1, tape = code.compute('s1', init={'whitenoise' : field}, return_tape=True)

    from fastpm.operators import lpt1, lpt2source
    s1_truth = lpt1(dlin_k, engine.q, method='cic')
    dlin2_k = lpt2source(dlin_k)
    s2_truth = lpt1(dlin2_k, engine.q, method='cic')

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
    pm = ParticleMesh(BoxSize=128.0, Nmesh=(4, 4, 4), dtype='f8')
    def pk(k):
        p = (k / 0.1) ** -3 * .4e4
        return p
    cosmo = lambda : None

    cosmo.Om0 = 0.3
    cosmo.Ode0 = 0.7
    cosmo.Ok0 = 0.0
    pt = PerturbationGrowth(cosmo)

    engine = FastPMEngine(pm)
    engine.q[...] += 0.5 * pm.BoxSize / pm.Nmesh

    code = CodeSegment(engine)
    code.create_linear_field(whitenoise='whitenoise', powerspectrum=pk, dlinear_k='dlinear_k')
    code.solve_fastpm(pt=pt, dlinear_k='dlinear_k', asteps=[0.1, 0.5, 1.0], s='s', v='v', s1='s1', s2='s2')

    field = pm.generate_whitenoise(seed=1234).c2r()

    eps = field.cnorm() ** 0.5 * 1e-3

    check_grad(code, 's', 'whitenoise', init={'whitenoise': field}, eps=eps,
                rtol=1e-2)
    check_grad(code, 'v', 'whitenoise', init={'whitenoise': field}, eps=eps,
                rtol=1e-2)

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

