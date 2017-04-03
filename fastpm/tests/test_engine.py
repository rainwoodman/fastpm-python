from __future__ import print_function
from abopt.vmad2 import CodeSegment, Engine, statement, programme, ZERO, logger

from numpy.testing import assert_raises, assert_array_equal, assert_allclose
from numpy.testing.decorators import skipif
import numpy
import logging

logger.setLevel(level=logging.WARNING)

from abopt.engines.pmesh import ParticleMesh, RealField, ComplexField, check_grad

from ..engine import FastPMEngine

pm = ParticleMesh(BoxSize=1.0, Nmesh=(4, 4, 4), dtype='f8')

def test_force():
    def transfer(k): return 2.0
    engine = FastPMEngine(pm)
    code = CodeSegment(engine)
    code.r2c(real='r', complex='c')
    code.force(density='c', s='s', force='force')

    s = engine.q * 0.0 + 0.1
    field = pm.generate_whitenoise(seed=1234).c2r()

    check_grad(code, 'force', 's', init={'r': field, 's': s}, eps=1e-4,
                rtol=1e-2)

    check_grad(code, 'force', 'r', init={'r': field, 's': s}, eps=1e-4,
                rtol=1e-2)

def test_solve_linear_displacement():
    engine = FastPMEngine(pm)
    code = CodeSegment(engine)
    code.r2c(real='source', complex='source_k')
    code.solve_linear_displacement(source_k='source_k', s='s')

    field = pm.generate_whitenoise(seed=1234).c2r()

    check_grad(code, 's', 'source', init={'source': field}, eps=1e-4,
                rtol=1e-2)

def test_solve_lpt():
    pm = ParticleMesh(BoxSize=128.0, Nmesh=(64, 64, 64), dtype='f8')
    def pk(k):
        p = (k / 0.1) ** -2 * .4e4
        return p

    engine = FastPMEngine(pm)
    code = CodeSegment(engine)
    code.create_linear_field(whitenoise='whitenoise', powerspectrum=pk, dlinear_k='dlinear_k')
    code.solve_lpt(dlinear_k='dlinear_k', s1='s1', s2='s2')

    field = pm.generate_whitenoise(seed=1234).c2r()
    s1, s2 = code.compute(['s1', 's2'], init={'whitenoise' : field})
    dlin_k = code.compute('dlinear_k', init={'whitenoise' : field})

    from fastpm.operators import lpt1, lpt2source
    s1_truth = lpt1(dlin_k, engine.q, method='cic')
    dlin2_k = lpt2source(dlin_k)
    s2_truth = lpt1(dlin2_k, engine.q, method='cic')

    print(abs(s1 - s1_truth).max())
    print(abs(s2 - s2_truth).max())
    print(s1[0], s1_truth[0])
    print(s2[0], s2_truth[0])

def test_generate_2nd_order_source():
    engine = FastPMEngine(pm)
    code = CodeSegment(engine)
    code.r2c(real='source', complex='source_k')
    code.generate_2nd_order_source(source_k='source_k', source2_k='source2_k')
    code.c2r(complex='source2_k', real='source2')
    field = pm.generate_whitenoise(seed=1234).c2r()

    check_grad(code, 'source2', 'source', init={'source': field}, eps=1e-4,
                rtol=1e-2)

