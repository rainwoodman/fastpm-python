from __future__ import print_function
from abopt.vmad2 import CodeSegment, Engine, statement, programme, ZERO, logger

from numpy.testing import assert_raises, assert_array_equal, assert_allclose
from numpy.testing.decorators import skipif
import numpy
import logging

logger.setLevel(level=logging.WARNING)

from abopt.engines.pmesh import ParticleMesh, RealField, ComplexField, check_grad

from ..engine import FastPMEngine

pm = ParticleMesh(BoxSize=1.0, Nmesh=(4, 4), dtype='f8')

def test_force():
    def transfer(k): return 2.0
    engine = FastPMEngine(pm)
    code = CodeSegment(engine)
    code.r2c(real='r', complex='c')
    code.force(density='c', s='s', force='force')

    s = engine.q * 0.0 + 0.1
    field = pm.generate_whitenoise(seed=1234).c2r()

    cshape = engine.pm.comm.allreduce(engine.q.shape[0]), engine.q.shape[1]

    check_grad(code, 'force', 's', init={'r': field, 's': s}, eps=1e-4,
                rtol=1e-2)

    check_grad(code, 'force', 'r', init={'r': field, 's': s}, eps=1e-4,
                rtol=1e-2)
