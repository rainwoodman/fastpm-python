from __future__ import print_function
import numpy
import logging

from abopt.engines.pmesh import (
        ParticleMeshEngine,
        ZERO,
        CodeSegment,
        programme,
        statement,
        ParticleMesh, RealField, ComplexField
        )

class FastPMEngine(ParticleMeshEngine):
    def __init__(self, pm, B=1):
        ParticleMeshEngine.__init__(self, pm)
        fpm = ParticleMesh(Nmesh=pm.Nmesh * B, BoxSize=pm.BoxSize, dtype=pm.dtype, comm=pm.comm)
        self.fengine = ParticleMeshEngine(fpm, q=self.q)

    @programme(ain=['whitenoise'], aout=['dlinear_k'])
    def create_linear_field(engine, whitenoise, powerspectrum, dlinear_k):
        code = CodeSegment(engine)
        code.r2c(real=whitenoise, complex=dlinear_k)
        def tr(k):
            k2 = sum(ki**2 for ki in k)
            return powerspectrum(k2 ** 0.5)
        code.transfer(complex=dlinear_k, tf=tf)
        return code

    @programme(ain=['source_k'], aout=['s'])
    def solve_linear_displacement(engine, source_k, s):
        code = CodeSegment(engine)
        code.decompose(s=ZERO, layout='layout')
        code.defaults['s'] = numpy.zeros_like(engine.q)
        for d in range(engine.pm.ndim):
            def tf(k):
                k2 = sum(ki ** 2 for ki in k)
                mask = k2 == 0
                k2[mask] = 1.0
                return 1j * k[d] / k2 * ~mask
            code.assign(x='source', y='disp1')
            code.transfer(complex='disp1', tf=tf)
            code.c2r(complex='disp1', real='disp1')
            code.readout(mesh='disp1', value='s1', s=ZERO)
            code.assign_componenet(attribute='s', value='s1', dim=d)
        return code

    @programme(ain=['source_k'], aout=['source2_k'])
    def generate_2nd_order_source(engine, source_k, source2_k):
        code = CodeSegment(engine)
        code.defaults['source2_k'] = engine.pm.create(mode='complex') * 0.0
        if engine.pm.ndim < 3:
            return code

        code.defaults['source2'] = engine.pm.create(mode='real') * 0.0
        code.defaults['source2_factor'] = 3.0 / 7.0

        D1 = [1, 2, 0]
        D2 = [2, 0, 1]
        varname = ['var_%d' % d for d in range(engine.pm.ndim)]
        for d in range(engine.pm.ndim):
            def tf(k):
                k2 = sum(ki ** 2 for ki in k)
                mask = k2 == 0
                k2[mask] = 1.0
                return 1j * k[d] * 1j * k[d] / k2 * ~mask
            code.assign(x='source_k', y=varname[d])
            code.transfer(complex=varname[d], tf=tf)
            code.c2r(complex=varname[d], real=varname[d])

        for d in range(engine.pm.ndim):
            code.multiply(x1=varname[D1[d]], x2=varname[D2[d]], y='phi_ii')
            code.add(x1='source2', x2='phi_ii', y='source2')

        for d in range(engine.pm.ndim):
            def tf(k):
                k2 = sum(ki ** 2 for ki in k)
                mask = k2 == 0
                k2[mask] = 1.0
                return 1j * 1j * k[D1[d]] * 1j * k[D2[d]] / k2 * ~mask
            code.assign(x='source_k', y='phi_ij')
            code.transfer(complex='phi_ij', tf=tf)
            code.c2r(complex='phi_ij', real='phi_ij')
            code.add(x1='source2', x2='phi_ij', y='source2')

        code.multiply(x1='source2', x2='source2_factor', y='source2')
        code.c2r(real='source2', complex='source2_k')
        return code

    @programme(aout=['force'], ain=['density', 's'])
    def force(engine, force, density, s):
        code = CodeSegment(engine)
        code.defaults['force'] = numpy.zeros_like(engine.q)
        code.decompose(s=s, layout='layout')
        for d in range(engine.pm.ndim):
            def tf(k):
                k2 = sum(ki ** 2 for ki in k)
                mask = k2 == 0
                k2[mask] = 1.0
                return 1j * k[d] / k2 * ~mask
            code.assign(x='density', y='complex')
            code.transfer(complex='complex', tf=tf)
            code.c2r(complex='complex', real='real')
            code.readout(value='force1', mesh='real', s=s, layout='layout')
            code.assign_component(attribute='force', dim=d, value='force1')
        return code

