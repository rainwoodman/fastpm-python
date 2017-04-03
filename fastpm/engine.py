from __future__ import print_function
import numpy
import logging

from abopt.engines.pmesh import (
        ParticleMeshEngine,
        ZERO, Literal,
        CodeSegment,
        programme,
        statement,
        ParticleMesh, RealField, ComplexField
        )

from fastpm.perturbation import PerturbationGrowth

class FastPMEngine(ParticleMeshEngine):
    def __init__(self, pm, B=1):
        ParticleMeshEngine.__init__(self, pm)
        fpm = ParticleMesh(Nmesh=pm.Nmesh * B, BoxSize=pm.BoxSize, dtype=pm.dtype, comm=pm.comm)
        self.fengine = ParticleMeshEngine(fpm, q=self.q)

    @programme(ain=['whitenoise'], aout=['dlinear_k'])
    def create_linear_field(engine, whitenoise, powerspectrum, dlinear_k):
        code = CodeSegment(engine)
        code.r2c(real=whitenoise, complex=dlinear_k)
        def tf(k):
            k2 = sum(ki**2 for ki in k)
            r = (powerspectrum(k2 ** 0.5) * (1.0 / engine.pm.BoxSize).prod()) ** 0.5
            r[k2 == 0] = 1.0
            return r
        code.transfer(complex=dlinear_k, tf=tf)
        return code

    @programme(ain=['source_k'], aout=['s'])
    def solve_linear_displacement(engine, source_k, s):
        code = CodeSegment(engine)
        code.decompose(s=Literal(ZERO), layout='layout')
        code.defaults['s'] = numpy.zeros_like(engine.q)
        for d in range(engine.pm.ndim):
            def tf(k, d=d):
                k2 = sum(ki ** 2 for ki in k)
                mask = k2 == 0
                k2[mask] = 1.0
                return 1j * k[d] / k2 * ~mask
            code.assign(x='source_k', y='disp1_k')
            code.transfer(complex='disp1_k', tf=tf)
            code.c2r(complex='disp1_k', real='disp1')
            code.readout(mesh='disp1', value='s1', s=Literal(ZERO))
            code.assign_component(attribute='s', value='s1', dim=d)
        return code

    @statement(ain=['x1', 'x2'], aout='y')
    def bilinear(engine, x1, c1, x2, c2, y):
        y[...] = x1 * c1 + x2 * c2

    @bilinear.defvjp
    def _(engine, _x1, _x2, _y, c1, c2):
        _x1[...] = _y * c1
        _x2[...] = _y * c2

    @programme(ain=['dlinear_k'], aout=['s', 'v'])
    def solve_lpt(engine, cosmo, aend, dlinear_k, s, v):
        code = CodeSegment(engine)
        pt = PerturbationGrowth(cosmo)
        code.solve_linear_displacement(source_k='dlinear_k', s='s1')
        code.generate_2nd_order_source(source_k='dlinear_k', source2_k='source2_k')
        code.solve_linear_displacement(source_k='source2_k', s='s2')

        code.bilinear(x1='s1', c1=pt.D1(aend),
                      x2='s2', c2=pt.D2(aend),
                       y='s')

        code.bilinear(x1='s1', c1=pt.f1(aend) * aend ** 2 * pt.E(aend) * pt.D1(aend),
                      x2='s2', c2=pt.f2(aend) * aend ** 2 * pt.E(aend) * pt.D2(aend),
                       y='s')
        return code

    @programme(ain=['dlinear_k'], aout=['s', 'v'])
    def solve_fastpm(engine, cosmo, asteps, dlinear_k, s, v):
        pt = PerturbationGrowth(cosmo)
        code = CodeSegment(engine)
        code.solve_lpt(cosmo=cosmo, aend=asteps[0], dlinear_k=dlinear_k, s='s', v='v')
        engine
    @programme(ain=['source_k'], aout=['source2_k'])
    def generate_2nd_order_source(engine, source_k, source2_k):
        code = CodeSegment(engine)
        if engine.pm.ndim < 3:
            code.defaults['source2_k'] = engine.pm.create(mode='complex', zeros=True)
            return code

        code.defaults['source2'] = engine.pm.create(mode='real', zeros=True)

        D1 = [1, 2, 0]
        D2 = [2, 0, 1]
        varname = ['var_%d' % d for d in range(engine.pm.ndim)]
        for d in range(engine.pm.ndim):
            def tf(k, d=d):
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
            def tf(k, d=d):
                k2 = sum(ki ** 2 for ki in k)
                mask = k2 == 0
                k2[mask] = 1.0
                return 1j * k[D1[d]] * 1j * k[D2[d]] / k2 * ~mask
            code.assign(x='source_k', y='phi_ij')
            code.transfer(complex='phi_ij', tf=tf)
            code.c2r(complex='phi_ij', real='phi_ij')
            code.multiply(x1='phi_ij', x2='phi_ij', y='phi_ij')
            code.multiply(x1='phi_ij', x2=Literal(-1.0), y='phi_ij')
            code.add(x1='source2', x2='phi_ij', y='source2')

        code.multiply(x1='source2', x2=Literal(3.0 /7), y='source2')
        code.r2c(real='source2', complex='source2_k')
        return code

    @programme(aout=['force'], ain=['s'])
    def force_paint(engine, force, s):
        code = CodeSegment(engine.fengine)
        code.paint(s='s', d

    @programme(aout=['force'], ain=['density_k', 's'])
    def force(engine, force, density_k, s):
        code = CodeSegment(engine.fengine)
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

