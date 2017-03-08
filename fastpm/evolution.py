from __future__ import print_function
import numpy
import logging

from abopt.vmad import VM, Zero, microcode
from pmesh.pm import ParticleMesh, RealField

from fastpm.perturbation import PerturbationGrowth

import fastpm.operators as operators

from fastpm.models import MPINumeric
from fastpm.pmeshvm import ParticleMeshVM

class Evolution(MPINumeric, ParticleMeshVM):
    def __init__(self, pm, B=1, shift=0, dtype='f8'):
        self.fpm = ParticleMesh(Nmesh=pm.Nmesh * B, BoxSize=pm.BoxSize, dtype=dtype, comm=pm.comm)
        q = operators.create_grid(pm, shift=shift, dtype=dtype)
        N = pm.comm.allreduce(len(q))

        self.mean_number_count = 1.0 * N / (1.0 * self.fpm.Nmesh.prod())

        ParticleMeshVM.__init__(self, pm, q)
        MPINumeric.__init__(self, pm.comm)

    @microcode(ain=['X'], aout=['dlin_k'])
    def MakeInitialCondition(self, X, powerspectrum, dlin_k):
        def filter(k, v):
            return (powerspectrum(k) / v.BoxSize.prod()) ** 0.5 * v
        dlin_k[...] = X.r2c().apply(filter, out=Ellipsis)

    @MakeInitialCondition.defvjp
    def _(self, _dlin_k, _X):
        def filter(k, v):
            return (powerspectrum(k) / v.BoxSize.prod()) ** 0.5 * v
        _X[...] = _dlin_k.r2c_gradient().apply(filter, out=Ellipsis)

    @microcode(ain=['dlin_k'], aout=['prior'])
    def Prior(self, dlin_k, prior, powerspectrum):
        prior[...] = dlin_k.cnorm(
                    metric=lambda k: 1 / (powerspectrum(k) / dlin_k.BoxSize.prod())
                    )
    @Prior.grad
    def _(self, _dlin_k, dlin_k, powerspectrum, _prior):
        _dlin_k[...] = dlin_k.cnorm_gradient(_prior,
                    metric=lambda k: 1 / (powerspectrum(k) / dlin_k.BoxSize.prod()),
                    )

    @VM.microcode(aout=['delta_k'], ain=['s'])
    def ForcePaint(self, s, delta_k):
        pm = self.fpm
        x = s + self.q
        field = pm.create(mode="real")
        layout = pm.decompose(x)
        field.paint(x, layout=layout, hold=False)
        delta_k[:] = field.r2c(out=Ellipsis)
        delta_k[:][...] /= self.mean_number_count

    @ForcePaint.grad
    def _(self, _delta_k, _s, s):
        x = s + self.q
        pm = self.fpm
        layout = pm.decompose(x)

        _field = _delta_k.r2c_gradient()
        _field[...] /= self.mean_number_count
        _x, _mass = _field.paint_gradient(x, layout=layout, out_mass=False)
        _s[...] = _x

    @VM.microcode(aout=['f'], ain=['delta_k', 's'])
    def Force(self, s, delta_k, factor, f):
        x = s + self.q
        pm = self.fpm
        layout = pm.decompose(x)

        f[:] = numpy.empty_like(x)
        for d in range(delta_k.ndim):
            force_d = delta_k.apply(operators.laplace_kernel) \
                      .apply(operators.diff_kernel(d), out=Ellipsis) \
                      .c2r(out=Ellipsis)
            force_d.readout(x, layout=layout, out=f[:][..., d])

        f[:][...] *= factor

    @Force.grad
    def _(self, _f, _s, _delta_k, delta_k, factor, s):
        x = s + self.q
        pm = self.fpm

        layout = pm.decompose(x)

        _delta_k[:] = delta_k * 0
        _s[:] = numpy.zeros_like(s)

        for d in range(delta_k.ndim):
            force_d = delta_k.apply(operators.laplace_kernel) \
                      .apply(operators.diff_kernel(d), out=Ellipsis) \
                      .c2r(out=Ellipsis)

            # factor because of the inplace multiplication
            _force_d, _x_d = force_d.readout_gradient(
                x, btgrad=_f[:, d] * factor, layout=layout)

            _delta_k_d = _force_d.c2r_gradient(out=Ellipsis) \
                            .apply(operators.laplace_kernel, out=Ellipsis) \
                            .apply(operators.diff_kernel(d, conjugate=True), out=Ellipsis)
            _delta_k[:][...] += _delta_k_d
            _s[:][...] += _x_d

    @VM.microcode(aout=['p'], ain=['f', 'p'])
    def Kick(self, f, p, dda):
        p[...] += f * dda

    @Kick.grad
    def _(self, _f, _p, dda):
        _f[...] = _p * dda

    @VM.microcode(aout=['s'], ain=['p', 's'])
    def Drift(self, p, s, dyyy):
        s[...] += p * dyyy

    @Drift.grad
    def _(self, _p, _s, dyyy):
        _p[...] = _s * dyyy

    @microcode(aout=['s', 'p'], ain=['dlin_k'])
    def LPTDisplace(self, dlin_k, s, p, D1, v1, D2, v2):
        q = self.q
        dx1 = operators.lpt1(dlin_k, q)
        source = operators.lpt2source(dlin_k)
        dx2 = operators.lpt1(source, q)
        s[...] = D1 * dx1 + D2 * dx2
        p[...] = v1 * dx1 + v2 * dx2

    @LPTDisplace.defvjp
    def _(self, _dlin_k, dlin_k, _s, _p, D1, v1, D2, v2):
        pm = self.pm
        q = self.q

        grad_dx1 = _p * v1 + _s * D1
        grad_dx2 = _p * v2 + _s * D2

        if grad_dx1 is not Zero:
            gradient = operators.lpt1_gradient(pm, q, grad_dx1)
        else:
            gradient = Zero

        if grad_dx2 is not Zero:
            gradient_lpt2source = operators.lpt1_gradient(pm, q, grad_dx2)
            gradient[...] +=  operators.lpt2source_gradient(dlin_k, gradient_lpt2source)

        _dlin_k[...] = gradient

    @VM.programme(aout=['mesh'], ain=['dlin_k'])
    def LPTSimulation(self, cosmo, aend, order, mesh, dlin_k):
        pt = PerturbationGrowth(cosmo)
        if order == 1:
            self.LPTDisplace(D1=pt.D1(aend), 
                          v1=pt.f1(aend) * pt.D1(aend) * aend ** 2 * pt.E(aend),
                          D2=0,
                          v2=0,
                          dlin_k=dlin_k,
                         )
        if order == 2:
            self.LPTDisplace(D1=pt.D1(aend), 
                          v1=pt.f1(aend) * pt.D1(aend) * aend ** 2 * pt.E(aend),
                          D2=pt.D2(aend),
                          v2=pt.f2(aend) * pt.D2(aend) * aend ** 2 * pt.E(aend),
                          dlin_k=dlin_k
                         )
        self.Paint(mesh=mesh)

    @VM.programme(aout=['mesh'], ain=['dlin_k'])
    def KDKSimulation(self, cosmo, astart, aend, Nsteps, mesh, dlin_k):
        pt = PerturbationGrowth(cosmo)
        self.LPTDisplace(D1=pt.D1(astart), 
                      v1=pt.f1(astart) * pt.D1(astart) * astart ** 2 * pt.E(astart),
                      D2=pt.D2(astart),
                      v2=pt.f2(astart) * pt.D2(astart) * astart ** 2 * pt.E(astart),
                      dlin_k=dlin_k)

        self.ForcePaint()
        self.Force(factor=1.5 * pt.Om0)

        a = numpy.linspace(astart, aend, Nsteps + 1, endpoint=True)
        def K(ai, af, ar):
            return 1 / (ar ** 2 * pt.E(ar)) * (pt.Gf(af) - pt.Gf(ai)) / pt.gf(ar)
        def D(ai, af, ar):
            return 1 / (ar ** 3 * pt.E(ar)) * (pt.Gp(af) - pt.Gp(ai)) / pt.gp(ar)

        for ai, af in zip(a[:-1], a[1:]):
            ac = (ai * af) ** 0.5

            self.Kick(dda=K(ai, ac, ai))
            self.Drift(dyyy=D(ai, ac, ac))
            self.Drift(dyyy=D(ac, af, ac))
            self.ForcePaint()
            self.Force(factor=1.5 * pt.Om0)
            self.Kick(dda=K(ac, af, af))

        self.Paint(mesh=mesh)
