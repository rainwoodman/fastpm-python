from __future__ import print_function
import numpy
import logging

from abopt.vmad import VM, Zero
from pmesh.pm import ParticleMesh, RealField

from fastpm.perturbation import PerturbationGrowth

import fastpm.operators as operators

from fastpm.models import PMesh, MPINumeric, LPT

class Evolution(PMesh, LPT, MPINumeric, VM):
    def __init__(self, pm, B=1, shift=0, dtype='f8'):
        self.fpm = ParticleMesh(Nmesh=pm.Nmesh * B, BoxSize=pm.BoxSize, dtype=dtype, comm=pm.comm)
        self.pm = pm
        self.q = operators.create_grid(self.pm, shift=shift, dtype=dtype)
        N = pm.comm.allreduce(len(self.q))

        self.mean_number_count = 1.0 * N / (1.0 * self.fpm.Nmesh.prod())

        VM.__init__(self)
        PMesh.__init__(self, self.pm)
        MPINumeric.__init__(self, self.pm.comm)
        LPT.__init__(self, self.pm, self.q)

    @VM.microcode(ain=['dlin_k'], aout=['prior'])
    def Prior(self, dlin_k, prior, powerspectrum):
        prior[...] = dlin_k.cnorm(
                    metric=lambda k: 1 / (powerspectrum(k) / dlin_k.BoxSize.prod())
                    )
    @Prior.grad
    def _(self, _dlin_k, dlin_k, powerspectrum, _prior):
        _dlin_k[...] = dlin_k.cnorm_gradient(_prior,
                    metric=lambda k: 1 / (powerspectrum(k) / dlin_k.BoxSize.prod()),
                    )

    @VM.microcode(aout=['f'], ain=['s'])
    def Force(self, s, factor, f):
        x = s + self.q
        f[...] = operators.gravity(x, self.fpm, factor=1.0 * factor / self.mean_number_count, f=None)

    @Force.grad
    def _(self, _s, s, _f, factor):
        if _f is Zero:
            _s[...] = Zero
        else:
            x = s + self.q
            _s[...] = operators.gravity_gradient(x, self.fpm, 1.0 * factor / self.mean_number_count, _f)

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
            self.Force(factor=1.5 * pt.Om0)
            self.Kick(dda=K(ac, af, af))

        self.Paint(mesh=mesh)
