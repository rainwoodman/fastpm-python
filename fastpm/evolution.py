from __future__ import print_function
import numpy
import logging

from abopt.vmad import VM

from pmesh.pm import ParticleMesh, RealField

from fastpm.perturbation import PerturbationGrowth

import fastpm.operators as operators

class Evolution(VM):
    def __init__(self, pm, B=1, shift=0, dtype='f8'):
        self.pm = pm
        self.fpm = ParticleMesh(Nmesh=pm.Nmesh * B, BoxSize=pm.BoxSize, dtype=dtype, comm=pm.comm)
        self.q = operators.create_grid(self.pm, shift=shift, dtype=dtype)
        VM.__init__(self)

    @VM.microcode(aout=['y'], ain=['x'])
    def CopyVariable(self, x, y):
        if hasattr(x, 'copy'):
            y[...] = x.copy()
        else:
            y[...] = 1.0 * x

    @CopyVariable.grad
    def _(self, _y, _x):
        _x[...] = _y

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

    @VM.microcode(aout=['s', 'p'], ain=['dlin_k'])
    def Displace(self, dlin_k, s, p, D1, v1, D2, v2):
        q = self.q
        dx1 = operators.lpt1(dlin_k, q)
        source = operators.lpt2source(dlin_k)
        dx2 = operators.lpt1(source, q)
        s[...] = D1 * dx1 + D2 * dx2
        p[...] = v1 * dx1 + v2 * dx2

    @Displace.grad
    def _(self, _dlin_k, dlin_k, _s, _p, D1, v1, D2, v2):
        q = self.q
        grad_dx1 = _p * v1 + _s * D1
        grad_dx2 = _p * v2 + _s * D2

        if grad_dx1 is not VM.Zero:
            gradient = operators.lpt1_gradient(self.pm, q, grad_dx1)
        else:
            gradient = VM.Zero

        if grad_dx2 is not VM.Zero:
            gradient_lpt2source = operators.lpt1_gradient(self.pm, q, grad_dx2)
            gradient[...] +=  operators.lpt2source_gradient(dlin_k, gradient_lpt2source)

        _dlin_k[...] = gradient

    @VM.microcode(aout=['y'], ain=['x'])
    def Multiply(self, x, f, y):
        y[...] = x * f

    @Multiply.grad
    def _(self, _x, _y, f):
        _x[...] = _y * f

    @VM.microcode(aout=['f'], ain=['s'])
    def Force(self, s, factor, f):
        density_factor = 1.0 * self.fpm.Nmesh.prod() / self.pm.Nmesh.prod()
        x = s + self.q
        f[...] = operators.gravity(x, self.fpm, factor=density_factor * factor, f=None)

    @Force.grad
    def _(self, _s, s, _f, factor):
        density_factor = 1.0 * self.fpm.Nmesh.prod() / self.pm.Nmesh.prod()

        if _f is VM.Zero:
            _s[...] = VM.Zero
        else:
            x = s + self.q
            _s[...] = operators.gravity_gradient(x, self.pm, density_factor * factor, _f)

    @VM.microcode(aout=['mesh'], ain=['s'])
    def Paint(self, s, mesh):
        x = s + self.q
        mesh[...] = self.pm.create(mode='real')
        layout = self.pm.decompose(x)
        mesh[...].paint(x, layout=layout, hold=False)
        # to have 1 + \delta on the mesh
        mesh[...][...] *= 1.0 * mesh.pm.Nmesh.prod() / self.pm.Nmesh.prod()

    @Paint.grad
    def _(self, _s, _mesh, s):
        if _mesh is VM.Zero:
            _s = VM.Zero
        else:
            x = s + self.q
            layout = _mesh.pm.decompose(x)
            _s[...], junk = _mesh.paint_gradient(x, layout=layout, out_mass=False)
            _s[...][...] *= _mesh.pm.Nmesh.prod() / self.pm.Nmesh.prod()

    @VM.microcode(aout=['mesh'], ain=['mesh'])
    def Transfer(self, mesh, transfer):
        mesh.r2c(out=Ellipsis)\
               .apply(lambda k, v: transfer(sum(ki ** 2 for ki in k) ** 0.5) * v, out=Ellipsis)\
               .c2r(out=Ellipsis)

    @Transfer.grad
    def _(self, _mesh, transfer):
        _mesh.c2r_gradient(out=Ellipsis)\
               .apply(lambda k, v: transfer(sum(ki ** 2 for ki in k) ** 0.5) * v, out=Ellipsis)\
               .r2c_gradient(out=Ellipsis)

    @VM.microcode(aout=['residual'], ain=['mesh'])
    def Residual(self, mesh, data_x, sigma_x, residual):
        diff = mesh - data_x
        diff[...] /= sigma_x[...]
        residual[...] = diff

    @Residual.grad
    def _(self, _mesh, _residual, sigma_x):
        _mesh[...] = _residual.copy()
        _mesh[...][...] /= sigma_x

    @VM.microcode(aout=['R'], ain=['C'])
    def C2R(self, R, C):
        R[...] = C.c2r()

    @C2R.grad
    def _(self, _R, _C):
        _C[...] = _R.c2r_gradient().decompress_gradient()

    @VM.microcode(aout=['mesh'], ain=['mesh'])
    def Resample(self, mesh, Neff):
        def _Resample_filter(k, v):
            k0s = 2 * numpy.pi / v.BoxSize
            mask = numpy.bitwise_and.reduce([abs(ki) <= Neff//2 * k0 for ki, k0 in zip(k, k0s)])
            return v * mask

        mesh.r2c(out=Ellipsis).apply(_Resample_filter, out=Ellipsis).c2r(out=Ellipsis)


    @Resample.grad
    def _(self, _mesh, Neff):
        def _Resample_filter(k, v):
            k0s = 2 * numpy.pi / v.BoxSize
            mask = numpy.bitwise_and.reduce([abs(ki) <= Neff//2 * k0 for ki, k0 in zip(k, k0s)])
            return v * mask

        _mesh.c2r_gradient().apply(_Resample_filter, out=Ellipsis).r2c_gradient(out=Ellipsis)

    @VM.microcode(aout=['chi2'], ain=['variable'])
    def Chi2(self, chi2, variable):
        variable = variable * variable
        if isinstance(variable, RealField):
            chi2[...] = variable.csum()
        else:
            chi2[...] = self.pm.comm.allreduce(variable.sum(dtype='f8'))

    @Chi2.grad
    def _(self, _variable, _chi2, variable):
        _variable[...] = variable * (2 * _chi2)

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

class LPT(Evolution):
    def __init__(self, pm, shift=0):
        Evolution.__init__(self, pm, B=1, shift=shift)

    def simulation(self, cosmo, aend, order, mesh='mesh', dlin_k='dlin_k'):
        pt = PerturbationGrowth(cosmo)
        code = Evolution.code(self)
        if order == 1:
            code.Displace(D1=pt.D1(aend), 
                          v1=pt.f1(aend) * pt.D1(aend) * aend ** 2 * pt.E(aend),
                          D2=0,
                          v2=0,
                          dlin_k=dlin_k,
                         )
        if order == 2:
            code.Displace(D1=pt.D1(aend), 
                          v1=pt.f1(aend) * pt.D1(aend) * aend ** 2 * pt.E(aend),
                          D2=pt.D2(aend),
                          v2=pt.f2(aend) * pt.D2(aend) * aend ** 2 * pt.E(aend),
                          dlin_k=dlin_k
                         )
        code.Paint(mesh=mesh)
        return code

class KickDriftKick(Evolution):
    def __init__(self, pm, B=1, shift=0):
        Evolution.__init__(self, pm, B=B, shift=shift)

    def simulation(self, cosmo, astart, aend, Nsteps, mesh='mesh', dlin_k='dlin_k'):
        pt = PerturbationGrowth(cosmo)
        code = Evolution.code(self)
        code.Displace(D1=pt.D1(astart), 
                      v1=pt.f1(astart) * pt.D1(astart) * astart ** 2 * pt.E(astart),
                      D2=pt.D2(astart),
                      v2=pt.f2(astart) * pt.D2(astart) * astart ** 2 * pt.E(astart),
                      dlin_k=dlin_k)
        code.Force(factor=1.5 * pt.Om0)

        a = numpy.linspace(astart, aend, Nsteps + 1, endpoint=True)
        def K(ai, af, ar):
            return 1 / (ar ** 2 * pt.E(ar)) * (pt.Gf(af) - pt.Gf(ai)) / pt.gf(ar)
        def D(ai, af, ar):
            return 1 / (ar ** 3 * pt.E(ar)) * (pt.Gp(af) - pt.Gp(ai)) / pt.gp(ar)

        for ai, af in zip(a[:-1], a[1:]):
            ac = (ai * af) ** 0.5

            code.Kick(dda=K(ai, ac, ai))
            code.Drift(dyyy=D(ai, ac, ac))
            code.Drift(dyyy=D(ac, af, ac))
            code.Force(factor=1.5 * pt.Om0)
            code.Kick(dda=K(ac, af, af))

        code.Paint(mesh=mesh)
        return code


