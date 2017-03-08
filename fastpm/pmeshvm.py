import numpy
from abopt.vmad import Zero, VM

class ParticleMeshVM(VM):
    def __init__(self, pm, q):
        self.pm = pm
        self.q = q

    @VM.microcode(aout=['mesh'], ain=['mesh'])
    def Transfer(self, mesh, transfer):
        mesh.r2c(out=Ellipsis)\
               .apply(lambda k, v: transfer(sum(ki ** 2 for ki in k) ** 0.5) * v, out=Ellipsis)\
               .c2r(out=Ellipsis)

    @Transfer.grad
    def _(self, _mesh, transfer):
        if _mesh is Zero: return
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
        if _residual is Zero:
            _mesh = Zero
        else:
            _mesh[...] = _residual.copy()
            _mesh[...][...] /= sigma_x

    @VM.microcode(aout=['R'], ain=['C'])
    def C2R(self, R, C):
        R[...] = C.c2r()

    @C2R.grad
    def _(self, _R, _C):
        if _R is Zero:
            _C[...] = Zero
        else:
            _C[...] = _R.c2r_gradient()

    @VM.microcode(aout=['C'], ain=['R'])
    def R2C(self, C, R):
        C[...] = R.r2c()

    @VM.microcode(aout=['C'], ain=['C'])
    def Decompress(self, C):
        return

    @Decompress.grad
    def _(self, _C):
        _C.decompress_gradient(out=Ellipsis)

    @R2C.grad
    def _(self, _C, _R):
        if _C is Zero:
            _R[...] = Zero
        else:
            _R[...] = _C.r2c_gradient()

    @VM.microcode(aout=['mesh'], ain=['mesh'])
    def Resample(self, mesh, Neff):
        def _Resample_filter(k, v):
            k0s = 2 * numpy.pi / v.BoxSize
            mask = numpy.bitwise_and.reduce([abs(ki) <= Neff//2 * k0 for ki, k0 in zip(k, k0s)])
            return v * mask

        mesh.r2c(out=Ellipsis).apply(_Resample_filter, out=Ellipsis).c2r(out=Ellipsis)

    @Resample.grad
    def _(self, _mesh, Neff):
        if _mesh is Zero: return

        def _Resample_filter(k, v):
            k0s = 2 * numpy.pi / v.BoxSize
            mask = numpy.bitwise_and.reduce([abs(ki) <= Neff//2 * k0 for ki, k0 in zip(k, k0s)])
            return v * mask

        _mesh.c2r_gradient().apply(_Resample_filter, out=Ellipsis).r2c_gradient(out=Ellipsis)

    @VM.microcode(aout=['mesh'], ain=['s'])
    def Paint(self, s, mesh):
        pm = self.pm
        x = s + self.q
        mesh[...] = pm.create(mode='real')
        layout = pm.decompose(x)
        N = pm.comm.allreduce(len(x))
        mesh[...].paint(x, layout=layout, hold=False)
        # to have 1 + \delta on the mesh
        mesh[...][...] *= 1.0 * pm.Nmesh.prod() / N

    @Paint.grad
    def _(self, _s, _mesh, s):
        pm = self.pm
        if _mesh is Zero:
            _s = Zero
        else:
            x = s + self.q
            N = pm.comm.allreduce(len(x))
            layout = pm.decompose(x)
            _s[...], junk = _mesh.paint_gradient(x, layout=layout, out_mass=False)
            _s[...][...] *= 1.0 * pm.Nmesh.prod() / N


