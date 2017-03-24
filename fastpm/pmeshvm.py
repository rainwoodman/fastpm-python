import numpy
from abopt.vmad import Zero, VM, microcode

class ParticleMeshVM(VM):
    def __init__(self, pm, q):
        self.pm = pm
        self.q = q

    @microcode(aout=['mesh'], ain=['mesh'])
    def Transfer(self, mesh, transfer):
        if isinstance(mesh, RealField):
            mesh.r2c(out=Ellipsis)\
                   .apply(lambda k, v: transfer(sum(ki ** 2 for ki in k) ** 0.5) * v, out=Ellipsis)\
                   .c2r(out=Ellipsis)
        else:
            mesh.apply(lambda k, v: transfer(sum(ki ** 2 for ki in k) ** 0.5) * v, out=Ellipsis)

    @Transfer.defvjp
    def _(self, _mesh, transfer):
        if _mesh is Zero: return
        if isinstance(_mesh, RealField):
            _mesh.c2r_gradient(out=Ellipsis)\
                   .apply(lambda k, v: transfer(sum(ki ** 2 for ki in k) ** 0.5) * v, out=Ellipsis)\
                   .r2c_gradient(out=Ellipsis)
        else:
            _mesh.apply(lambda k, v: transfer(sum(ki ** 2 for ki in k) ** 0.5) * v, out=Ellipsis)

    @microcode(aout=['residual'], ain=['mesh'])
    def Residual(self, mesh, data_x, sigma_x, residual):
        diff = mesh - data_x
        diff[...] /= sigma_x[...]
        residual[...] = diff

    @Residual.defvjp
    def _(self, _mesh, _residual, sigma_x):
        if _residual is Zero:
            _mesh = Zero
        else:
            _mesh[...] = _residual.copy()
            _mesh[...][...] /= sigma_x

    @microcode(aout=['R'], ain=['C'])
    def C2R(self, R, C):
        R[...] = C.c2r()

    @C2R.defvjp
    def _(self, _R, _C):
        if _R is Zero:
            _C[...] = Zero
        else:
            _C[...] = _R.c2r_gradient()

    @microcode(aout=['C'], ain=['R'])
    def R2C(self, C, R):
        C[...] = R.r2c()

    @microcode(aout=['C'], ain=['C'])
    def Decompress(self, C):
        return

    @Decompress.defvjp
    def _(self, _C):
        _C.decompress_gradient(out=Ellipsis)

    @R2C.defvjp
    def _(self, _C, _R):
        if _C is Zero:
            _R[...] = Zero
        else:
            _R[...] = _C.r2c_gradient()

    @microcode(aout=['mesh'], ain=['mesh'])
    def Resample(self, mesh, Neff):
        def _Resample_filter(k, v):
            k0s = 2 * numpy.pi / v.BoxSize
            mask = numpy.bitwise_and.reduce([abs(ki) <= Neff//2 * k0 for ki, k0 in zip(k, k0s)])
            return v * mask

        mesh.r2c(out=Ellipsis).apply(_Resample_filter, out=Ellipsis).c2r(out=Ellipsis)

    @Resample.defvjp
    def _(self, _mesh, Neff):
        if _mesh is Zero: return

        def _Resample_filter(k, v):
            k0s = 2 * numpy.pi / v.BoxSize
            mask = numpy.bitwise_and.reduce([abs(ki) <= Neff//2 * k0 for ki, k0 in zip(k, k0s)])
            return v * mask

        _mesh.c2r_gradient().apply(_Resample_filter, out=Ellipsis).r2c_gradient(out=Ellipsis)

    @microcode(aout=['layout'], ain=['s'])
    def Decompose(self, layout, s):
        x = s + self.q
        pm = self.pm
        layout[...] = pm.decompose(x)

    @Decompose.defvjp
    def _(self, _layout, _s):
        _s[...] = Zero

    @microcode(aout=['mesh'], ain=['s'])
    def QuickPaint(self, s, mesh):
        code = self.code()
        code.Decompose(s='s', layout='layout')
        code.Paint(s='s', layout='layout', mesh='mesh')
        mesh[...] = code.compute('mesh', init={'s' : s})

    @microcode(aout=['mesh'], ain=['s', 'layout'])
    def Paint(self, s, mesh, layout):
        pm = self.pm
        x = s + self.q
        mesh[...] = pm.create(mode='real')
        N = pm.comm.allreduce(len(x))
        mesh[...].paint(x, layout=layout, hold=False)
        # to have 1 + \delta on the mesh
        mesh[...][...] *= 1.0 * pm.Nmesh.prod() / N

    @Paint.defvjp
    def _(self, _s, _mesh, s, layout, _layout):
        pm = self.pm
        _layout[...] = Zero
        if _mesh is Zero:
            _s = Zero
        else:
            x = s + self.q
            N = pm.comm.allreduce(len(x))
            _s[...], junk = _mesh.paint_gradient(x, layout=layout, out_mass=False)
            _s[...][...] *= 1.0 * pm.Nmesh.prod() / N
