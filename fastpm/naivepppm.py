from . import core
from kdcount import KDTree
from numpy import pi as PI

class Solver(core.Solver):
    def __init__(self, pm, cosmology, B=1, r_split=None):
        core.Solver.__init__(self, pm, cosmology, B)

        if r_split is None:
            r_split = self.fpm.BoxSize[0] / self.fpm.Nmesh[0]
        self.r_split = r_split

    @property
    def nbodystep(self):
        return PPPMStep(self)

class PPPMStep(core.FastPMStep):
    def __init__(self, solver):
        core.FastPMStep.__init__(self, solver)

        self.r_split = solver.r_split
        self.r_cut = solver.r_split * 4.5
        self.r_smth = solver.r_split

    def Force(self, state, ai, ac, af):
        from .force.gravity import longrange
        from .force.gravity import shortrange
        nbar = 1.0 * state.csize / self.pm.Nmesh.prod()

        support = max([self.r_cut, self.pm.resampler.support * 0.5])

        layout, X1, rho, nbar = self.prepare_force(state, smoothing=support)

        rho /= nbar # 1 + delta

        state.RHO[...] = layout.gather(rho.readout(X1))

        delta_k = rho.r2c(out=Ellipsis)

        state.F[...] = layout.gather(
                longrange(X1, delta_k, split=0, factor=1.5 * self.cosmology.Om0)
                )

        tree = KDTree(X1, boxsize=self.pm.BoxSize)
        GM0 = 1.0 / ( 4 * PI ) * 1.5 * self.cosmology.Om0 / nbar

        Fs = layout.gather(
            shortrange(tree, tree, self.r_split, self.r_cut, self.r_smth, factor=GM0),
            mode='local'
            )
        state.F[...] += Fs
        state.a['F'] = af

