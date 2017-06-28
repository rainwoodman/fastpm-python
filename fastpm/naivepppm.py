from . import core
from kdcount import KDTree

class Solver(core.Solver):
    def __init__(self, pm, cosmology, B=1, r_split=None):
        core.Solver.__init__(self, pm, cosmology, B)

        if r_split is None:
            r_split = self.fpm.BoxSize[0] / self.fpm.Nmesh[0]
        self.r_split = r_split
        self.r_cut = r_split * 4.5
        self.r_smth = r_split

    @property
    def nbodystep(self):
        return PPPMStep(self)

    def compute_shortrange(self, tree1, tree2, factor):
        from .force.gravity import shortrange
        return shortrange(tree1, tree2,
            self.r_split, self.r_cut, self.r_smth,
            factor=factor)

    def compute_longrange(self, X1, delta_k, factor):
        from .force.gravity import longrange
        return longrange(X1, delta_k, split=self.r_split, factor=factor)

class PPPMStep(core.FastPMStep):
    def __init__(self, solver):
        core.FastPMStep.__init__(self, solver)

    def Force(self, state, ai, ac, af):
        nbar = 1.0 * state.csize / self.pm.Nmesh.prod()

        support = max([self.solver.r_cut, self.pm.resampler.support * 0.5])

        layout, X1, rho = self.prepare_force(state, smoothing=support)

        state.RHO[...] = layout.gather(rho.readout(X1))

        delta_k = rho.r2c(out=Ellipsis)

        state.F[...] = layout.gather(
                self.solver.compute_longrange(X1, delta_k, factor=1.5 * self.cosmology.Om0)
                )

        tree = KDTree(X1, boxsize=self.pm.BoxSize)

        Fs = layout.gather(
            self.solver.compute_shortrange(tree, tree, factor=state.GM0 / state.H0**2),
            mode='local'
            )
        state.F[...] += Fs
        state.a['F'] = af

