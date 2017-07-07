from . import core
import numpy

from kdcount import KDTree
from .background import PerturbationGrowth

class Solver(core.Solver):
    def __init__(self, pm, cosmology, B=1, r_split=None, NTimeBin=4):
        core.Solver.__init__(self, pm, cosmology, B)

        if r_split is None:
            r_split = self.fpm.BoxSize[0] / self.fpm.Nmesh[0]

        self.r_split = r_split
        self.r_cut = r_split * 4.5
        self.r_smth = r_split / 32
        self.NTimeBin = NTimeBin

    @property
    def nbodystep(self):
        return PPPMStep(self)

    def compute_shortrange(self, tree1, tree2, factor, out=None):
        from .force.gravity import shortrange
        return shortrange(tree1, tree2,
            self.r_split, self.r_cut, self.r_smth,
            factor=factor, out=out)

    def compute_longrange(self, X1, delta_k, factor):
        from .force.gravity import longrange
        return longrange(X1, delta_k, split=self.r_split, factor=factor)

    def compute_stepsize(self, tree, P, ac, E, Eprime, factor, out=None):
        from .force.gravity import compute_stepsize
        return compute_stepsize(
                tree,
                P, ac, E, Eprime,
                self.r_cut, self.r_smth,
                factor=factor, out=out)

class Timeline(object):
    def __init__(self, solver, NTimeBin, ai, af, size):
        self.ai = ai
        self.af = af
        self.solver = solver

        self.NTimeBin = NTimeBin

        self.Trees = [None] * self.NTimeBin
        self.TimeStamp = {'S' : [0] * self.NTimeBin,
                          'P' : [0] * self.NTimeBin,
                        }

        self.NumPart = numpy.zeros(self.NTimeBin, dtype='i8')
        self.BaseTime = 1 << self.NTimeBin

        self.pt = PerturbationGrowth(solver.cosmology,
                a=numpy.linspace(ai, af, self.BaseTime, endpoint=True))

        self.CurTime = 0

        self.ind = numpy.arange(size, dtype='i8')
        self.NumPart[0] = size
        self.offset = numpy.concatenate([[0], self.NumPart.cumsum()])


    def get_a_from_stamp(self, stamp):
        return (self.af * stamp + self.ai * (self.BaseTime - stamp)) / self.BaseTime

    def get_a(self, bin, variable, dt=0):
        stamp = self.TimeStamp[variable][bin]
        return self.get_a_from_stamp(stamp + dt)

    def stepbin(self):
        """ the last bin with active particles. This is the global step size """
        for bin, n in reversed(list(enumerate(self.NumPart))):
            if n != 0: return bin + 1
        raise RuntimeError("no particles on the time line")
        # consider return 0 instead

    def isedge(self, bin):
        t = self.NTimeBin - bin
        return self.CurTime % (1 << t) == 0

    def iscenter(self, bin):
        t = self.NTimeBin - bin
        return self.isedge(bin + 1) and not self.isedge(bin)

    def select(self, bin):
        return self.ind[self.offset[bin]:self.offset[bin] + self.NumPart[bin]]

    def invalidate_tree(self, bin):
        self.Trees[bin] = None

    def sync_empty(self, minbin):
        dt = self.BaseTime >> minbin
        for bin in range(minbin, self.NTimeBin):
            self.TimeStamp['S'][bin] += dt
            self.TimeStamp['P'][bin] += dt

    def get_tree(self, state, bin):
        if self.Trees[bin] is None:
            #print("Tree bin ==", bin)
            self.Trees[bin] = KDTree(state.X, ind=self.select(bin), boxsize=state.pm.BoxSize)
        return self.Trees[bin]

    def kdk(self, state, slow, fast):
        dt = self.BaseTime >> slow
        hdt = self.BaseTime >> fast
        self.kick(state, fast, slow, hdt)
        self.kick(state, slow, fast, hdt)
        self.kick(state, slow, slow, hdt)
        self.drift(state, slow, dt)
        self.kick(state, fast, slow, hdt)
        self.kick(state, slow, fast, hdt)
        self.kick(state, slow, slow, hdt)

    def kick(self, state, bin1, bin2, dt):
        if bin1 == self.NTimeBin: return
        if bin2 == self.NTimeBin: return

        pt = self.pt

        ai = self.get_a(bin1, 'P')
        af = self.get_a(bin1, 'P', dt=dt)
        ac = (ai * af) ** 0.5

        if bin1 == bin2:
            self.TimeStamp['P'][bin1] += dt

        # timeline updated, do we really need to compute anything?
        if self.NumPart[bin1] == 0: return
        if self.NumPart[bin2] == 0: return

        if bin1 == bin2:
            #print('Mom  bin >=', bin1, self.TimeStamp)
            pass
        fac = 1 / (ac ** 2 * pt.E(ac)) * (af - ai)

        tree1 = self.get_tree(state, bin1)
        tree2 = self.get_tree(state, bin2)

        state.F[tree1.ind] = 0

        self.solver.compute_shortrange(tree1, tree2,
                factor=state.GM0 / state.H0 ** 2, out=state.F)

        state.P[tree1.ind] += fac * state.F[tree1.ind]

    def drift(self, state, bin, dt):
        if bin == self.NTimeBin: return

        pt = self.pt

        ac = self.get_a(bin, 'P')
        ai = self.get_a(bin, 'S')
        af = self.get_a(bin, 'S', dt=dt)

        self.TimeStamp['S'][bin] += dt

        # timeline updated, do we really need to compute anything?
        if self.NumPart[bin] == 0: return

        #print('Pos  bin ==', bin, self.TimeStamp)

        fac = 1 / (ac ** 3 * pt.E(ac)) * (pt.Gp(af) - pt.Gp(ai)) / pt.gp(ac)

        s = self.select(bin)
        state.X[s] += state.P[s] * fac

        self.invalidate_tree(bin)

    def run(self, state, monitor=None):
        while self.CurTime < self.BaseTime:
            level = self.stepbin()

            for bin in range(level - 1, -1, -1):
                if self.iscenter(bin):
                    self.kdk(state, bin, bin + 1)

            for bin in range(level):
                if self.isedge(bin):
                    # when CurTime is 0 this triggers the first timebin rebuild
                    self.build_timebins(state, bin)
                    break

            level = self.stepbin()
            self.sync_empty(level)

            self.CurTime = self.CurTime + (self.BaseTime >> level)
            #print(self.TimeStamp)

    def build_timebins(self, state, binmin):
        pt = self.pt
        solver = self.solver

        for bin in range(binmin, self.NTimeBin):
            tree = self.get_tree(state, bin)
            a = self.get_a(bin, 'S')
            a1 = self.get_a(bin, 'S', dt=self.BaseTime >> binmin)
            state.stepsize[tree.ind] = (a1 - a) * 0.9 # make sure it falls on binmin
            solver.compute_stepsize(tree,
                state.P, a, pt.E(a), pt.E(a, order=1),
                factor=state.GM0 / state.H0 ** 2, out=state.stepsize)
            if len(tree.ind) > 0:
                #print("update", bin, tree)
                pass

        bins = numpy.log2((self.af - self.ai) / state.stepsize).astype('int')
        bins.clip(0, self.NTimeBin -1, out=bins)

        self.ind = numpy.argsort(bins)
        self.NumPart = numpy.bincount(bins, minlength=self.NTimeBin)
        self.offset = numpy.concatenate([[0], self.NumPart.cumsum()])

        #print('Time', a, 'bin ==', binmin, 'stepsize updated', state.stepsize.max(), state.stepsize.min(), self.NumPart)

        for i in range(binmin, len(self.Trees)):
            self.invalidate_tree(i)

class HOLDState:
    def __init__(self, state, support):
        X = state.X
        layout = state.pm.decompose(X, smoothing=support)
        self.X = layout.exchange(X)
        self.P = layout.exchange(state.P)
        self.F = numpy.empty_like(self.P)
        self.stepsize = numpy.empty(len(self.P))
        self.GM0 = state.GM0
        self.H0 = state.H0
        self.layout = layout
        self.pm = state.pm

    def gather(self, state):
        X = self.layout.gather(self.X, mode='local')
        P = self.layout.gather(self.P, mode='local')
        state.S[...] = X - state.Q
        state.P[...] = P

class PPPMStep(core.FastPMStep):
    def __init__(self, solver):
        core.FastPMStep.__init__(self, solver)

    def Drift(self, state, ai, ac, af):
        # instead of Drift, do it with a HOLD evolve step
        support = max([self.solver.r_cut, self.pm.resampler.support * 0.5])
        hstate = HOLDState(state, support)

        timeline = Timeline(self.solver, self.solver.NTimeBin, ai, af, len(hstate.stepsize))

        timeline.run(hstate)

        hstate.gather(state)
        
        state.a['S'] = af
        return dict(hstate=hstate)

