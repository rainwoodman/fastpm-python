import numpy

from pmesh.pm import ParticleMesh
from .background import PerturbationGrowth
from .operators import lpt1, lpt2source, gravity
from nbodykit.cosmology import Cosmology

class StateVector(object):
    def __init__(self, solver, Q, S=None, P=None, F=None):
        self.solver = solver
        self.pm = solver.pm
        self.Q = Q
        self.csize = solver.pm.comm.allreduce(len(self.Q))
        self.dtype = self.Q.dtype
        self.cosmology = solver.cosmology

        if S is None: S = numpy.zeros_like(self.Q)
        if P is None: P = numpy.zeros_like(self.Q)
        if F is None: F = numpy.zeros_like(self.Q)

        self.S = S
        self.P = P
        self.F = F
        self.a = dict(S=None, P=None, F=None)

    def copy(self):
        return StateVector(self.solver, self.Q, self.S.copy(), self.P.copy(), self.F.copy())

    @property
    def synchronized(self):
        a = self.a['S']
        return a == self.a['P'] and a == self.a['F']

    @property
    def X(self):
        return self.S + self.Q

    def to_mesh(self):
        real = self.pm.create(mode='real')
        x = self.X
        layout = self.pm.decompose(x)
        real.paint(x, layout=layout, hold=False)
        return real

    def save(self, filename, attrs={}):
        from bigfile import FileMPI
        H0 = 100. # in Mpc/h units
        a = self.a['S']

        with FileMPI(self.pm.comm, filename, create=True) as ff:
            with ff.create('Header') as bb:
                keylist = ['Om0', 'Tcmb0', 'Neff', 'Ob0', 'Ode0']
                if hasattr(self.cosmology, 'm_nu'):
                    keylist.append('m_nu')
                for key in keylist:
                    bb.attrs[key] = getattr(self.cosmology, key)
                bb.attrs['Time'] = a
                bb.attrs['h'] = self.cosmology.H0 / H0 # relative h
                bb.attrs['RSDFactor'] = 1.0 / (H0 * a * self.cosmology.efunc(1.0 / a - 1))
                for key in attrs:
                    try:
                        #best effort
                        bb.attrs[key] = attrs[key]
                    except:
                        pass
            ff.create_from_array('1/Position', self.X)
            # Peculiar velocity in km/s
            ff.create_from_array('1/Velocity', self.P * (H0 / a))

class Solver(object):
    def __init__(self, pm, cosmology, B=1):
        self.pm = pm
        if not isinstance(cosmology, Cosmology):
            raise TypeError("only nbodykit.cosmology object is supported")

        self.cosmology = cosmology

        fpm = ParticleMesh(Nmesh=pm.Nmesh * B, BoxSize=pm.BoxSize, dtype=pm.dtype, comm=pm.comm, resampler=pm.resampler)
        self.boosted_pm = fpm

    def whitenoise(self, seed, unitary=False):
        return self.pm.generate_whitenoise(seed, mode='complex', unitary=unitary)

    def linear(self, whitenoise, Pk):
        return whitenoise.apply(lambda k, v:
                        Pk(sum(ki ** 2 for ki in k)**0.5) ** 0.5 * v / v.BoxSize.prod() ** 0.5)

    def lpt(self, linear, Q, a, order=2):
        assert order in (1, 2)

        state = StateVector(self, Q)
        pt = PerturbationGrowth(self.cosmology, a=[a])
        DX1 = pt.D1(a) * lpt1(linear, Q)

        V1 = a ** 2 * pt.f1(a) * pt.E(a) * DX1
        if order == 2:
            DX2 = pt.D2(a) * lpt1(lpt2source(linear), Q)
            V2 = a ** 2 * pt.f2(a) * pt.E(a) * DX2
            state.S[...] = DX1 + DX2
            state.P[...] = V1 + V2
        else:
            state.S[...] = DX1
            state.P[...] = V1

        state.a['S'] = a
        state.a['P'] = a

        return state

    def nbody(self, state, stepping, monitor=None):
        step = FastPMStep(self.cosmology, self.boosted_pm, monitor)

        for action, ai, ac, af in stepping:

            step.run(action, ai, ac, af, state)

        return state

class FastPMStep(object):
    def __init__(self, cosmology, pm, monitor):
        self.cosmology = cosmology
        self.pm = pm
        self.monitor = monitor

    def run(self, action, ai, ac, af, state):
        actions = dict(K=self.Kick, D=self.Drift, F=self.Force)

        event = actions[action](state, ai, ac, af)
        if self.monitor is not None:
            self.monitor(action, ai, ac, af, state, event)

    def Kick(self, state, ai, ac, af):
        pt = PerturbationGrowth(self.cosmology, a=[ai, ac, af])
        fac = 1 / (ac ** 2 * pt.E(ac)) * (pt.Gf(af) - pt.Gf(ai)) / pt.gf(ac)
        state.P[...] = state.P[...] + fac * state.F[...]
        state.a['P'] = af

    def Drift(self, state, ai, ac, af):
        pt = PerturbationGrowth(self.cosmology, a=[ai, ac, af])
        fac = 1 / (ac ** 3 * pt.E(ac)) * (pt.Gp(af) - pt.Gp(ai)) / pt.gp(ac)
        state.S[...] = state.S[...] + fac * state.P[...]
        state.a['S'] = af

    def Force(self, state, ai, ac, af):
        nbar = 1.0 * state.csize / self.pm.Nmesh.prod()
        state.F[...], delta_k = gravity(state.X, self.pm, factor=1.5 * self.cosmology.Om0 / nbar, return_deltak = True)

        delta_k[...] /= nbar
        state.a['F'] = af
        return dict(delta_k=delta_k)

def autostages(knots, N, astart=None, N0=None):
    """ Generate an optimized list of N stages that includes time steps at the knots.

        Parameters
        ----------
        astart : float, or None
            starting time, default is knots[0]
        N : int
            total number of stages
        N0 : int or None
            at least add this many stages before the earlist knot, default None;
            useful only if astart != min(knots), and len(knots) > 1
        knots : list
            stages that must exist


        >>> autostages(0.1, N=11, knots=[0.1, 0.2, 0.5, 1.0])

    """

    knots = numpy.array(knots)
    knots.sort()

    stages = numpy.array([], dtype='f8')
    if astart is not None and astart != knots.min():
        assert astart < knots.min()
        if N0 is None: N0 = 1
        knots = numpy.append([astart], knots)
    else:
        N0 = 1

    for i in range(0, len(knots) - 1):
        da = (knots[-1] - knots[i]) / (N - len(stages) - 1)

        N_this_span = int((knots[i + 1] - knots[i]) / da + 0.5)
        if i == 0 and N_this_span < N0:
            N_this_span = N0

        add = numpy.linspace(knots[i], knots[i + 1], N_this_span, endpoint=False)

        #print('i = =====', i)
        #print('knots[i]', knots[i], da, N_this_span, stages, add)

        stages = numpy.append(stages, add)

    stages = numpy.append(stages, [knots[-1]])

    return stages

def leapfrog(stages):
    """ Generate a leap frog stepping scheme.

        Parameters
        ----------
        stages : array_like
            Time (a) where force computing stage is requested.
    """
    if len(stages) == 0:
        return

    ai = stages[0]
    # first force calculation for jump starting
    yield 'F', ai, ai, ai
    x, p, f = ai, ai, ai

    for i in range(len(stages) - 1):
        a0 = stages[i]
        a1 = stages[i + 1]
        ah = (a0 * a1) ** 0.5
        yield 'K', p, f, ah
        p = ah
        yield 'D', x, p, a1
        x = a1
        yield 'F', f, x, a1
        f = a1
        yield 'K', p, f, a1
        p = a1
