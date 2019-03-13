import numpy

from pmesh.pm import ParticleMesh
from .background import MatterDominated
from nbodykit.cosmology import Cosmology

class StateVector(object):
    def __init__(self, solver, Q):
        self.solver = solver
        self.pm = solver.pm
        self.Q = Q
        self.csize = solver.pm.comm.allreduce(len(self.Q))
        self.dtype = self.Q.dtype
        self.cosmology = solver.cosmology

        self.H0 = 100. # in km/s / Mpc/h units
        # G * (mass of a particle)
        self.GM0 = self.H0 ** 2 / ( 4 * numpy.pi ) * 1.5 * self.cosmology.Om0 * self.pm.BoxSize.prod() / self.csize

        self.S = numpy.zeros_like(self.Q)
        self.P = numpy.zeros_like(self.Q)
        self.F = numpy.zeros_like(self.Q)
        self.RHO = numpy.zeros_like(self.Q[..., 0])
        self.a = dict(S=None, P=None, F=None)

    def copy(self):
        obj = object.__new__(type(self))
        od = obj.__dict__
        od.update(self.__dict__)
        obj.S = self.S.copy()
        obj.P = self.P.copy()
        obj.F = self.F.copy()
        obj.RHO = self.RHO.copy()
        return obj

    @property
    def synchronized(self):
        a = self.a['S']
        return a == self.a['P'] and a == self.a['F']

    @property
    def X(self):
        return self.S + self.Q

    @property
    def V(self):
        a = self.a['P']
        return self.P * (self.H0 / a)

    def to_mesh(self):
        real = self.pm.create(mode='real')
        x = self.X
        layout = self.pm.decompose(x)
        real.paint(x, layout=layout, hold=False)
        return real

    def to_catalog(self, **kwargs):
        from nbodykit.lab import ArrayCatalog
        source = ArrayCatalog({'Position' : self.X, 'Velocity' : self.V},
            BoxSize=self.pm.BoxSize, Om0=self.cosmology.Om0,
            Time=self.a['S'], comm=self.pm.comm, **kwargs
        )
        return source

    def save(self, filename, attrs={}):
        from bigfile import FileMPI
        a = self.a['S']

        with FileMPI(self.pm.comm, filename, create=True) as ff:
            with ff.create('Header') as bb:
                keylist = ['Om0', 'Tcmb0', 'Neff', 'Ob0', 'Ode0']
                if getattr(self.cosmology, 'm_nu', None) is not None:
                    keylist.insert(3,'m_nu')
                for key in keylist:
                    bb.attrs[key] = getattr(self.cosmology, key)
                bb.attrs['Time'] = a
                bb.attrs['h'] = self.cosmology.H0 / self.H0 # relative h
                bb.attrs['RSDFactor'] = 1.0 / (self.H0 * a * self.cosmology.efunc(1.0 / a - 1))
                for key in attrs:
                    try:
                        #best effort
                        bb.attrs[key] = attrs[key]
                    except:
                        pass
            ff.create_from_array('1/Position', self.X)
            # Peculiar velocity in km/s
            ff.create_from_array('1/Velocity', self.V)
            # dimensionless potential (check this)
            ff.create_from_array('1/Density', self.RHO)

class Solver(object):
    def __init__(self, pm, cosmology, B=1, a_linear=1.0):
        """
            a_linear : float
                scaling factor at the time of the linear field.
                The growth function will be calibrated such that at a_linear D1 == 0.

        """
        if not isinstance(cosmology, Cosmology):
            raise TypeError("only nbodykit.cosmology object is supported")

        fpm = ParticleMesh(Nmesh=pm.Nmesh * B, BoxSize=pm.BoxSize, dtype=pm.dtype, comm=pm.comm, resampler=pm.resampler)
        self.pm = pm
        self.fpm = fpm
        self.cosmology = cosmology
        self.a_linear = a_linear

    # override nbodystep in subclasses
    @property
    def nbodystep(self):
        return FastPMStep(self)

    def whitenoise(self, seed, unitary=False):
        return self.pm.generate_whitenoise(seed, type='complex', unitary=unitary)

    def linear(self, whitenoise, Pk):
        return whitenoise.apply(lambda k, v:
                        Pk(sum(ki ** 2 for ki in k)**0.5) ** 0.5 * v / v.BoxSize.prod() ** 0.5)

    def lpt(self, linear, Q, a, order=2):
        """ This computes the 'force' from LPT as well. """
        assert order in (1, 2)

        from .force.lpt import lpt1, lpt2source

        state = StateVector(self, Q)
        pt = MatterDominated(self.cosmology.Om0, a=[a], a_normalize=self.a_linear)
        DX1 = pt.D1(a) * lpt1(linear, Q)

        V1 = a ** 2 * pt.f1(a) * pt.E(a) * DX1
        if order == 2:
            DX2 = pt.D2(a) * lpt1(lpt2source(linear), Q)
            V2 = a ** 2 * pt.f2(a) * pt.E(a) * DX2
            state.S[...] = DX1 + DX2
            state.P[...] = V1 + V2
            state.F[...] = a ** 2 * pt.E(a) * (pt.gf(a) / pt.D1(a) * DX1 + pt.gf2(a) / pt.D2(a) * DX2)
        else:
            state.S[...] = DX1
            state.P[...] = V1
            state.F[...] = a ** 2 * pt.E(a) * (pt.gf(a) / pt.D1(a) * DX1)

        state.a['S'] = a
        state.a['P'] = a

        return state

    def nbody(self, state, stepping, monitor=None):
        step = self.nbodystep
        for action, ai, ac, af in stepping:
            step.run(action, ai, ac, af, state, monitor)

        return state


class FastPMStep(object):
    def __init__(self, solver):
        self.cosmology = solver.cosmology
        self.pm = solver.fpm
        self.solver = solver

    def run(self, action, ai, ac, af, state, monitor):
        actions = dict(K=self.Kick, D=self.Drift, F=self.Force)

        event = actions[action](state, ai, ac, af)
        if monitor is not None:
            monitor(action, ai, ac, af, state, event)

    def Kick(self, state, ai, ac, af):
        assert ac == state.a['F']
        pt = MatterDominated(self.cosmology.Om0, a=[ai, ac, af], a_normalize=self.solver.a_linear)
        fac = 1 / (ac ** 2 * pt.E(ac)) * (pt.Gf(af) - pt.Gf(ai)) / pt.gf(ac)
        state.P[...] = state.P[...] + fac * state.F[...]
        state.a['P'] = af

    def Drift(self, state, ai, ac, af):
        assert ac == state.a['P']
        pt = MatterDominated(self.cosmology.Om0, a=[ai, ac, af], a_normalize=self.solver.a_linear)
        fac = 1 / (ac ** 3 * pt.E(ac)) * (pt.Gp(af) - pt.Gp(ai)) / pt.gp(ac)
        state.S[...] = state.S[...] + fac * state.P[...]
        state.a['S'] = af

    def prepare_force(self, state, smoothing):
        nbar = 1.0 * state.csize / self.pm.Nmesh.prod()

        X = state.X

        layout = self.pm.decompose(X, smoothing)

        X1 = layout.exchange(X)

        rho = self.pm.paint(X1)
        rho /= nbar # 1 + delta
        return layout, X1, rho

    def Force(self, state, ai, ac, af):
        from .force.gravity import longrange

        assert ac == state.a['S']

        # use the default PM support
        layout, X1, rho = self.prepare_force(state, smoothing=None)

        state.RHO[...] = layout.gather(rho.readout(X1))

        delta_k = rho.r2c(out=Ellipsis)

        state.F[...] = layout.gather(longrange(X1, delta_k, split=0, factor=1.5 * self.cosmology.Om0))

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

def leapfrog_with_lpt(stages):
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
