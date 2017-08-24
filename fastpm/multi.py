import numpy

from pmesh.pm import ParticleMesh
from .background import PerturbationGrowth
from nbodykit.cosmology import Cosmology

from .state import StateVector, Matter, Baryon, CDM, NCDM

class Step(object):
    pass

class Solver(object):
    def __init__(self, pm, cosmology, B=1):
        """
        """
        if not isinstance(cosmology, Cosmology):
            raise TypeError("only nbodykit.cosmology object is supported")

        fpm = ParticleMesh(Nmesh=pm.Nmesh * B, BoxSize=pm.BoxSize, dtype=pm.dtype, comm=pm.comm, resampler=pm.resampler)
        self.pm = pm
        self.fpm = fpm
        self.cosmology = cosmology

    # override nbodystep in subclasses
    @property
    def nbodystep(self):
        return FastPMStep(self)

    def whitenoise(self, seed, unitary=False):
        return self.pm.generate_whitenoise(seed, mode='complex', unitary=unitary)

    def primordial(self, wn, Pk):
        get_k = lambda k : sum(ki ** 2 for ki in k) ** 0.5
        def apply_primordial(k, v):
            k = get_k(k)
            t = (2 * numpy.pi ** 2 * k ** -3 * Pk(k)) ** 0.5 / wn.BoxSize.prod() ** 0.5
            t[k == 0] = 0
            return t * v
        return wn.apply(apply_primordial)

    def lpt(self, primordial, species_spec, Q, a, order=1):
        """ This computes the 'force' from LPT as well.

            transfer_function: dict
                a dictionary of species_name : (d(k), d d/ d a(k))

        """
        assert order in (1,) # only first order is supported for now.

        # FIXME: Add 2LPT. According to Vlah, the 2LPT of CDM is the only
        # important term, but it gains some interaction terms
        # with other species.

        z = 1. / a - 1
        from .force.lpt import lpt1

        get_k = lambda k : sum(ki ** 2 for ki in k) ** 0.5

        species = {}
        for spname, (sptype, d, dd) in species_spec.items():
            sp = sptype(self.cosmology, self.pm.BoxSize, Q, self.pm.comm)

            def apply_density(k, v):
                return d(get_k(k)) * v

            source = primordial.apply(apply_density)
            DX1 = lpt1(source, Q)
            sp.S[...] = DX1

            def apply_velocity(k, v):
                return a * self.cosmology.efunc(z) * dd(get_k(k)) * v

            source = primordial.apply(apply_velocity)
            sp.P[...] = a ** 2 * lpt1(source, Q)
            sp.F[...] = 0

            sp.a['S'] = a
            sp.a['P'] = a

            species[spname] = sp

        state = StateVector(self.cosmology, species, self.pm.comm)
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
        pt = PerturbationGrowth(self.cosmology, a=[ai, ac, af], a_normalize=self.solver.a_linear)
        fac = 1 / (ac ** 2 * pt.E(ac)) * (pt.Gf(af) - pt.Gf(ai)) / pt.gf(ac)
        state.P[...] = state.P[...] + fac * state.F[...]
        state.a['P'] = af

    def Drift(self, state, ai, ac, af):
        pt = PerturbationGrowth(self.cosmology, a=[ai, ac, af], a_normalize=self.solver.a_linear)
        fac = 1 / (ac ** 3 * pt.E(ac)) * (pt.Gp(af) - pt.Gp(ai)) / pt.gp(ac)
        state.S[...] = state.S[...] + fac * state.P[...]
        state.a['S'] = af

    def prepare_force(self, state, smoothing):
        nbar = 1.0 * state.csize / self.pm.Nmesh.prod()

        X = state.X

        layout = self.pm.decompose(X, smoothing)

        X1 = layout.exchange(X)

        rho = self.pm.create(mode="real")
        rho.paint(X1, hold=False)
        rho /= nbar # 1 + delta
        return layout, X1, rho

    def Force(self, state, ai, ac, af):
        from .force.gravity import longrange

        # use the default PM support
        layout, X1, rho = self.prepare_force(state, smoothing=None)

        state.RHO[...] = layout.gather(rho.readout(X1))

        delta_k = rho.r2c(out=Ellipsis)

        state.F[...] = layout.gather(longrange(X1, delta_k, split=0, factor=1.5 * self.cosmology.Om0))

        state.a['F'] = af
        return dict(delta_k=delta_k)

def get_species_transfer_function_from_class(cosmology, z):
    """ compuate the species transfer functions (d and dd/da)
        from the result of class.
    """
    tf = cosmology.get_transfer(z=z)
    d = {}

    # flip the sign to meet preserve the phase of the
    d['d_cdm'] = tf['d_cdm'] * -1
    d['d_b'] = tf['d_b'] * -1
    d['d_ncdm[0]'] = tf['d_ncdm[0]'] * -1
    if cosmology.gauge == 'newtonian':
        # dtau to da, the negative sign in the 3 fluid equation of motion
        # eliminated due to the flip in d
        fac = 1.0 / (cosmology.hubble_function(z) * (1. + z) ** -2)
        d['dd_cdm'] = tf['t_cdm'] * fac
        d['dd_b'] = tf['t_b'] * fac
        d['dd_ncdm[0]'] = tf['t_ncdm[0]'] * fac
    elif cosmology.gauge == 'synchronous':
        # FIXME: 
        raise NotImplementedError("This needs to be written, code below is junk")
        fac = 1.0 / (cosmology.hubble_function(z) * (1. + z) ** -2)
        d['dd_cdm'] = 0.5 * tf['h_prime'] * fac
        d['dd_b'] = 0.5 * tf['h_prime'] * tf['t_b'] * fac
        d['dd_ncdm[0]'] = 0.5 * tf['h_prime'] * tf['t_ncdm[0]'] * fac

    k = tf['k'].copy()
    e = {}
    for name in d:
        e[name] = lambda k, x=tf['k'], y=d[name]: numpy.interp(k, x, y, left=0, right=0)
    return e

