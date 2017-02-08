from fastpm import Evolution, KickDriftKick, PerturbationGrowth
from nbodykit.base.particles import ParticleSource, column

from pmesh.pm import ParticleMesh, RealField

class FastPMParticleSource(ParticleSource):
    def __repr__(self):
        return "FastPMSimulation()" %self.attrs

    def __init__(self, linear, astart=0.1, aend=1.0, boost=2, Nsteps=5, cosmo=None):
        self.comm = linear.comm

        if cosmo is None:
            self.cosmo = linear.Plin.cosmo

        # the linear density field mesh
        self.linear = linear

        self.attrs.update(linear.attrs)

        self.attrs['astart'] = astart
        self.attrs['aend'] = aend
        self.attrs['Nsteps'] = Nsteps
        self.attrs['boost'] = boost

        self.evolution = KickDriftKick(self.linear.pm, B=boost, shift=0.5)

        self.pt = PerturbationGrowth(self.cosmo)

        self.model = self.evolution.simulation(self.pt, astart, aend, Nsteps)
        self.linear = linear

        H0 = 100.
        self.RSD = 1.0 / (H0 * aend * self.cosmo.efunc(1.0 / aend - 1))

        ParticleSource.__init__(self, comm=linear.comm, use_cache=False)

        self.update_csize()

        # this is slow
        self._run()

    @property
    def size(self):
        return len(self.evolution.q)

    def _run(self):
        s, p = self.model.compute(['s', 'p'], init={'dlin_k': self.linear.to_field(mode='complex')})
        q = self.evolution.q
        self['Displacement'] = s
        self['InitialPosition'] = q
        self['ConjugateMomentum'] = p # a**2 H0 v_pec

    @column
    def Position(self):
        return self['InitialPosition'] + self['Displacement']

    @column
    def Velocity(self):
        H0 = 100.0
        return self['ConjugateMomentum'] * H0 / self.attrs['aend']

    @column
    def VelocityOffset(self):
        return self['Velocity'] * self.RSD
