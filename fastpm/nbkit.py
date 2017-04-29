from fastpm.engine import FastPMEngine, ParticleMesh, CodeSegment
from fastpm.perturbation import PerturbationTheory

from nbodykit.base.catalog import CatalogSource, column
import numpy

class FastPMCatalogSource(CatalogSource):
    def __repr__(self):
        return "FastPMSimulation()" %self.attrs

    def __init__(self, linear, astart=0.1, aend=1.0, boost=2, Nsteps=5, cosmo=None):
        self.comm = linear.comm

        if cosmo is None:
            cosmo = linear.Plin.cosmo

        self.cosmo = cosmo

        # the linear density field mesh
        self.linear = linear

        self.attrs.update(linear.attrs)

        asteps = numpy.linspace(astart, aend, Nsteps)
        self.attrs['astart'] = astart
        self.attrs['aend'] = aend
        self.attrs['Nsteps'] = Nsteps
        self.attrs['asteps'] = asteps
        self.attrs['boost'] = boost

        pt = PerturbationTheory(cosmo)

        engine = FastPMEngine(self.linear.pm, B=boost, shift=0.5)
        self.model = CodeSegment(engine)
        self.model.solve_fastpm(pt=pt, asteps=asteps, dlinear_k='dlinear_k', s='s', v='v')
        self.linear = linear

        H0 = 100.
        self.RSD = 1.0 / (H0 * aend * self.cosmo.efunc(1.0 / aend - 1))

        CatalogSource.__init__(self, comm=linear.comm, use_cache=False)

        self.update_csize()

        # this is slow
        self._run()

    @property
    def size(self):
        return len(self.model.engine.q)

    def _run(self):
        s, p = self.model.compute(['s', 'v'], init={'dlinear_k': self.linear.to_field(mode='complex')})
        q = self.model.engine.q
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
