from fastpm.core import Solver, leapfrog

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

        solver = Solver(self.linear.pm, cosmology=self.cosmo, B=boost)
        Q = self.linear.pm.generate_uniform_particle_grid(shift=0.5)

        self.linear = linear

        dlin = self.linear.to_field(mode='complex')
        state = solver.lpt(dlin, Q, a=astart, order=2)
        state = solver.nbody(state, leapfrog(numpy.linspace(astart, aend, Nsteps + 1, endpoint=True)))

        H0 = 100.
        self.RSD = 1.0 / (H0 * aend * self.cosmo.efunc(1.0 / aend - 1))

        self._size = len(Q)
        CatalogSource.__init__(self, comm=linear.comm)

        self._csize = self.comm.allreduce(self._size)

        self['Displacement'] = state.S
        self['InitialPosition'] = state.Q
        self['ConjugateMomentum'] = state.P # a ** 2  / H0 dx / dt

    @property
    def size(self):
        return self._size

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
