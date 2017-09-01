from argparse import ArgumentParser
from pmesh.pm import ParticleMesh
ap = ArgumentParser()
ap.add_argument("config")

from .core import Solver
from .core import leapfrog
from .core import autostages
from .background import PerturbationGrowth

from nbodykit.cosmology import Planck15
from nbodykit.cosmology import EHPower
from nbodykit.cosmology import Cosmology
from nbodykit.lab import FFTPower, FieldMesh
import numpy

class Config(dict):
    def __init__(self, path):
        self.prefix = '%s' % path
        filename = self.makepath('config.py')

        self['boxsize'] = 1380.0
        self['shift'] = 0.0
        self['nc'] = 64
        self['ndim'] = 3
        self['seed'] = 1985
        self['pm_nc_factor'] = 2
        self['resampler'] = 'tsc'
        self['cosmology'] = Planck15
        self['powerspectrum'] = EHPower(Planck15, 0)
        self['unitary'] = False
        self['stages'] = numpy.linspace(0.1, 1.0, 5, endpoint=True)
        self['aout'] = [1.0]

        local = {} # these names will be usable in the config file
        local['EHPower'] = EHPower
        local['Cosmology'] = Cosmology
        local['Planck15'] = Planck15
        local['linspace'] = numpy.linspace
        local['autostages'] = autostages

        import nbodykit.lab as nlab
        local['nlab'] = nlab

        names = set(self.__dict__.keys())

        exec(open(filename).read(), local, self)

        unknown = set(self.__dict__.keys()) - names
        assert len(unknown) == 0

        self.finalize()
        global _config
        _config = self

    def finalize(self):
        self['aout'] = numpy.array(self['aout'])

        self.pm = ParticleMesh(BoxSize=self['boxsize'], Nmesh= [self['nc']] * self['ndim'], resampler=self['resampler'])
        mask = numpy.array([ a not in self['stages'] for a in self['aout']], dtype='?')
        missing_stages = self['aout'][mask]
        if len(missing_stages):
            raise ValueError('Some stages are requested for output but missing: %s' % str(missing_stages))

    def makepath(self, filename):
        import os.path
        return os.path.join(self.prefix, filename)

def main(args=None):
    ns = ap.parse_args(args)
    config = Config(ns.config)

    solver = Solver(config.pm, cosmology=config['cosmology'], B=config['pm_nc_factor'])
    whitenoise = solver.whitenoise(seed=config['seed'], unitary=config['unitary'])
    dlin = solver.linear(whitenoise, Pk=lambda k : config['powerspectrum'](k))

    Q = config.pm.generate_uniform_particle_grid(shift=config['shift'])

    state = solver.lpt(dlin, Q=Q, a=config['stages'][0], order=2)

    def write_power(d, path, a):
        meshsource = FieldMesh(d)
        r = FFTPower(meshsource, mode='1d')
        if config.pm.comm.rank == 0:
            print('Writing matter power spectrum at %s' % path)
            # only root rank saves
            numpy.savetxt(path, 
                numpy.array([
                  r.power['k'], r.power['power'].real, r.power['modes'],
                  r.power['power'].real / solver.cosmology.scale_independent_growth_factor(1.0 / a - 1) ** 2,
                ]).T,
                comments='# k p N p/D**2')

    write_power(dlin, config.makepath('power-linear.txt'), a=1.0)

    def monitor(action, ai, ac, af, state, event):
        if config.pm.comm.rank == 0:
            print('Step %s %06.4f - (%06.4f) -> %06.4f' %( action, ai, ac, af),
                  'S %(S)06.4f P %(P)06.4f F %(F)06.4f' % (state.a))

        if action == 'F':
            a = state.a['F']
            path = config.makepath('power-%06.4f.txt' % a)
            write_power(event['delta_k'], path, a)

        if state.synchronized:
            a = state.a['S']
            if a in config['aout']:
                path = config.makepath('fpm-%06.4f' % a) % a
                if config.pm.comm.rank == 0:
                    print('Writing a snapshot at %s' % path)
                # collective save
                state.save(path, attrs=config)

    solver.nbody(state, stepping=leapfrog(config['stages']), monitor=monitor)

if __name__ == '__main__':
    main()
