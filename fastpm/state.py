import numpy

H0 = 100. # in km/s / Mpc/h units
G = 43007.1 

class Species(object):
    def __init__(self, cosmology, BoxSize, Q, comm):
        self.cosmology = cosmology
        self.comm = comm
        self.BoxSize = numpy.zeros(Q.shape[-1], dtype='f8')
        self.BoxSize[...] = BoxSize

        self.Q = Q
        self.dtype = self.Q.dtype

        self.csize = self.comm.allreduce(len(self.Q))

        self.S = numpy.zeros_like(self.Q)
        self.P = numpy.zeros_like(self.Q)
        self.F = numpy.zeros_like(self.Q)
        self.RHO = numpy.zeros_like(self.Q[..., 0])
        self.a = dict(S=None, P=None, F=None)

    def Omega(self, a): 
        """ default is matter, override for other species; time dependent for neutrinos """
        raise NotImplementedError

    def M1(self, a):
        """ mass of a particle, time dependent for neutrinos """
        # rhoc = 3 * H(a) ** 2 / (8 pi G)
        z = 1.0 / a - 1
        rhoc = 3 * self.cosmology.efunc(z) ** 2 * H0 ** 2 / (8 * numpy.pi * G)
        return rhoc * self.Omega(a) * self.BoxSize.prod() / self.csize

    def copy(self):
        obj = object.__new__(type(self))
        od = obj.__dict__
        od.update(self.__dict__)

        # but important data is copied.
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
        return self.P * (H0 / a)

    def to_mesh(self, pm):
        real = pm.create(mode='real')
        x = self.X
        layout = pm.decompose(x)
        real.paint(x, layout=layout, hold=False)
        return real

    def to_catalog(self, **kwargs):
        from nbodykit.source import ArrayCatalog
        from nbodykit.transform import ConstantArray
        Omega = self.Omega(self.a['S'])
        source = ArrayCatalog({'Position' : self.X, 'Velocity' : self.V,
            'Weight' : ConstantArray(Omega, len(self.X))},
            BoxSize=self.BoxSize, Omega=Omega, Omega0=self.Omega(1.0),
            Time=self.a['S'], comm=self.comm, **kwargs
        )
        return source

    def save(self, filename, dataset):
        from bigfile import FileMPI
        with FileMPI(self.comm, filename, create=True) as ff:
            ff.create_from_array(dataset + '/Position', self.X)
            # Peculiar velocity in km/s
            ff.create_from_array(dataset + '/Velocity', self.V)
            # dimensionless potential (check this)
            ff.create_from_array(dataset + '/Density', self.RHO)

class Matter(Species):
    def Omega(self, a):
        return self.cosmology.Om(z=1. / a - 1)

class Baryon(Species):
    def Omega(self, a):
        return self.cosmology.Ob(z=1. / a - 1)

class CDM(Species):
    def Omega(self, a):
        return self.cosmology.Odm(z=1. / a - 1)

class NCDM(Species):
    def Omega(self, a):
        # FIXME: using Omega_ncdm after switching to new nbodykit cosmology.
        return self.cosmology.Onu(z=1. / a - 1)

class TiledNCDM(Species):
    @staticmethod
    def generate_nu_template(m_nu, nside, nshell):
        # Arka's Tiled scheme that samples neutrion thermal velocity coherently.
        # arxiv: 1801.03906

        import numpy
        from scipy.integrate import quad

        import healpy

        # the tempalte of directions.
        vec = numpy.array(healpy.pix2vec(nside, numpy.arange(healpy.nside2npix(nside)))).T
        # make sure the vecs are isotropic.
        vec /= (vec**2 * 3).sum(axis=0)**0.5

        # equal probablitiy shells in the distribution function
        shells = numpy.linspace(0, 1, nshell + 1, endpoint=True)

        def fermi_dirac(p):
            return p**2 / (numpy.exp(p) + 1)

        def fermi_dirac_p2(p):
            return p**4 / (numpy.exp(p) + 1)

        factor = (quad(fermi_dirac_p2, 0, 20)[0] / quad(fermi_dirac, 0, 20)[0] ) ** 0.5

        fermi_dirac_vel_nu = numpy.linspace(0, 20, 4097, endpoint=True)
        fermi_dirac_cumprob_nu = numpy.array([quad(fermi_dirac, 0, vel)[0]
                                  for vel in fermi_dirac_vel_nu]) / quad(fermi_dirac, 0, 20)[0]

        ind = fermi_dirac_cumprob_nu.searchsorted(shells)

        vedge = fermi_dirac_vel_nu[ind]

        vbar = numpy.array([(
            quad(fermi_dirac_p2, start, end)[0] / quad(fermi_dirac, start, end)[0])**0.5
               for start, end in zip(vedge[:-1], vedge[1:])])
            
        # to a **2 dx /dt in km/s
        p = factor * 50.3 / m_nu * vbar

        template = (vec[None, ...] * p[:, None, None]).reshape(-1, 3)

        return template

    def __init__(self, cosmology, BoxSize, Q, comm):
        self.cosmology = cosmology
        self.comm = comm
        self.BoxSize = numpy.zeros(Q.shape[-1], dtype='f8')
        self.BoxSize[...] = BoxSize

        self.dtype = Q.dtype

        template = self.generate_nu_template(cosmology.m_ncdm[0], 1, 10)

        nQ = len(Q)

        Q = numpy.repeat(Q, len(template), axis=0)
        self.Q = Q
        
        self.S = numpy.zeros_like(self.Q)
        self.P = numpy.repeat(template[None, ...], nQ, axis=0).reshape(-1, len(BoxSize))
        self.F = numpy.zeros_like(self.Q)
        self.RHO = numpy.zeros_like(self.Q[..., 0])
        self.a = dict(S=None, P=None, F=None)

        self.csize = self.comm.allreduce(len(self.Q))

    def Omega(self, a):
        # FIXME: using Omega_ncdm after switching to new nbodykit cosmology.
        return self.cosmology.Onu(z=1. / a - 1)

from collections import OrderedDict
class StateVector(object):
    def __init__(self, cosmology, species, comm):
        """ A state vector is a dict of Species """
        self.cosmology = cosmology
        self.species = OrderedDict(sorted(species.items(), key=lambda t: t[0]))
        self.comm = comm
        self.a = dict(S=None, P=None, F=None)

    def __getitem__(self, spname):
        return self.species[spname]

    def __iter__(self):
        return iter(self.species)

    def __contains__(self, name):
        return name in self.species

    def copy(self):
        vc = {}
        for k, v in self.species.items():
            vc[k] = v.copy()
        return StateVector(self.cosmology, vc, self.comm)

    def to_catalog(self, **kwargs):
        from nbodykit.source import MultipleSpeciesCatalog
        names = []
        sources = []

        for spname, sp in self.species.items():
            sources.append(sp.to_catalog())
            names.append(spname)

        print(kwargs)
        cat = MultipleSpeciesCatalog(names, *sources, **kwargs)
        return cat

    def save(self, filename, attrs={}):
        from bigfile import FileMPI
        a = self.a['S']
        with FileMPI(self.comm, filename, create=True) as ff:
            with ff.create('Header') as bb:
                keylist = ['Om0', 'Tcmb0', 'Neff', 'Ob0', 'Ode0']
                if getattr(self.cosmology, 'm_nu', None) is not None:
                    keylist.insert(3,'m_nu')
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

            for k, v in sorted(self.species.items()):
                v.save(filename, k)
