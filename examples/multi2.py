from numpy.testing import assert_allclose

from fastpm.multi import Solver as SolverMulti
from fastpm.state import Baryon, CDM, NCDM, TiledNCDM
from fastpm.core import Solver as Solver
from fastpm.core import leapfrog

from nbodykit.cosmology import Planck15, LinearPower
from nbodykit.algorithms.fof import FOF
from nbodykit.algorithms.fftpower import FFTPower
from nbodykit.source import ArrayCatalog, MultipleSpeciesCatalog
from fastpm.multi import get_species_transfer_function_from_class

from pmesh.pm import ParticleMesh
import numpy

#Planck15 = Planck15.clone(gauge='newtonian')

pm = ParticleMesh(BoxSize=1024, Nmesh=[128, 128, 128], dtype='f4', resampler='tsc')
Q = pm.generate_uniform_particle_grid()

stages = numpy.linspace(0.1, 1.0, 10, endpoint=True)
#stages = [1.]

solver_multi = SolverMulti(pm, Planck15, B=2)

wn = solver_multi.whitenoise(400, unitary=True)

def monitor_multi(action, ai, ac, af, state, event):
    if pm.comm.rank == 0:
        print(state.a['S'],  state.a['P'], state.a['F'], state['1'].S[0], state['1'].P[0], action, ai, ac, af)

tf = get_species_transfer_function_from_class(Planck15, 1.0 / stages[0] - 1)

prm = solver_multi.primordial(wn, Planck15.Primordial.get_pkprim)

ic = solver_multi.lpt(prm, {
            '0': (Baryon, tf['d_b'], tf['dd_b']),
            '1': (CDM, tf['d_cdm'], tf['dd_cdm']),
            '4': (TiledNCDM, tf['d_ncdm[0]'], tf['dd_ncdm[0]']),
        }, Q, a=stages[0], order=2)

state2 = solver_multi.nbody(ic.copy(), leapfrog(stages), monitor=monitor_multi)

if pm.comm.rank == 0:
    print('----------------')

cat2 = state2.to_catalog(Nmesh=pm.Nmesh)
cat2_cdm = state2['1'].to_catalog(Nmesh=pm.Nmesh)
cat2_b = state2['0'].to_catalog(Nmesh=pm.Nmesh)
cat2_ncdm = state2['4'].to_catalog(Nmesh=pm.Nmesh)
print(cat2.attrs)

r2 = FFTPower(cat2, mode='1d')
r2_cdm = FFTPower(cat2_cdm, mode='1d')
r2_b = FFTPower(cat2_b, mode='1d')
r2_ncdm = FFTPower(cat2_ncdm, mode='1d')

cat2.save('MULTI/mult', ('0/Position','1/Position', '4/Position'))

r2.save('MULTI/mult-power.json')

if pm.comm.rank == 0:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = Figure()
    canvas = FigureCanvasAgg(fig)

    ax = fig.add_subplot(111)

    ax.plot(r2.power['k'], r2_b.power['power'] / r2.power['power'], label='b / tot')
    ax.plot(r2.power['k'], r2_cdm.power['power'] / r2.power['power'], label='cdm/ tot')
    ax.plot(r2.power['k'], r2_ncdm.power['power'] / r2.power['power'], label='ncdm / tot')
    ax.legend()
    ax.set_xscale('log')
    fig.savefig('multi-result.png')
