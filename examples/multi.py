from numpy.testing import assert_allclose

from fastpm.multi import Solver as SolverMulti
from fastpm.state import Baryon, CDM, NCDM
from fastpm.core import Solver as Solver
from fastpm.core import leapfrog

from nbodykit.cosmology import Planck15, EHPower
from nbodykit.algorithms.fof import FOF
from nbodykit.algorithms.fftpower import FFTPower
from nbodykit.source import ArrayCatalog, MultipleSpeciesCatalog
from fastpm.multi import get_species_transfer_function_from_class

from pmesh.pm import ParticleMesh
import numpy

Planck15 = Planck15.clone(gauge='newtonian')

pm = ParticleMesh(BoxSize=512, Nmesh=[64, 64, 64], dtype='f4', resampler='tsc')
Q = pm.generate_uniform_particle_grid()

stages = numpy.linspace(0.1, 1.0, 10, endpoint=True)

solver = Solver(pm, Planck15, B=2)
solver_multi = SolverMulti(pm, Planck15, B=2)

wn = solver.whitenoise(400)
dlin = solver.linear(wn, EHPower(Planck15, 0))
lpt = solver.lpt(dlin, Q, stages[0])

def monitor(action, ai, ac, af, state, event):
    if pm.comm.rank == 0:
        print(state.a['S'],  state.a['P'], state.a['F'], state.S[0], state.P[0], action, ai, ac, af)

def monitor_multi(action, ai, ac, af, state, event):
    if pm.comm.rank == 0:
        print(state.a['S'],  state.a['P'], state.a['F'], state['1'].S[0], state['1'].P[0], action, ai, ac, af)

state1 = solver.nbody(lpt.copy(), leapfrog(stages), monitor=monitor)

tf = get_species_transfer_function_from_class(Planck15, 9)

prm = solver_multi.primordial(wn, Planck15.Primordial.get_pk)

ic = solver_multi.lpt(prm, {
            '0': (Baryon, tf['d_b'], tf['dd_b']),
            '1': (CDM, tf['d_cdm'], tf['dd_cdm']),
            '4': (NCDM, tf['d_ncdm[0]'], tf['dd_ncdm[0]']),
        }, Q, a=0.1)

state2 = solver_multi.nbody(ic.copy(), leapfrog(stages), monitor=monitor_multi)

if pm.comm.rank == 0:
    print('----------------')

cat1 = state1.to_catalog(Nmesh=pm.Nmesh)
cat2 = state2.to_catalog(Nmesh=pm.Nmesh)
print(cat2.attrs)

r1 = FFTPower(cat1, mode='1d')
r2 = FFTPower(cat2, mode='1d')

cat1.save('MULTI/core', ('Position',))
cat2.save('MULTI/mult', ('0/Position','1/Position', '4/Position'))

r1.save('MULTI/core-power.json')
r2.save('MULTI/mult-power.json')

if pm.comm.rank == 0:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = Figure()
    canvas = FigureCanvasAgg(fig)

    ax = fig.add_subplot(111)

    ax.plot(r1.power['k'], r2.power['power'] / r1.power['power'], label='ncdm / core')

    fig.savefig('ncdm-result.png')
