# This script tests if full reversability like
# JANUS is achieved by FastPM, without
# using int64 fixed point.
#
# JANUS: https://arxiv.org/pdf/1704.07715v1.pdf
#
# We do not need the high precision int64 representable
# because we only use very few steps (I think)
# Using float32 increases the error to about 1e-5.

from numpy.testing import assert_allclose

from fastpm.ncdm import Solver as SolverNCDM
from fastpm.core import Solver as Solver
from fastpm.core import leapfrog

from nbodykit.cosmology import Planck15, EHPower
from nbodykit.algorithms.fof import FOF
from nbodykit.algorithms.fftpower import FFTPower
from nbodykit.source import ArrayCatalog

from pmesh.pm import ParticleMesh
import numpy

pm = ParticleMesh(BoxSize=512, Nmesh=[256, 256, 256], dtype='f8', resampler='tsc')
Q = pm.generate_uniform_particle_grid()

stages = numpy.linspace(0.1, 1.0, 20, endpoint=True)

solver = Solver(pm, Planck15, B=2)
solver_ncdm = SolverNCDM(pm, Planck15, B=2)

wn = solver.whitenoise(400)
dlin = solver.linear(wn, EHPower(Planck15, 0))
lpt = solver.lpt(dlin, Q, stages[0])
#lpt.S = numpy.float32(lpt.S)

def monitor(action, ai, ac, af, state, event):
    if pm.comm.rank == 0:
        print(state.a['S'],  state.a['P'], state.a['F'], state.S[0], state.P[0], action, ai, ac, af)

state1 = solver.nbody(lpt.copy(), leapfrog(stages), monitor=monitor)

state2 = solver_ncdm.nbody(lpt.copy(), leapfrog(stages), monitor=monitor)

if pm.comm.rank == 0:
    print('----------------')

cat1 = ArrayCatalog({'Position' : state1.X}, BoxSize=pm.BoxSize, Nmesh=pm.Nmesh)
cat2 = ArrayCatalog({'Position' : state2.X}, BoxSize=pm.BoxSize, Nmesh=pm.Nmesh)
r1 = FFTPower(cat1, mode='1d')
r2 = FFTPower(cat2, mode='1d')

cat1.save('NCDM/core', ('Position',))
cat2.save('NCDM/ncdm', ('Position',))
r1.save('NCDM/core-power.json')
r2.save('NCDM/ncdm-power.json')

if pm.comm.rank == 0:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = Figure()
    canvas = FigureCanvasAgg(fig)

    ax = fig.add_subplot(111)

    ax.plot(r1.power['k'], r2.power['power'] / r1.power['power'], label='ncdm / core')

    fig.savefig('ncdm-result.png')
