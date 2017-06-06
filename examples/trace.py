from fastpm.core import Solver, leapfrog

from nbodykit.cosmology import Planck15, EHPower
from nbodykit.algorithms.fof import FOF
from nbodykit.source import ArrayCatalog

from pmesh.pm import ParticleMesh
import numpy

pm = ParticleMesh(BoxSize=128, Nmesh=[128, 128, 128])
Q = pm.generate_uniform_particle_grid()

stages = numpy.linspace(0.1, 1.0, 20, endpoint=True)

solver = Solver(pm, Planck15, B=2)
wn = solver.whitenoise(400)
dlin = solver.linear(wn, EHPower(Planck15, 0))
state = solver.lpt(dlin, Q, stages[0])

X = []

def monitor(action, ai, ac, af, state, event):
    if not state.synchronized: return
    X.append((state.a['S'], state.X.copy(), state.P.copy()))

state = solver.nbody(state, leapfrog(stages), monitor=monitor)

cat = ArrayCatalog({'Position' : X[-1][1]}, BoxSize=pm.BoxSize, Nmesh=pm.Nmesh)

fof = FOF(cat, linking_length=0.2, nmin=12)
for label in [1, 100, 1000, 5000]:
    select = fof.labels == label

    for i, (a, x, p) in enumerate(X):
        cat = ArrayCatalog({'Position' : x[select], 'Momentum' : p[select]}, BoxSize=pm.BoxSize, Time=a)
        cat.save('TraceSim/20-halo-%d-%06.4f' % (label, a), ('Position', 'Momentum'))
