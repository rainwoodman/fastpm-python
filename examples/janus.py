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

from fastpm.core import Solver, leapfrog

from nbodykit.cosmology import Planck15, EHPower
from nbodykit.algorithms.fof import FOF
from nbodykit.source import ArrayCatalog

from pmesh.pm import ParticleMesh
import numpy

pm = ParticleMesh(BoxSize=64, Nmesh=[64, 64, 64], dtype='f8')
Q = pm.generate_uniform_particle_grid()

stages = numpy.linspace(0.1, 1.0, 10, endpoint=True)

solver = Solver(pm, Planck15, B=2)
wn = solver.whitenoise(400)
dlin = solver.linear(wn, EHPower(Planck15, 0))
lpt = solver.lpt(dlin, Q, stages[0])
#lpt.S = numpy.float32(lpt.S)

def monitor(action, ai, ac, af, state, event):
    if pm.comm.rank == 0:
        print(state.a['S'],  state.a['P'], state.a['F'], state.S[0], state.P[0], action, ai, ac, af)

def gorfpael(stages): 
    # Reversed Leap-Frog
    stack = []
    for action, ai, ac, af in list(leapfrog(stages))[::-1]:
        if action == 'K':
            # need to pop the F before K to ensure the correct force is used.
            stack.append((action, af, ac, ai))
        elif action == 'F':
            yield action, af, ac, ai
            for item in stack:
                yield item
            stack = []
        else:
            yield action, af, ac, ai

state = solver.nbody(lpt.copy(), leapfrog(stages), monitor=monitor)
if pm.comm.rank == 0:
    print('----------------')
reverse = solver.nbody(state, gorfpael(stages), monitor=monitor)
#print((lpt.X - reverse.X).max())

assert_allclose(lpt.X, reverse.X)

cat1 = ArrayCatalog({'Position' : lpt.X}, BoxSize=pm.BoxSize, Nmesh=pm.Nmesh)
cat2 = ArrayCatalog({'Position' : reverse.X}, BoxSize=pm.BoxSize, Nmesh=pm.Nmesh)

cat1.save('Janus/Truth', ('Position',))
cat2.save('Janus/Reverse', ('Position',))

