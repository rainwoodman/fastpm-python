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
from nbodykit.lab import FOF
from nbodykit.lab import ArrayCatalog

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
    # Reversed Leap-Frog, the verbose way
    # this is identical to leapfrog(stages[::-1])
    stack = []

    # compute the force first as we didn't save it
    yield ('F', stages[-1], stages[-1], stages[-1])

    for action, ai, ac, af in list(leapfrog(stages))[::-1]:
        if action == 'D':
            yield action, af, ac, ai
            for action, af, ac, ai in stack:
                assert action == 'F'
                yield action, af, ac, ai
            stack = []
        elif action == 'F':
            # need to do F after D to ensure the time tag is right.
            assert ac == af
            stack.append((action, af, ai, ai))
        else:
            yield action, af, ac, ai

print(list(leapfrog(stages)))
print('----')
print(list(gorfpael(stages)))
print('++++')
print(list(leapfrog(stages[::-1])))

state = solver.nbody(lpt.copy(), leapfrog(stages), monitor=monitor)
if pm.comm.rank == 0:
    print('----------------')
reverse = solver.nbody(state, leapfrog(stages[::-1]), monitor=monitor)
#print((lpt.X - reverse.X).max())

assert_allclose(lpt.X, reverse.X)

cat1 = ArrayCatalog({'Position' : lpt.X}, BoxSize=pm.BoxSize, Nmesh=pm.Nmesh)
cat2 = ArrayCatalog({'Position' : reverse.X}, BoxSize=pm.BoxSize, Nmesh=pm.Nmesh)

cat1.save('Janus/Truth', ('Position',))
cat2.save('Janus/Reverse', ('Position',))

