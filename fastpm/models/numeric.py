import numpy

from abopt.vmad import VM

class MPINumeric:
    def __init__(self, comm):
        self._numeric_comm = comm

    @VM.microcode(aout=['y'], ain=['x'])
    def Multiply(self, x, f, y):
        y[...] = x * f

    @Multiply.grad
    def _(self, _x, _y, f):
        _x[...] = _y * f

    @VM.microcode(aout=['y'], ain=['x'])
    def CopyVariable(self, x, y):
        if hasattr(x, 'copy'):
            y[...] = x.copy()
        else:
            y[...] = 1.0 * x

    @CopyVariable.grad
    def _(self, _y, _x):
        _x[...] = _y

    @VM.microcode(aout=['chi2'], ain=['variable'])
    def Chi2(self, chi2, variable):
        comm = self._numeric_comm

        variable = variable * variable
        if hasattr(variable, "csum"):
            # use the collective csum method
            chi2[...] = variable.csum()
        else:
            # sum and collect for compatability
            s = variable.sum(dtype='f8')
            chi2[...] = comm.allreduce(s)

    @Chi2.grad
    def _(self, _variable, _chi2, variable):
        _variable[...] = variable * (2 * _chi2)


