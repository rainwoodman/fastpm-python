from fastpm import operators
from abopt.vmad import VM, Zero
class LPT:
    def __init__(self, pm, q):
        self._lpt_pm = pm
        self._lpt_q = q

    @VM.microcode(aout=['s', 'p'], ain=['dlin_k'])
    def LPTDisplace(self, dlin_k, s, p, D1, v1, D2, v2):
        q = self._lpt_q
        dx1 = operators.lpt1(dlin_k, q)
        source = operators.lpt2source(dlin_k)
        dx2 = operators.lpt1(source, q)
        s[...] = D1 * dx1 + D2 * dx2
        p[...] = v1 * dx1 + v2 * dx2

    @LPTDisplace.grad
    def _(self, _dlin_k, dlin_k, _s, _p, D1, v1, D2, v2):
        pm = self._lpt_pm

        q = self._lpt_q
        grad_dx1 = _p * v1 + _s * D1
        grad_dx2 = _p * v2 + _s * D2

        if grad_dx1 is not Zero:
            gradient = operators.lpt1_gradient(pm, q, grad_dx1)
        else:
            gradient = Zero

        if grad_dx2 is not Zero:
            gradient_lpt2source = operators.lpt1_gradient(pm, q, grad_dx2)
            gradient[...] +=  operators.lpt2source_gradient(dlin_k, gradient_lpt2source)

        _dlin_k[...] = gradient

