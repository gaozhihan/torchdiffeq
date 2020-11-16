import torch
from torchdiffeq._impl.solvers import FixedGridODESolver
from torchdiffeq._impl.rk_common import rk4_alt_step_func, _runge_kutta_step, _ButcherTableau
from torchdiffeq._impl.dopri5 import _DORMAND_PRINCE_SHAMPINE_TABLEAU
from torchdiffeq._impl.misc import _compute_error_ratio, _check_inputs, _flat_to_shape, _rms_norm


class RK4Error(FixedGridODESolver):
    order = 4
    tableau = _DORMAND_PRINCE_SHAMPINE_TABLEAU

    def __init__(self, eps=0., rtol=1e-7, atol=1e-9, **kwargs):
        super(RK4Error, self).__init__(**kwargs)
        self.eps = torch.as_tensor(eps, dtype=self.dtype, device=self.device)
        self.rtol = rtol
        self.atol = atol
        device = self.device
        y0 = self.y0
        self.tableau = _ButcherTableau(alpha=self.tableau.alpha.to(device=device, dtype=y0.dtype),
                                       beta=[b.to(device=device, dtype=y0.dtype) for b in self.tableau.beta],
                                       c_sol=self.tableau.c_sol.to(device=device, dtype=y0.dtype),
                                       c_error=self.tableau.c_error.to(device=device, dtype=y0.dtype))
        self.norm = _rms_norm
        self.total_error_ratio = []

    def _step_func(self, func, t, dt, y):
        f = func(t, y)
        y1, f1, y1_error, k = _runge_kutta_step(func, y, f, t, dt, self.tableau)
        error_ratio = _compute_error_ratio(y1_error, self.rtol, self.atol, y, y1, self.norm)
        # print("%.4f" % error_ratio.item())
        self.total_error_ratio.append(error_ratio)
        accept_step = error_ratio <= 1
        return y1
        # return rk4_alt_step_func(func, t + self.eps, dt - 2 * self.eps, y)

SOLVERS = {
    "rk4": RK4Error
}

def odeint_rk4error(func, y0, t, rtol=1e-7, atol=1e-9, method=None, options=None, return_error_ratio=False):
    shapes, func, y0, t, rtol, atol, method, options = _check_inputs(func, y0, t, rtol, atol, method, options, SOLVERS)

    solver = RK4Error(func=func, y0=y0, rtol=rtol, atol=atol, **options)
    solution = solver.integrate(t)

    if shapes is not None:
        solution = _flat_to_shape(solution, (len(t),), shapes)
    if return_error_ratio:
        return solution, solver.total_error_ratio
    else:
        return solution
