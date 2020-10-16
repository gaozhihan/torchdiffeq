import torch
from torchdiffeq.covar.solvers import FixedGridODESolver_Covar
from torchdiffeq._impl.rk_common import rk4_alt_step_func


class Euler(FixedGridODESolver_Covar):
    order = 1

    def __init__(self, eps=0., **kwargs):
        super(Euler, self).__init__(**kwargs)
        self.eps = torch.as_tensor(eps, dtype=self.dtype, device=self.device)

    def _step_func(self, func, t, dt, y):
        return dt * func(t + self.eps, y)


class Midpoint(FixedGridODESolver_Covar):
    order = 2

    def __init__(self, eps=0., **kwargs):
        super(Midpoint, self).__init__(**kwargs)
        self.eps = torch.as_tensor(eps, dtype=self.dtype, device=self.device)

    def _step_func(self, func, t, dt, y):
        half_dt = 0.5 * dt
        y_mid = y + func(t + self.eps, y) * half_dt
        return dt * func(t + half_dt, y_mid)


class RK4(FixedGridODESolver_Covar):
    order = 4

    def __init__(self, eps=0., **kwargs):
        super(RK4, self).__init__(**kwargs)
        self.eps = torch.as_tensor(eps, dtype=self.dtype, device=self.device)

    def _step_func(self, func, t, dt, y):
        return rk4_alt_step_func(func, t + self.eps, dt - 2 * self.eps, y)
