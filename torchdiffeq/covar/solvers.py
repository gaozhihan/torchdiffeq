import abc
import torch
from torchdiffeq._impl.misc import _handle_unused_kwargs
from torchdiffeq._impl.solvers import FixedGridODESolver, AdaptiveStepsizeODESolver

class AdaptiveStepsizeODESolver_Covar(AdaptiveStepsizeODESolver, metaclass=abc.ABCMeta):
    def __init__(self, dtype, y0, norm, covar_func, ret_covar=False, **unused_kwargs):
        super(AdaptiveStepsizeODESolver_Covar, self).__init__(dtype, y0, norm, **unused_kwargs)
        self.covar_func = covar_func
        self.ret_covar = ret_covar

    def update_func(self, y1):
        z1 = self.covar_func(y1)
        self.func.update_by_covar(z1)
        return z1

    def integrate(self, t):
        solution = torch.empty(len(t), *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device)
        solution[0] = self.y0
        covar0 = self.covar_func(self.y0)
        if self.ret_covar:
            covar_list = torch.empty(len(t), *covar0.shape, dtype=covar0.dtype, device=covar0.device)
            covar_list[0] = covar0
        t = t.to(self.dtype)
        self._before_integrate(t)
        for i in range(1, len(t)):
            solution[i] = self._advance(t[i])
            if self.ret_covar:
                covar_list[i] = self.covar_func(solution[i])
        if self.ret_covar:
            return solution, covar_list
        else:
            return solution, None

class FixedGridODESolver_Covar(FixedGridODESolver, metaclass=abc.ABCMeta):
    def __init__(self, func, y0, covar_func, ret_covar=False,
                 step_size=None, grid_constructor=None, **unused_kwargs):
        super(FixedGridODESolver_Covar, self).__init__(func, y0, step_size, grid_constructor, **unused_kwargs)
        self.covar_func = covar_func
        self.ret_covar = ret_covar

    def update_func(self, y1):
        z1 = self.covar_func(y1)
        self.func.update_by_covar(z1)
        return z1

    def integrate(self, t):
        time_grid = self.grid_constructor(self.func, self.y0, t)
        assert time_grid[0] == t[0] and time_grid[-1] == t[-1]

        solution = torch.empty(len(t), *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device)
        solution[0] = self.y0
        covar0 = self.covar_func(self.y0)
        if self.ret_covar:
            covar_list = torch.empty(len(t), *covar0.shape, dtype=covar0.dtype, device=covar0.device)
            covar_list[0] = covar0

        j = 1
        y0 = self.y0
        for t0, t1 in zip(time_grid[:-1], time_grid[1:]):
            dy = self._step_func(self.func, t0, t1 - t0, y0)
            y1 = y0 + dy
            # update the covariate z using new y and update func using covariate z
            _ = self.update_func(y1)

            while j < len(t) and t1 >= t[j]:
                solution[j] = self._linear_interp(t0, t1, y0, y1, t[j])
                if self.ret_covar:
                    covar_list[j] = self.covar_func(solution[j])
                j += 1
            y0 = y1

        if self.ret_covar:
            return solution, covar_list
        else:
            return solution, None

    def _linear_interp(self, t0, t1, y0, y1, t):
        if t == t0:
            return y0
        if t == t1:
            return y1
        slope = (t - t0) / (t1 - t0)
        return y0 + slope * (y1 - y0)
