import abc
import torch
from scipy.integrate import solve_ivp
from .misc import _handle_unused_kwargs


class ScipyWrapperODESolver(metaclass=abc.ABCMeta):

    def __init__(self, func, y0, rtol, atol, solver="LSODA", **unused_kwargs):
        unused_kwargs.pop('norm', None)
        unused_kwargs.pop('grid_points', None)
        unused_kwargs.pop('eps', None)
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.dtype = y0.dtype
        self.device = y0.device
        self.shape = y0.shape
        self.y0 = y0.detach().cpu().numpy().reshape(-1)
        self.rtol = rtol
        self.atol = atol
        self.solver = solver
        self.func = convert_func_to_numpy(func, self.device, self.dtype)

    def integrate(self, t):
        t = t.detach().cpu().numpy()
        sol = solve_ivp(
            self.func,
            t_span=[t.min(), t.max()],
            y0=self.y0,
            t_eval=t,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol,
        )
        sol = torch.tensor(sol.y).T.to(self.device, self.dtype)
        if len(self.shape) == 0:
            # handle 0D Tensors.
            sol = sol.reshape(-1)
        return sol


def convert_func_to_numpy(func, device, dtype):

    def np_func(t, y):
        t = torch.tensor(t).to(device, dtype)
        y = torch.tensor(y).to(device, dtype)
        with torch.no_grad():
            f = func(t, y)
        return f.detach().cpu().numpy()

    return np_func