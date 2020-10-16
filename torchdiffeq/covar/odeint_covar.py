from torchdiffeq._impl.misc import _flat_to_shape
from torchdiffeq.covar.fixed_grid import Euler, Midpoint, RK4
from torchdiffeq.covar.dopri5 import Dopri5Solver
from torchdiffeq.covar.misc import _check_inputs_covar

SOLVERS = {
    'euler': Euler,
    'midpoint': Midpoint,
    'rk4': RK4,
    'dopri5': Dopri5Solver
}

def odeint_covar(func, y0, t, covar_func, ret_covar=False, rtol=1e-7, atol=1e-9, method=None, options=None):
    """Integrate a system of ordinary differential equations.

    Solves the initial value problem for a non-stiff system of first order ODEs:
        ```
        dy/dt = func(t, y), y(t[0]) = y0
        ```
    where y is a Tensor or tuple of Tensors of any shape.

    Output dtypes and numerical precision are based on the dtypes of the inputs `y0`.

    Args:
        func: torch.nn.Module. `func(t, y)` maps a scalar Tensor `t` and a Tensor holding the state `y`
            into a Tensor of state derivatives with respect to time. Optionally, `y`
            can also be a tuple of Tensors.
            must implement method 'update_by_covar(z)' to get updated when the covariate 'z' is updated.
        y0: N-D Tensor giving starting value of `y` at time point `t[0]`. Optionally, `y0`
            can also be a tuple of Tensors.
        t: 1-D Tensor holding a sequence of time points for which to solve for
            `y`. The initial time point should be the first element of this sequence,
            and each time must be larger than the previous time.
        covar_func: torch.nn.Module. `covar_func(y)` maps a Tensor holding the state `y` into a covariate Tensor `z`.
        rtol: optional float64 Tensor specifying an upper bound on relative error,
            per element of `y`.
        atol: optional float64 Tensor specifying an upper bound on absolute error,
            per element of `y`.
        ret_covar: bool. If True, solver will return the corresponding covariates as well as ODE solution.
        method: optional string indicating the integration method to use.
        options: optional dict of configuring options for the indicated integration
            method. Can only be provided if a `method` is explicitly set.

    Returns:
        solution: Tensor, where the first dimension corresponds to different
            time points. Contains the solved value of `y` for each desired time point in
            `t`, with the initial value `y0` being the first element along the first
            dimension.
        sol_covar:
            if ret_covar == True:
                it is a torch.Tensor, where the first dimension corresponds to different
                time points. Contains the solved value of `z` for each desired time point in
                `t`, with the initial value `z0 = covar_func(y0)` being the first element along the first
                dimension.
            if ret_covar == False:
                None

    Raises:
        ValueError: if an invalid `method` is provided.
    """
    shapes, func, y0, t, rtol, atol, method, options = _check_inputs_covar(func, y0, t, covar_func, rtol, atol, method, options, SOLVERS)

    solver = SOLVERS[method](func=func, y0=y0, covar_func=covar_func, ret_covar=ret_covar, rtol=rtol, atol=atol, **options)
    solution, sol_covar = solver.integrate(t)

    if shapes is not None:
        solution = _flat_to_shape(solution, (len(t),), shapes)
        if sol_covar is not None:
            sol_covar = _flat_to_shape(sol_covar, (len(t),), shapes)
    return solution, sol_covar

if __name__ == "__main__":
    import torch
    from torch import nn
    from torch.nn import init
    batch_size = 2
    feat_dim = 10
    t_len = 3
    class Func(nn.Module):
        def __init__(self, feat_dim=feat_dim):
            super(Func, self).__init__()
            self.net = nn.Linear(in_features=feat_dim,
                                 out_features=feat_dim,
                                 bias=False)
            self.covar = torch.ones(feat_dim)
            self.reset_params()
        def reset_params(self):
            init.ones_(self.net.weight)
            self.covar = torch.ones(batch_size, feat_dim)
            self.update_counter = 0

        def forward(self, t, x):
            return self.net(x) + self.covar
        def update_by_covar(self, z):
            self.covar = z
            self.update_counter += 1

    class CovarFunc(nn.Module):
        def __init__(self, feat_dim=feat_dim):
            super(CovarFunc, self).__init__()
            self.covar_net = nn.Linear(in_features=feat_dim,
                                       out_features=feat_dim,
                                       bias=False)
            self.reset_params()
        def reset_params(self):
            init.ones_(self.covar_net.weight)
        def forward(self, x):
            return self.covar_net(x)

    y0 = torch.ones(batch_size, feat_dim)
    t = torch.Tensor(range(t_len)) / t_len
    func = Func(feat_dim=feat_dim)
    covar_func = CovarFunc(feat_dim=feat_dim)
    solution_dopri5, sol_covar_dopri5 = odeint_covar(func=func, y0=y0, t=t, ret_covar=True,
                                                     covar_func=covar_func, method="dopri5")
    print(func.update_counter)
    # solution_dopri5 and solution_rk4 are different! The number of discrete steps may be different
    func.reset_params()
    covar_func.reset_params()

    # method = "euler"
    # method = "midpoint"
    method="rk4"
    # method = "dopri5"
    solution, sol_covar = odeint_covar(func=func, y0=y0, t=t, ret_covar=True,
                                       covar_func=covar_func, method=method)
    print(func.update_counter)