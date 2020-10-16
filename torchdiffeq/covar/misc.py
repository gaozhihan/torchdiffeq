import torch
from torch import nn
import warnings
from torchdiffeq._impl.misc import _check_inputs, _decreasing

def _check_inputs_covar(func, y0, t, covar_func, rtol, atol, method, options, SOLVERS):
    assert isinstance(func, nn.Module)
    # func must has method `func.update_by_covar(z)` to get updated when the covariate `z` is updated
    # method name can be found in `FixedGridODESolver_CoVar.update_func()` and `AdaptiveStepsizeODESolver_CoVar.update_func()`
    required_method = "update_by_covar"
    assert hasattr(func, required_method) and callable(getattr(func, required_method)), \
        "func must implement method {}".format(required_method)
    assert isinstance(covar_func, nn.Module)

    # to avoid bugs that arise from converting the `func`
    assert not _decreasing(t), "t must be increasing Tensor"

    # `y0` can not be a tuple of Tensors like in the original torchdiffeq,
    # since different `y0` leads to different parameters of func, which makes it impossible to parallel.
    assert torch.is_tensor(y0), "y0 must be a torch.Tensor"
    if not torch.is_floating_point(y0):
        raise TypeError('`y0` must be a floating point Tensor but is a {}'.format(y0.type()))
    if not torch.is_floating_point(t):
        raise TypeError('`t` must be a floating point Tensor but is a {}'.format(t.type()))

    shapes, func, y0, t, rtol, atol, method, options = _check_inputs(func, y0, t, rtol, atol, method, options, SOLVERS)
    return shapes, func, y0, t, rtol, atol, method, options