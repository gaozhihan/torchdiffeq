import torch
from torch import nn
from torch.nn import init
from torchdiffeq.tmp.odeint import odeint_rk4error
from torchdiffeq import odeint

feat_dim = 4
num_step = 10

class TmpFunc(nn.Module):
    def __init__(self, feat_dim):
        super(TmpFunc, self).__init__()
        self.linear = nn.Linear(in_features=feat_dim, out_features=feat_dim)
        init.zeros_(self.linear.weight)

    def forward(self, t, y):
        return self.linear(y)

func = TmpFunc(feat_dim=feat_dim)

y0 = torch.ones(size=(feat_dim, ))
t = torch.arange(num_step).double() / num_step
# y = odeint(func, y0, t, rtol=1e-7, atol=1e-9, method="dopri5", options=None)
y = odeint_rk4error(func, y0, t, rtol=1e-7, atol=1e-9, method="rk4", options=None)