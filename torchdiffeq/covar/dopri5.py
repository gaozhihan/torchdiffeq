from torchdiffeq._impl.dopri5 import _DORMAND_PRINCE_SHAMPINE_TABLEAU, DPS_C_MID
from torchdiffeq.covar.rk_common import RKAdaptiveStepsizeODESolver_Covar

class Dopri5Solver_Covar(RKAdaptiveStepsizeODESolver_Covar):
    order = 5
    tableau = _DORMAND_PRINCE_SHAMPINE_TABLEAU
    mid = DPS_C_MID
