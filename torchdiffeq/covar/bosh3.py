from torchdiffeq._impl.bosh3 import _BOGACKI_SHAMPINE_TABLEAU, _BS_C_MID
from torchdiffeq.covar.rk_common import RKAdaptiveStepsizeODESolver_Covar

class Bosh3Solver_Covar(RKAdaptiveStepsizeODESolver_Covar):
    order = 3
    tableau = _BOGACKI_SHAMPINE_TABLEAU
    mid = _BS_C_MID