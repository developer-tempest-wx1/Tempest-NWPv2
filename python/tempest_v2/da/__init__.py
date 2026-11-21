"""Tempest-NWP v2 Data Assimilation Suite (0% AI / 0% ML).

Classical variational & nudging methods with verified discrete adjoints:
- balance.py  : Hydrostatic + mid-latitude geostrophic balance control transform K and K^T
- var3d.py    : Fast vectorized SPD recursive Gaussian filter, trilinear 3D/4D-FGAT
                forward/adjoint operators (fixed ZTD, T2m, soil_m, LCC winds), PCG solver & IAU
- lhn_radar.py: Latent Heat Nudging (LHN) for 1 km KNMI QPE / DWD RADOLAN rainfall rates
"""

from tempest_v2.da.balance import BalanceOperator
from tempest_v2.da.lhn_radar import compute_lhn_tendencies
from tempest_v2.da.var3d import (
    RecursiveSpatialFilter,
    Var3DConfig,
    Var3DResult,
    VarObservation,
    VarObservationOperator,
    VarStateVec,
    apply_analysis_increments_with_bounds,
    compute_iau_tendencies,
    solve_3dvar_pcg,
)

__all__ = [
    "BalanceOperator",
    "RecursiveSpatialFilter",
    "Var3DConfig",
    "Var3DResult",
    "VarObservation",
    "VarObservationOperator",
    "VarStateVec",
    "apply_analysis_increments_with_bounds",
    "compute_iau_tendencies",
    "compute_lhn_tendencies",
    "solve_3dvar_pcg",
]
