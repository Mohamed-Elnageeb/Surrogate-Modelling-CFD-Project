"""Simple SU2 runner wrapper for CFD surrogate modeling."""

from .run_cfd import Su2RunConfig, run_su2_case
from .param_sweep import CaseSample, generate_param_grid, run_param_sweep

__all__ = [
    "Su2RunConfig",
    "run_su2_case",
    "CaseSample",
    "generate_param_grid",
    "run_param_sweep",
]
