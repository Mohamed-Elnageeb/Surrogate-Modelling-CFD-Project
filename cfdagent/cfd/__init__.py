"""Simple SU2 runner wrapper for CFD surrogate modeling."""

from .run_cfd import Su2RunConfig, run_su2_case

__all__ = ["Su2RunConfig", "run_su2_case"]
