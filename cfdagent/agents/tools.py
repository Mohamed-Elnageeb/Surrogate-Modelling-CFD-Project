from ..cfd.run_cfd import run_cfd
import uuid


def cfd_tool(design_vec):
    """
    Tool wrapper for running CFD. Accepts a list or numpy array of 10 floats.
    Returns a dict with simulation results.
    """
    design_id = str(uuid.uuid4())[:8]
    return run_cfd(design_id, design_vec)
