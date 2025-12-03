import numpy as np

N_CAMBER = 5
N_THICKNESS = 5
CONTROL_X = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32)

def sample_random_design(low: float = -0.02, high: float = 0.02) -> np.ndarray:
    """
    Return a random 10-D design vector:
    first 5 entries: camber offsets (Δc_i),
    next 5 entries: thickness offsets (Δt_i).
    """
    vec = np.random.uniform(low, high, size=(N_CAMBER + N_THICKNESS,))
    return vec.astype(np.float32)

def design_to_airfoil_coords(design_vec: np.ndarray, n_points: int = 200) -> np.ndarray:
    """
    Convert a 10-D design vector into airfoil surface coordinates.
    For Phase 0, just define the function and raise NotImplementedError.
    It will later:
      - start from a baseline NACA camber/thickness distribution,
      - add spline-based offsets from design_vec,
      - return coordinates as an array of shape (N, 2).
    """
    raise NotImplementedError("geometry generation not implemented yet")
