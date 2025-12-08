import numpy as np

N_CAMBER = 5
N_THICKNESS = 5
CONTROL_X = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32)


def sample_random_design(low: float = -0.02, high: float = 0.02, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Return a random 10-D design vector:
    first 5 entries: camber offsets (Δc_i),
    next 5 entries: thickness offsets (Δt_i).

    Args:
        low: Lower bound for uniform sampling.
        high: Upper bound for uniform sampling.
        rng: Optional :class:`numpy.random.Generator` for reproducibility.
    """
    if rng is None:
        rng = np.random.default_rng()
    vec = rng.uniform(low, high, size=(N_CAMBER + N_THICKNESS,))
    return vec.astype(np.float32)


def _smooth_control_points(ctrl: np.ndarray) -> np.ndarray:
    """Apply a lightweight moving-average filter to dampen jagged shapes."""

    kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    padded = np.pad(ctrl.astype(np.float32), (1, 1), mode="edge")
    smoothed = (
        kernel[0] * padded[:-2]
        + kernel[1] * padded[1:-1]
        + kernel[2] * padded[2:]
    )
    return smoothed


def design_to_airfoil_coords(design_vec: np.ndarray, n_points: int = 200) -> np.ndarray:
    """
    Convert a 10-D design vector into airfoil surface coordinates.

    A smooth, cosine-spaced chordwise distribution is generated and modified by
    the camber and thickness control points contained in ``design_vec``. A
    NACA-0012 thickness law is used as the baseline; thickness control points
    act as additive offsets on the thickness ratio while camber control points
    directly perturb the camber line.

    Args:
        design_vec: Array of shape ``(10,)`` with 5 camber and 5 thickness values.
        n_points: Number of surface points to return (upper + lower surfaces).

    Returns:
        ``(2 * n_points - 1, 2)`` array of ``[x, y]`` coordinates ordered from
        leading edge along the upper surface and back along the lower surface.
    """
    design_vec = np.asarray(design_vec, dtype=np.float32).flatten()
    if design_vec.size != N_CAMBER + N_THICKNESS:
        raise ValueError(f"Expected design vector of length {N_CAMBER + N_THICKNESS}, got {design_vec.size}")
    if n_points < 10:
        raise ValueError("n_points should be at least 10 for a smooth airfoil")

    # Cosine spacing along the chord for better leading-edge resolution.
    theta = np.linspace(0.0, np.pi, n_points)
    x = 0.5 * (1.0 - np.cos(theta))

    camber_ctrl = _smooth_control_points(design_vec[:N_CAMBER])
    thickness_ctrl = _smooth_control_points(design_vec[N_CAMBER:])

    camber_offsets = np.interp(x, CONTROL_X, camber_ctrl, left=camber_ctrl[0], right=camber_ctrl[-1])
    thickness_offsets = np.interp(x, CONTROL_X, thickness_ctrl, left=thickness_ctrl[0], right=thickness_ctrl[-1])

    # Baseline NACA-0012 thickness distribution with additive offsets.
    baseline_t = 0.12
    t_dist = np.clip(baseline_t + thickness_offsets, 0.02, 0.30)
    y_t = 5.0 * t_dist * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )

    y_c = camber_offsets
    y_upper = y_c + y_t
    y_lower = y_c - y_t

    upper_surface = np.stack([x, y_upper], axis=1)
    lower_surface = np.stack([x[::-1], y_lower[::-1]], axis=1)

    coords = np.concatenate([upper_surface, lower_surface[1:]], axis=0)
    return coords.astype(np.float32)
