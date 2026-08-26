import pytest

np = pytest.importorskip("numpy")

from cfdagent.geometry.airfoil_param import (
    design_to_airfoil_coords,
    to_selig_order,
)


def _naca0012_thickness(x: np.ndarray) -> np.ndarray:
    """Reference NACA 4-digit thickness distribution for a 12% section."""

    return 5.0 * 0.12 * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )


def test_baseline_design_reproduces_naca0012():
    coords = design_to_airfoil_coords(np.zeros(10))
    num = (len(coords) + 1) // 2
    upper_x, upper_y = coords[:num, 0], coords[:num, 1]

    expected = _naca0012_thickness(upper_x)
    assert np.allclose(upper_y, expected, atol=1e-6)
    assert np.ptp(coords[:, 1]) == pytest.approx(0.12, abs=2e-3)


def test_default_ordering_is_leading_edge_anchored():
    # Documents the module's own convention: the loop starts and ends at the LE.
    coords = design_to_airfoil_coords(np.zeros(10))
    assert coords[0] == pytest.approx([0.0, 0.0], abs=1e-6)
    assert coords[-1] == pytest.approx([0.0, 0.0], abs=1e-6)


def test_to_selig_order_starts_and_ends_at_trailing_edge():
    coords = design_to_airfoil_coords(np.zeros(10))
    selig = to_selig_order(coords)

    assert selig.shape == coords.shape
    # Selig order runs TE -> upper -> LE -> lower -> TE.
    assert selig[0, 0] == pytest.approx(1.0, abs=1e-6)
    assert selig[-1, 0] == pytest.approx(1.0, abs=1e-3)
    # The leading edge sits at the midpoint of the loop.
    mid = selig[len(selig) // 2]
    assert mid[0] == pytest.approx(0.0, abs=1e-6)
    # Upper surface comes first, so it should carry non-negative camber offset.
    assert selig[1, 1] > 0
    assert selig[-2, 1] < 0


def test_to_selig_order_preserves_point_set():
    coords = design_to_airfoil_coords(np.linspace(-0.01, 0.01, 10))
    selig = to_selig_order(coords)

    # Reordering must not add, drop, or alter any point.
    a = np.array(sorted(map(tuple, np.round(coords, 10))))
    b = np.array(sorted(map(tuple, np.round(selig, 10))))
    assert np.allclose(a, b)


def test_to_selig_order_rejects_bad_shape():
    with pytest.raises(ValueError):
        to_selig_order(np.zeros((5, 3)))
