"""Tests for the experimental-data loaders and the transition study."""

import pytest

np = pytest.importorskip("numpy")

from cfdagent.validation.experimental import load_abbott, load_gregory, load_ladson


def test_ladson_is_tripped_at_re6e6():
    d = load_ladson()
    assert d.tripped is True
    assert d.transition == "tripped"
    assert d.reynolds == pytest.approx(6e6)
    assert d.cd is not None and d.alpha is not None
    assert len(d) > 40


def test_abbott_is_untripped():
    d = load_abbott()
    assert d.tripped is False
    assert d.transition == "free"
    assert d.cd is not None
    # Stored sorted by lift so interpolation on Cl is well defined.
    assert np.all(np.diff(d.cl) >= 0)


def test_gregory_has_lift_only():
    d = load_gregory()
    assert d.tripped is True
    assert d.reynolds == pytest.approx(3e6)
    assert d.cd is None


def test_experimental_values_are_physical():
    for d in (load_ladson(), load_abbott()):
        assert np.all(d.cd > 0), "drag must be positive"
        assert np.all(np.abs(d.cl) < 2.5)
        # Ladson sweeps past stall (alpha to 19.3 deg), where Cd reaches ~0.43.
        # Filtering on lift does not isolate attached flow because Cl falls back
        # after stall, so bound the attached range by incidence where available.
        if d.alpha is not None:
            attached = d.cd[np.abs(d.alpha) <= 12.0]
        else:
            attached = d.cd
        assert np.all(attached < 0.05)


def test_ladson_matches_published_reference_point():
    # NASA TM 4074: alpha = 4.04 deg -> Cl = 0.4316, Cd = 0.00823
    d = load_ladson()
    i = int(np.argmin(np.abs(d.alpha - 4.04)))
    assert d.cl[i] == pytest.approx(0.4316, abs=1e-3)
    assert d.cd[i] == pytest.approx(0.00823, abs=1e-4)


def test_tripped_drag_exceeds_untripped_at_matched_lift():
    """The physical premise of the study: tripping the boundary layer adds drag."""

    trip, free = load_ladson(), load_abbott()
    order = np.argsort(free.cl)
    for target in (0.2, 0.4, 0.6):
        i = int(np.argmin(np.abs(trip.cl - target)))
        free_cd = np.interp(trip.cl[i], free.cl[order], free.cd[order])
        assert trip.cd[i] > free_cd


@pytest.mark.slow
def test_transition_study_reproduces_reported_ratio():
    pytest.importorskip("neuralfoil")
    from cfdagent.validation.transition_study import run

    r = run()
    # Drag error against the tripped experiment is several times larger.
    assert r["drag_tripped"].mean_abs_pct > r["drag_free"].mean_abs_pct
    assert r["drag_ratio"] > 2.0
    # Bias is negative: a transition-modelling tool cannot predict tripped drag.
    assert r["drag_tripped"].bias_pct < 0
    # Lift is far less sensitive than drag.
    assert r["lift_tripped"].mean_abs_pct < r["drag_tripped"].mean_abs_pct
