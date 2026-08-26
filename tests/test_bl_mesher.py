"""Tests for boundary-layer mesh sizing and generation."""

import pytest

np = pytest.importorskip("numpy")

from cfdagent.cfd.bl_mesher import BLMeshConfig, first_cell_height


def test_first_cell_height_matches_known_sizing():
    # At Re=1e6, y+=1 the flat-plate correlation gives ~2.4e-5 chords. The old
    # uniform mesh sat at ~9.6e-3, i.e. two-plus orders of magnitude too coarse.
    h = first_cell_height(1.0e6, y_plus=1.0)
    assert 2.0e-5 < h < 2.8e-5
    assert 9.55e-3 / h > 300


def test_first_cell_height_scales_with_reynolds_and_yplus():
    assert first_cell_height(1e7) < first_cell_height(1e6)
    assert first_cell_height(1e6, y_plus=2.0) == pytest.approx(
        2 * first_cell_height(1e6, y_plus=1.0)
    )


def test_bl_thickness_follows_geometric_growth():
    cfg = BLMeshConfig(n_layers=3, growth_ratio=2.0)
    h = cfg.first_layer()
    # h + 2h + 4h = 7h
    assert cfg.bl_thickness() == pytest.approx(7 * h)


def test_bl_thickness_handles_unit_growth():
    cfg = BLMeshConfig(n_layers=10, growth_ratio=1.0)
    assert cfg.bl_thickness() == pytest.approx(10 * cfg.first_layer())


def test_default_config_resolves_boundary_layer():
    cfg = BLMeshConfig()
    # The extrusion must be thick enough to contain a Re=1e6 airfoil BL
    # (order 1% of chord) while starting fine enough for y+ ~ 1.
    assert cfg.bl_thickness() > 0.01
    assert cfg.first_layer() < 5e-5
    assert cfg.farfield_radius >= 50


@pytest.mark.slow
def test_build_bl_mesh_produces_resolved_mesh(tmp_path):
    pytest.importorskip("gmsh")
    from cfdagent.cfd.bl_mesher import build_bl_mesh

    out = tmp_path / "m.su2"
    m = build_bl_mesh(np.zeros(10), out, BLMeshConfig(n_layers=12))
    assert out.exists()
    assert m["n_elements"] > 1000
    # Nodes must actually be placed inside the first cell height.
    assert m["wall_normal_min"] < m["first_layer"]
