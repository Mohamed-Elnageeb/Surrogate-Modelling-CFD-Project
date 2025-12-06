from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from cfdagent.surrogate.snapshot_builder import (
    SnapshotConfig,
    build_field_tensor,
    extract_forces,
    read_su2_table,
    save_snapshot_from_arrays,
    su2_to_unet_snapshot,
)


def test_read_su2_table_parses_header_and_data(tmp_path: Path):
    volume_path = tmp_path / "volume.dat"
    volume_path.write_text(
        """
# Comment line
% Another comment
x,y,p,u,v
0.0,0.0,1.0,0.1,0.2
1.0,0.0,2.0,0.3,0.4
0.0,1.0,3.0,0.5,0.6
1.0,1.0,4.0,0.7,0.8
""".strip()
    )

    table = read_su2_table(volume_path)

    for key in ["x", "y", "p", "u", "v"]:
        assert key in table
        assert isinstance(table[key], np.ndarray)
        assert table[key].shape == (4,)
    assert table["p"][2] == 3.0
    assert table["u"][3] == 0.7


def test_read_su2_table_handles_inline_comments(tmp_path: Path):
    volume_path = tmp_path / "volume.dat"
    volume_path.write_text(
        """
% Header comment
x, y, p
0.0, 0.0, 1.0  # leading point
1.0, 0.0, 2.0  % another point
""".strip()
    )

    table = read_su2_table(volume_path)

    assert table["p"].shape == (2,)
    assert table["p"][0] == pytest.approx(1.0)
    assert table["p"][1] == pytest.approx(2.0)


def test_read_su2_table_ignores_trailing_text_tokens(tmp_path: Path):
    volume_path = tmp_path / "volume.dat"
    volume_path.write_text(
        """
x, y, p
0.0, 0.0, 1.0, , , Trailing text
1.0, 0.0, 2.0, success
""".strip()
    )

    table = read_su2_table(volume_path)

    assert table["p"].tolist() == [1.0, 2.0]


def test_read_su2_table_pads_and_trims_rows(tmp_path: Path):
    volume_path = tmp_path / "volume.dat"
    volume_path.write_text(
        """
x, y, p
0.0, 0.0
1.0, 0.0, 2.0, 4.0, 5.0
""".strip()
    )

    table = read_su2_table(volume_path)

    assert table["x"].tolist() == [0.0, 1.0]
    assert table["y"].tolist() == [0.0, 0.0]
    assert np.isnan(table["p"][0])
    assert table["p"][1] == pytest.approx(2.0)


def test_read_su2_table_handles_non_numeric_and_missing_values(tmp_path: Path):
    volume_path = tmp_path / "volume.dat"
    volume_path.write_text(
        """
design_id,dc1,dc2,dc3,dc4,dc5,dt1,dt2,dt3,dt4,dt5,Cl,Cd,residual,success,error,snapshot_error
34eb1a35,-0.0172,0.0003,-0.0194,-0.0005,-0.0149,0.0079,0.0079,-0.0149,0.0064,0.0121,,,,True,,Row column count does not match header
""".strip()
    )

    table = read_su2_table(volume_path)

    assert table["design_id"].shape == (1,)
    assert np.isnan(table["design_id"][0])
    assert table["dc1"][0] == pytest.approx(-0.0172)
    assert np.isnan(table["Cl"][0])


def test_build_field_tensor_stacks_and_reshapes(tmp_path: Path):
    volume_path = tmp_path / "volume.dat"
    volume_path.write_text(
        """
# Comment line
x,y,p,u,v
0.0,0.0,1.0,0.1,0.2
1.0,0.0,2.0,0.3,0.4
0.0,1.0,3.0,0.5,0.6
1.0,1.0,4.0,0.7,0.8
""".strip()
    )
    table = read_su2_table(volume_path)

    tensor = build_field_tensor(table, ["p", "u"], grid_shape=(2, 2))
    assert tensor.shape == (2, 2, 2)
    assert np.allclose(tensor[0].flatten(), table["p"])
    assert np.allclose(tensor[1].flatten(), table["u"])

    with pytest.raises(ValueError):
        build_field_tensor(table, ["p", "u"], grid_shape=(3, 2))


def test_extract_forces_uses_last_entry():
    table = {
        "CL": np.array([0.1, 0.2, 0.3], dtype=float),
        "CD": np.array([0.01, 0.02, 0.03], dtype=float),
    }

    cl, cd = extract_forces(table, "CL", "CD")
    assert cl == pytest.approx(0.3)
    assert cd == pytest.approx(0.03)


def test_su2_to_unet_snapshot_creates_expected_npz(tmp_path: Path):
    volume_path = tmp_path / "volume.dat"
    surface_path = tmp_path / "surface.dat"
    out_path = tmp_path / "snap.npz"

    volume_path.write_text(
        """
# Comment line
x,y,p,u,v
0.0,0.0,1.0,0.1,0.2
1.0,0.0,2.0,0.3,0.4
0.0,1.0,3.0,0.5,0.6
1.0,1.0,4.0,0.7,0.8
""".strip()
    )
    surface_path.write_text(
        """
# Surface data
CL,CD
0.5,0.01
0.6,0.02
""".strip()
    )

    cfg = SnapshotConfig(
        input_fields=["p", "u"],
        target_fields=["v"],
        grid_shape=(2, 2),
        cl_name="CL",
        cd_name="CD",
    )

    su2_to_unet_snapshot(volume_path, surface_path, out_path, cfg)

    assert out_path.exists()
    with np.load(out_path) as data:
        assert set(data.keys()) == {"input", "target_fields", "cl", "cd"}
        assert data["input"].shape == (2, 2, 2)
        assert data["target_fields"].shape == (1, 2, 2)
        assert float(data["cl"]) == pytest.approx(0.6)
        assert float(data["cd"]) == pytest.approx(0.02)


def test_save_snapshot_from_arrays_roundtrip(tmp_path: Path):
    out_path = tmp_path / "snap.npz"
    input_arr = np.zeros((2, 2, 2), dtype=np.float32)
    target_arr = np.ones((1, 2, 2), dtype=np.float32)

    save_snapshot_from_arrays(out_path, input_arr, target_arr, cl=0.5, cd=0.1)

    with np.load(out_path) as data:
        assert data["input"].shape == (2, 2, 2)
        assert data["target_fields"].shape == (1, 2, 2)
        assert float(data["cl"]) == pytest.approx(0.5)
        assert float(data["cd"]) == pytest.approx(0.1)

    with pytest.raises(ValueError):
        save_snapshot_from_arrays(out_path, input_arr.reshape(2, 4), target_arr, cl=0.5, cd=0.1)
    with pytest.raises(ValueError):
        save_snapshot_from_arrays(out_path, input_arr, target_arr.reshape(4, 1), cl=0.5, cd=0.1)
