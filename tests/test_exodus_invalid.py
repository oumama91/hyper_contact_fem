"""
Rejets explicites du lecteur Exodus (F-007) : maillage vide, coordonnées non finies, variables
masquées, formes incohérentes et jacobien NaN.

Chaque cas part d'un petit cube HEX8 valide écrit avec ``netCDF4`` et n'en modifie qu'un point :
une seule règle de validation est ainsi testée à la fois. Aucun solveur n'est lancé.
"""
from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy as np
import pytest

from exodus_mesh import _check_mesh
from exodus_reader import ExodusMesh, read_exodus

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_MESH = ROOT / "mesh" / "O-ring.exo"
LEN_NAME = 64

# netCDF4 1.7.4 affecte encore ``data.shape`` (internal) : dépréciation NumPy 2.5 sans effet ici.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated:DeprecationWarning"
)

# Cube unité HEX8, nœuds dans l'ordre Exodus (1-based pour la connectivité du fichier).
CUBE_POINTS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ]
)
CUBE_CELLS = np.arange(1, 9).reshape(1, 8)


def _spec(dtype: str, dims: tuple[str, ...], data: np.ndarray, **extra) -> dict:
    return {"dtype": dtype, "dims": dims, "data": data, **extra}


def _layout(
    points: np.ndarray,
    cells: np.ndarray,
    *,
    cell_type: str = "HEX8",
    sidesets: tuple[tuple[str, np.ndarray, np.ndarray], ...] = (),
) -> tuple[dict[str, int], dict[str, dict]]:
    """Dimensions et variables d'un fichier Exodus minimal lisible par ``read_exodus``."""
    n_nodes, n_elems = len(points), len(cells)
    dims = {
        "num_dim": 3,
        "num_nodes": n_nodes,
        "num_elem": n_elems,
        "num_el_blk": 1,
        "num_el_in_blk1": n_elems,
        "num_nod_per_el1": cells.shape[1],
        "num_side_sets": len(sidesets),
        "len_name": LEN_NAME,
    }
    variables = {
        "coordx": _spec("f8", ("num_nodes",), points[:, 0]),
        "coordy": _spec("f8", ("num_nodes",), points[:, 1]),
        "coordz": _spec("f8", ("num_nodes",), points[:, 2]),
        "connect1": _spec(
            "i4", ("num_el_in_blk1", "num_nod_per_el1"), cells.astype("i4"), attrs={"elem_type": cell_type}
        ),
    }
    if sidesets:
        chars = np.zeros((len(sidesets), LEN_NAME), dtype="S1")
        for i, (name, _, _) in enumerate(sidesets):
            chars[i, : len(name)] = np.frombuffer(name.encode("ascii"), dtype="S1")
        variables["ss_names"] = _spec("S1", ("num_side_sets", "len_name"), chars)
        for s, (_, elems, sides) in enumerate(sidesets, start=1):
            dims[f"num_side_ss{s}"] = len(elems)
            variables[f"elem_ss{s}"] = _spec("i4", (f"num_side_ss{s}",), np.asarray(elems, dtype="i4"))
            variables[f"side_ss{s}"] = _spec("i4", (f"num_side_ss{s}",), np.asarray(sides, dtype="i4"))
    return dims, variables


def _write(path: Path, dims: dict[str, int], variables: dict[str, dict]) -> None:
    with netCDF4.Dataset(str(path), "w") as ds:
        for name, size in dims.items():
            ds.createDimension(name, size)
        for name, spec in variables.items():
            kwargs = {"fill_value": spec["fill"]} if spec.get("fill") is not None else {}
            var = ds.createVariable(name, spec["dtype"], spec["dims"], **kwargs)
            var[:] = spec["data"]
            for key, value in spec.get("attrs", {}).items():
                var.setncattr(key, value)


def _valid_file(path: Path) -> None:
    dims, variables = _layout(CUBE_POINTS, CUBE_CELLS, sidesets=(("bottom", np.array([1]), np.array([5])),))
    _write(path, dims, variables)


def test_valid_minimal_file_is_readable(tmp_path: Path) -> None:
    path = tmp_path / "cube.exo"
    _valid_file(path)
    exodus = read_exodus(path)
    assert exodus.n_nodes == 8
    assert exodus.n_elems == 1
    assert exodus.cell_type == "HEX8"
    assert set(exodus.sidesets) == {"bottom"}
    assert np.array_equal(exodus.points, CUBE_POINTS)
    assert np.array_equal(exodus.cells, CUBE_CELLS - 1)


def test_reference_mesh_still_readable() -> None:
    exodus = read_exodus(REFERENCE_MESH)
    assert exodus.cell_type == "HEX20"
    assert exodus.n_elems == 2592
    assert exodus.n_nodes == 12221
    assert set(exodus.sidesets) == {"symmetry_x", "symmetry_y", "symmetry_z", "compression"}
    _check_mesh(exodus)  # ne doit pas lever


def test_empty_sideset_is_still_accepted(tmp_path: Path) -> None:
    # Un rang sans facette produit un sideset de longueur nulle : le lecteur l'accepte.
    dims, variables = _layout(
        CUBE_POINTS, CUBE_CELLS, sidesets=(("empty", np.array([], dtype=int), np.array([], dtype=int)),)
    )
    path = tmp_path / "empty_sideset.exo"
    _write(path, dims, variables)
    exodus = read_exodus(path)
    elems, sides = exodus.sidesets["empty"]
    assert elems.size == 0 and sides.size == 0


def test_mesh_without_node_rejected(tmp_path: Path) -> None:
    dims, variables = _layout(np.empty((0, 3)), CUBE_CELLS)
    path = tmp_path / "no_node.exo"
    _write(path, dims, variables)
    with pytest.raises(ValueError, match="sans nœud"):
        read_exodus(path)


def test_mesh_without_element_rejected(tmp_path: Path) -> None:
    dims, variables = _layout(CUBE_POINTS, np.empty((0, 8), dtype=int))
    path = tmp_path / "no_elem.exo"
    _write(path, dims, variables)
    with pytest.raises(ValueError, match="sans élément"):
        read_exodus(path)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_coordinates_rejected(tmp_path: Path, bad: float) -> None:
    points = CUBE_POINTS.copy()
    points[3, 1] = bad
    dims, variables = _layout(points, CUBE_CELLS)
    path = tmp_path / "non_finite.exo"
    _write(path, dims, variables)
    with pytest.raises(ValueError, match="coordonnées non finies"):
        read_exodus(path)


def test_masked_coordinates_rejected(tmp_path: Path) -> None:
    dims, variables = _layout(CUBE_POINTS, CUBE_CELLS)
    masked = CUBE_POINTS[:, 0].copy()
    masked[0] = -9999.0
    variables["coordx"] = _spec("f8", ("num_nodes",), masked, fill=-9999.0)
    path = tmp_path / "masked_coord.exo"
    _write(path, dims, variables)
    with pytest.raises(ValueError, match="masquée"):
        read_exodus(path)


def test_masked_connectivity_rejected(tmp_path: Path) -> None:
    dims, variables = _layout(CUBE_POINTS, CUBE_CELLS)
    masked = CUBE_CELLS.astype("i4").copy()
    masked[0, 0] = -1
    variables["connect1"] = _spec(
        "i4",
        ("num_el_in_blk1", "num_nod_per_el1"),
        masked,
        attrs={"elem_type": "HEX8"},
        fill=-1,
    )
    path = tmp_path / "masked_connect.exo"
    _write(path, dims, variables)
    with pytest.raises(ValueError, match="masquée"):
        read_exodus(path)


def test_masked_sideset_rejected(tmp_path: Path) -> None:
    dims, variables = _layout(CUBE_POINTS, CUBE_CELLS, sidesets=(("bottom", np.array([1]), np.array([5])),))
    variables["elem_ss1"] = _spec("i4", ("num_side_ss1",), np.array([-1], dtype="i4"), fill=-1)
    path = tmp_path / "masked_sideset.exo"
    _write(path, dims, variables)
    with pytest.raises(ValueError, match="masquée"):
        read_exodus(path)


def test_inconsistent_coordinate_shape_rejected(tmp_path: Path) -> None:
    dims, variables = _layout(CUBE_POINTS, CUBE_CELLS)
    variables["coordx"] = _spec("f8", ("num_dim",), np.zeros(3))  # 3 valeurs pour num_nodes = 8
    path = tmp_path / "bad_coord_shape.exo"
    _write(path, dims, variables)
    with pytest.raises(ValueError, match="dimensions incohérentes"):
        read_exodus(path)


def test_inconsistent_connectivity_width_rejected(tmp_path: Path) -> None:
    # HEX8 annoncé, mais 4 nœuds par élément dans connect1.
    dims, variables = _layout(CUBE_POINTS, CUBE_CELLS[:, :4])
    path = tmp_path / "bad_connect_width.exo"
    _write(path, dims, variables)
    with pytest.raises(ValueError, match="nœuds par élément"):
        read_exodus(path)


def test_inconsistent_sideset_lengths_rejected(tmp_path: Path) -> None:
    dims, variables = _layout(
        CUBE_POINTS, CUBE_CELLS, sidesets=(("bottom", np.array([1, 1]), np.array([5, 6])),)
    )
    dims["one"] = 1
    variables["side_ss1"] = _spec("i4", ("one",), np.array([5], dtype="i4"))
    path = tmp_path / "bad_sideset_len.exo"
    _write(path, dims, variables)
    with pytest.raises(ValueError, match="longueurs différentes"):
        read_exodus(path)


@pytest.mark.filterwarnings("ignore:invalid value encountered in det:RuntimeWarning")
def test_nan_jacobian_rejected() -> None:
    points = CUBE_POINTS.copy()
    points[4, 0] = np.nan
    exodus = ExodusMesh(
        points=points,
        cells=CUBE_CELLS - 1,
        cell_type="HEX8",
        sidesets={},
    )
    with pytest.raises(RuntimeError, match="jacobien"):
        _check_mesh(exodus)


def test_empty_mesh_rejected_by_check_mesh() -> None:
    exodus = ExodusMesh(
        points=np.empty((0, 3)),
        cells=np.empty((0, 8), dtype=np.int64),
        cell_type="HEX8",
        sidesets={},
    )
    with pytest.raises(RuntimeError, match="Maillage vide"):
        _check_mesh(exodus)
