"""
Caractérisation de ``exodus_reader.read_exodus`` : contenu, dtypes et ordre des tableaux.

Le maillage de référence est figé par un hash sha256 de chaque tableau binaire ; les petits
fichiers synthétiques HEX8 et HEX20 sont fabriqués avec les mêmes helpers que
``tests/test_exodus_invalid.py`` et leurs valeurs sont vérifiées à la main. Les rejets sont déjà
couverts par ce dernier fichier et ne sont pas dupliqués ici.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from exodus_reader import ExodusMesh, read_exodus
from test_exodus_invalid import CUBE_CELLS, CUBE_POINTS, _layout, _write

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_MESH = ROOT / "mesh" / "O-ring.exo"

# Même dépréciation netCDF4/NumPy 2.5 que dans test_exodus_invalid.py, sans effet sur la lecture.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated:DeprecationWarning"
)

# Cube unité HEX20 : les huit coins puis les douze milieux d'arêtes, dans l'ordre Exodus.
HEX20_POINTS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.5, 0.0, 0.0],
        [1.0, 0.5, 0.0],
        [0.5, 1.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
        [1.0, 0.0, 0.5],
        [1.0, 1.0, 0.5],
        [0.0, 1.0, 0.5],
        [0.5, 0.0, 1.0],
        [1.0, 0.5, 1.0],
        [0.5, 1.0, 1.0],
        [0.0, 0.5, 1.0],
    ],
    dtype=np.float64,
)
HEX20_CELLS = np.arange(1, 21, dtype=np.int64).reshape(1, 20)


def _sha256(array: np.ndarray) -> str:
    """Empreinte du contenu binaire, indépendante de l'ordre de lecture des variables netCDF."""
    return hashlib.sha256(array.tobytes()).hexdigest()


def _write_synthetic_cube(path: Path, points: np.ndarray, cells: np.ndarray, cell_type: str) -> None:
    """Cube minimal avec un sideset ``bottom`` (élément 1, face 5), comme les tests existants."""
    sidesets = (("bottom", np.array([1]), np.array([5])),)
    dims, variables = _layout(points, cells, cell_type=cell_type, sidesets=sidesets)
    _write(path, dims, variables)


def test_reference_mesh_content_is_frozen() -> None:
    exodus = read_exodus(REFERENCE_MESH)

    assert exodus.cell_type == "HEX20"
    assert exodus.n_elems == 2592
    assert exodus.n_nodes == 12221
    assert list(exodus.sidesets) == ["symmetry_y", "symmetry_x", "symmetry_z", "compression"]

    assert exodus.points.shape == (12221, 3)
    assert exodus.points.dtype == np.float64
    assert _sha256(exodus.points) == "565410b1fa898e1ee35f29b2a2474aacaf926086e79840be363e5a50ba382257"

    assert exodus.cells.shape == (2592, 20)
    assert exodus.cells.dtype == np.int64
    assert _sha256(exodus.cells) == "ba04763c82b9d493ca7effa96ad3ef9e31648a468e1ae3c6489d0903024711e3"

    expected_sidesets = {
        "symmetry_y": ((108,), "41c24bacc1b1fc695a396e01893a0eaa16e716138eb797de2cb4d9912ad9f158",
                       (108,), "f78a8a7df584e21ca10dca4755b971bd7ad6b5b46fd40e0c10b2ac0241f2e82b"),
        "symmetry_x": ((108,), "399f44abf7876810648cb5e4d165536b34cb02676279522061bb9c89cc5b5ff7",
                       (108,), "705c6e1b907b6242f624ae2df4d1b158ed470278957b0019ccaf24f99410b655"),
        "symmetry_z": ((192,), "a4c99f9685c5de24fe448d956e55c1bd9b512ad039aa926d09f7381ee680f9c0",
                       (192,), "e06c90bb5ae2c19b688516619bc9d3d6af06da1bc36a531318714549c72ed87b"),
        "compression": ((768,), "57a3618fe3f2d7268835a91864c4fefdd45f5084c56e577207cf50d53af6b11a",
                        (768,), "b31cd09ece3f37b1dec1258eae49788fa456200d1b158c5bb89e4f4bb13d0ce7"),
    }
    for name, (elems_shape, elems_hash, sides_shape, sides_hash) in expected_sidesets.items():
        elems, sides = exodus.sidesets[name]
        assert elems.shape == elems_shape
        assert elems.dtype == np.int64
        assert _sha256(elems) == elems_hash
        assert sides.shape == sides_shape
        assert sides.dtype == np.int64
        assert _sha256(sides) == sides_hash


@pytest.mark.parametrize(
    ("cell_type", "points", "cells", "points_hash", "cells_hash"),
    [
        (
            "HEX8",
            CUBE_POINTS,
            CUBE_CELLS,
            "07ad26aededecfdac641feac2145fafd78ea0ac32e4dde4018d954878040a360",
            "fece8d601cd4c9020e24f9e4a47feedefb2bceff5e9798d8056aea8700052eaa",
        ),
        (
            "HEX20",
            HEX20_POINTS,
            HEX20_CELLS,
            "8e850cfe0274c632926b4308f4cc5fd0e2f96cee28195d98f45a9a434cde423e",
            "f5c4cb24f4c9b43e624e0a83cb11934f932dbd5db24eee67fed696354ac88a61",
        ),
    ],
)
def test_synthetic_cube_content_is_read_verbatim(
    tmp_path: Path,
    cell_type: str,
    points: np.ndarray,
    cells: np.ndarray,
    points_hash: str,
    cells_hash: str,
) -> None:
    path = tmp_path / f"cube_{cell_type}.exo"
    _write_synthetic_cube(path, points, cells, cell_type)
    exodus = read_exodus(path)

    assert exodus.cell_type == cell_type
    assert exodus.n_elems == 1
    assert exodus.n_nodes == len(points)
    assert list(exodus.sidesets) == ["bottom"]

    assert exodus.points.shape == points.shape
    assert exodus.points.dtype == np.float64
    assert _sha256(exodus.points) == points_hash
    assert np.array_equal(exodus.points, points)

    expected_cells = cells - 1
    assert exodus.cells.shape == expected_cells.shape
    assert exodus.cells.dtype == np.int64
    assert _sha256(exodus.cells) == cells_hash
    assert np.array_equal(exodus.cells, expected_cells)

    elems, sides = exodus.sidesets["bottom"]
    assert elems.shape == (1,) and sides.shape == (1,)
    assert elems.dtype == np.int64 and sides.dtype == np.int64
    assert elems.tolist() == [0]
    assert sides.tolist() == [5]
    assert _sha256(elems) == "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc"
    assert _sha256(sides) == "f13ee6ed54ea2aae9fc49a9faeb5da6e8ddef0e12ed5d30d35a624ae813e0485"
