"""
Lecteur Exodus et permutations de numérotation : contrôles sur HEX8/HEX20 et sur le maillage
de l'étude de référence (`mesh/O-ring.exo`).

Les permutations attendues sont celles documentées dans exodus_mesh.py, vérifiées ici
explicitement : une inversion silencieuse produirait un maillage faux sans lever d'erreur.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hyper_contact_fem.exodus_mesh import EXODUS_ORDER, _check_mesh, _local_facet_of_side
from hyper_contact_fem.exodus_reader import read_exodus

MESH_FILE = Path(__file__).resolve().parents[1] / "mesh" / "O-ring.exo"

HEX8_EXPECTED = [0, 1, 3, 2, 4, 5, 7, 6]
HEX20_EXPECTED = [0, 1, 3, 2, 4, 5, 7, 6, 8, 11, 12, 9, 13, 10, 15, 14, 16, 19, 17, 18]


@pytest.mark.parametrize(("cell_type", "expected"), [("HEX8", HEX8_EXPECTED), ("HEX20", HEX20_EXPECTED)])
def test_exodus_order(cell_type: str, expected: list[int]) -> None:
    order = EXODUS_ORDER[cell_type]
    assert order.tolist() == expected
    assert np.array_equal(np.sort(order), np.arange(len(expected)))


@pytest.mark.parametrize("cell_type", ["HEX8", "HEX20"])
def test_side_to_facet_mapping_is_bijective(cell_type: str) -> None:
    mapping = _local_facet_of_side(cell_type, EXODUS_ORDER[cell_type])
    assert sorted(mapping) == [1, 2, 3, 4, 5, 6]
    assert sorted(mapping.values()) == [0, 1, 2, 3, 4, 5]


def test_reference_mesh_is_readable_and_valid() -> None:
    exodus = read_exodus(MESH_FILE)
    assert exodus.cell_type == "HEX20"
    assert exodus.n_elems == 2592
    assert exodus.n_nodes == 12221
    assert set(exodus.sidesets) == {"symmetry_x", "symmetry_y", "symmetry_z", "compression"}
    for elems, sides in exodus.sidesets.values():
        assert elems.size > 0
        assert sides.min() >= 1 and sides.max() <= 6
    _check_mesh(exodus)  # jacobiens > 0, maillage conforme : ne doit pas lever


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        read_exodus(MESH_FILE.with_name("absent.exo"))
