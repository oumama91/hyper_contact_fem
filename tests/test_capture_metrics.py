"""
Capture du déplacement réellement exporté (F-009).

``postproc.write_vtx`` interpole ``u_h`` dans ``u_out`` puis écrit ``u_out``. Ces tests
vérifient, avec un double de ``write_vtx``, que la capture lit bien la sortie (u_out) et non
l'entrée (u_h). Aucun solveur ni calcul mécanique n'est lancé.
"""
from __future__ import annotations

import basix.ufl
import numpy as np
import pytest
import ufl
from dolfinx import fem
from dolfinx import mesh as dm
from mpi4py import MPI

import hyper_contact_fem.postproc as pp
from tools import capture_metrics as cm

_POINTS = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
_CELLS = np.array([[0, 1, 2, 3]], dtype=np.int64)
_DOMAIN = ufl.Mesh(basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,)))
_MESH = dm.create_mesh(MPI.COMM_SELF, _CELLS, _DOMAIN, _POINTS)


@pytest.fixture(autouse=True)
def _restore_postproc():
    """Annule le monkeypatch de write_vtx et le point d'observation après chaque test."""
    write_vtx = pp.write_vtx
    observer = pp.displacement_observer
    yield
    pp.write_vtx = write_vtx
    pp.displacement_observer = observer


def _displacement(values: list[float]) -> fem.Function:
    V = fem.functionspace(_MESH, ("Lagrange", 1, (3,)))
    u = fem.Function(V)
    u.x.array[:] = 0.0
    u.x.array[: len(values)] = values
    return u


def test_capture_uses_exported_displacement_not_input(tmp_path) -> None:
    u_h = _displacement([1.0, 2.0, 3.0])       # entrée de write_vtx
    u_out = _displacement([9.0, -9.0, 9.0])    # ce que write_vtx écrit réellement
    capture = cm._FieldCapture()
    seen: dict[str, fem.Function] = {}

    def fake_write_vtx(comm, config, spaces, u_h_arg, requested, values):
        seen["u_h"] = u_h_arg
        pp.displacement_observer(u_out)  # même appel que write_vtx après interpolation
        return str(tmp_path / "fake.bp"), ["displacement"]

    pp.write_vtx = fake_write_vtx
    cm._install_field_capture(capture)
    pp.write_vtx(MPI.COMM_SELF, None, {3: u_out.function_space}, u_h, ("displacement",), {})

    assert seen["u_h"] is u_h
    stats = cm._exported_field_statistics(MPI.COMM_SELF, capture)["displacement"]
    # Les statistiques sont celles de u_out (max 9), pas celles de u_h (max 3).
    assert stats["max"] == 9.0
    assert stats["min"] == -9.0
    assert stats["l2"] == pytest.approx(float(np.sqrt(3.0 * 81.0)))
    assert stats["components"] == 3


def test_statistics_without_capture_raise() -> None:
    capture = cm._FieldCapture()
    with pytest.raises(RuntimeError, match="write_vtx n'a pas été appelé"):
        cm._exported_field_statistics(MPI.COMM_SELF, capture)
