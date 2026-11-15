"""
Tests du maximum distribué de ``postproc._owned_max`` et de la norme de déplacement.

Aucun calcul mécanique : un unique tétraèdre fournit un espace P1 vectoriel dont on fixe
directement les valeurs nodales. Le cas multi-rangs est simulé par un communicateur factice
qui applique la même réduction MAX que MPI, sans collectif réel.
"""
from __future__ import annotations

import basix.ufl
import numpy as np
import ufl
from dolfinx import fem
from dolfinx import mesh as dm
from mpi4py import MPI

from postproc import _max_displacement, _owned_max

_POINTS = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
_CELLS = np.array([[0, 1, 2, 3]], dtype=np.int64)
_MESH_DOMAIN = ufl.Mesh(basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,)))
_MESH = dm.create_mesh(MPI.COMM_SELF, _CELLS, _MESH_DOMAIN, _POINTS)


class _FakeComm:
    """Communicateur minimal : réduit comme MPI.MAX sur les valeurs des autres rangs."""

    def __init__(self, others: list[float]):
        self.others = others

    def allreduce(self, value: float, op) -> float:
        assert op is MPI.MAX
        return max([value, *self.others])


def test_owned_max_all_negative() -> None:
    assert _owned_max(MPI.COMM_SELF, np.array([-5.0, -2.0])) == -2.0


def test_owned_max_empty_returns_neutral() -> None:
    assert np.isneginf(_owned_max(MPI.COMM_SELF, np.array([])))


def test_owned_max_positive_unchanged() -> None:
    assert _owned_max(MPI.COMM_SELF, np.array([1.0, 3.0, 2.0])) == 3.0


def test_owned_max_empty_rank_does_not_mask_negative_rank() -> None:
    # Rang local vide (-inf) et autre rang à -5 : le maximum global doit rester -5.
    assert _owned_max(_FakeComm([-5.0]), np.array([])) == -5.0


def test_owned_max_two_negative_ranks() -> None:
    assert _owned_max(_FakeComm([-5.0]), np.array([-2.0])) == -2.0


def _displacement(values: list[float]) -> fem.Function:
    V = fem.functionspace(_MESH, ("Lagrange", 1, (3,)))
    u = fem.Function(V)
    u.x.array[:] = 0.0
    u.x.array[: len(values)] = values
    return u


def test_max_displacement_zero() -> None:
    assert _max_displacement(MPI.COMM_SELF, _displacement([]), 3) == 0.0


def test_max_displacement_nonzero() -> None:
    assert _max_displacement(MPI.COMM_SELF, _displacement([3.0, 4.0, 0.0]), 3) == 5.0
