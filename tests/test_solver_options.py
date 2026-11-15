"""
Restauration des options PETSc globales par ``LinearSolver`` (F-008) et garde du KSP.

Aucun système n'est résolu : un KSP se crée sans maillage (``MPI.COMM_SELF``), seules les
options ``fea_*`` globales et leur cycle de vie sont vérifiés.
"""
from __future__ import annotations

import pytest
from mpi4py import MPI
from petsc4py import PETSc

from solver import LinearSolver

KEYS = ("fea_ksp_type", "fea_pc_type", "fea_pc_factor_mat_solver_type")


@pytest.fixture(autouse=True)
def _clean_options():
    """Aucun test ne doit laisser d'option globale PETSc derrière lui."""
    options = PETSc.Options()
    for key in KEYS:
        options.delValue(key)
    yield
    for key in KEYS:
        options.delValue(key)


def _solver() -> LinearSolver:
    # V n'est pas utilisé par __enter__/close : aucun maillage ni espace n'est nécessaire.
    return LinearSolver(MPI.COMM_SELF, None)


def test_absent_keys_are_set_inside_and_removed_after() -> None:
    options = PETSc.Options()
    for key in KEYS:
        assert key not in options
    with _solver():
        assert options["fea_ksp_type"] == "preonly"
        assert options["fea_pc_type"] == "lu"
        assert options["fea_pc_factor_mat_solver_type"] == "mumps"
    for key in KEYS:
        assert key not in options


def test_preexisting_value_is_restored_after_normal_exit() -> None:
    options = PETSc.Options()
    options["fea_ksp_type"] = "gmres"
    with _solver():
        assert options["fea_ksp_type"] == "preonly"
    assert options["fea_ksp_type"] == "gmres"
    assert "fea_pc_type" not in options


def test_preexisting_value_is_restored_after_exception() -> None:
    options = PETSc.Options()
    options["fea_pc_type"] = "jacobi"
    with pytest.raises(ValueError, match="boum"):
        with _solver():
            assert options["fea_pc_type"] == "lu"
            raise ValueError("boum")
    assert options["fea_pc_type"] == "jacobi"
    assert "fea_ksp_type" not in options


def test_nested_solvers_restore_outer_then_initial_state() -> None:
    options = PETSc.Options()
    options["fea_ksp_type"] = "gmres"
    with _solver():
        assert options["fea_ksp_type"] == "preonly"
        with _solver():
            assert options["fea_ksp_type"] == "preonly"
        # La sortie du contexte intérieur rétablit les valeurs du contexte extérieur.
        assert options["fea_ksp_type"] == "preonly"
        assert options["fea_pc_type"] == "lu"
    assert options["fea_ksp_type"] == "gmres"
    assert "fea_pc_type" not in options


def test_solve_without_initialised_ksp_raises() -> None:
    lin = _solver()
    with pytest.raises(RuntimeError, match="KSP non initialisé"):
        lin.solve(None, None, None)
