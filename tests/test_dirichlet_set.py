"""
Tests unitaires de ``DirichletSet`` (dirichlet.py) : contrat ``names`` / ``owner`` / ``comp_of``.

Le golden n'exerce qu'un cas simple (trois symétries sur des composantes distinctes) et ne
couvre ni les frontières sécantes, ni les dofs partagés, ni les dofs fantômes. Ces tests
construisent des ``LocatedBC`` synthétiques : le constructeur ne dépend que de ``entries`` et de
``n_flat`` (``component_spaces`` et ``owned`` sont seulement mémorisés pour ``__call__`` et les
réactions), donc aucun maillage ni solveur n'est nécessaire.
"""
from __future__ import annotations

import numpy as np

from dirichlet import DirichletSet, LocatedBC


def _bc(name: str, comp: int, dofs: list[int]) -> LocatedBC:
    return LocatedBC(name=name, comp=comp, dofs=np.asarray(dofs, dtype=np.int64), value=0.0)


def _set(entries: list[LocatedBC], n_flat: int) -> DirichletSet:
    """DirichletSet sans espace ni dofs possédés : seuls ``entries`` et ``n_flat`` sont utilisés."""
    return DirichletSet(entries, [], [], n_flat)


def test_names_follow_first_appearance_without_duplicates() -> None:
    # Une même frontière sur deux composantes ne doit produire qu'un seul nom.
    entries = [_bc("left", 0, [0, 1]), _bc("left", 1, [2]), _bc("right", 0, [3])]
    ds = _set(entries, 4)
    assert ds.names == ("left", "right")


def test_shared_dof_belongs_to_last_declared_boundary() -> None:
    # Deux frontières sécantes sur la même composante : le dof 2 est contraint deux fois.
    entries = [_bc("first", 0, [0, 1, 2]), _bc("second", 0, [2, 3])]
    ds = _set(entries, 4)
    assert ds.names == ("first", "second")
    assert ds.owner.tolist() == [0, 0, 1, 1]
    assert ds.comp_of.tolist() == [0, 0, 0, 0]


def test_same_boundary_on_different_components() -> None:
    entries = [_bc("corner", 0, [0, 1]), _bc("corner", 1, [2, 3])]
    ds = _set(entries, 4)
    assert ds.names == ("corner",)
    assert ds.owner.tolist() == [0, 0, 0, 0]
    assert ds.comp_of.tolist() == [0, 0, 1, 1]


def test_unconstrained_dofs_have_owner_minus_one() -> None:
    ds = _set([_bc("left", 0, [1, 4])], 6)
    assert ds.owner.tolist() == [-1, 0, -1, -1, 0, -1]


def test_ghost_dofs_are_excluded_without_error() -> None:
    # Indices >= n_flat : dofs fantômes de la numérotation de V, à ignorer silencieusement.
    ds = _set([_bc("left", 0, [1, 7, 9])], 4)
    assert ds.owner.shape == (4,)
    assert ds.owner.tolist() == [-1, 0, -1, -1]


def test_empty_boundary_keeps_its_name() -> None:
    # Cas d'un rang qui ne possède aucune facette du sideset : tableau de dofs vide.
    ds = _set([_bc("empty", 0, []), _bc("right", 2, [1])], 3)
    assert ds.names == ("empty", "right")
    assert ds.owner.tolist() == [-1, 1, -1]
    assert ds.comp_of[1] == 2
