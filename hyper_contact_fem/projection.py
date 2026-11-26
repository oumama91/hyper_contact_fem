"""Projection nodale P1 (L2 à masse concentrée) et espaces d'export autonomes communs à tous les champs.

La projection à masse concentrée est implémentée en P1 : sur les éléments HEX8/HEX20 supportés, le
dénominateur concentré ∫φ_i dx y est strictement positif. Ce choix n'est pas généralisé aux autres
ordres ni aux autres familles d'éléments. Conséquence assumée : les maxima tirés de cette projection
(sigma_1_max) sont des maxima lissés P1, même sur un maillage P2.
"""
from __future__ import annotations

import basix.ufl
import numpy as np
import ufl
from dolfinx import fem
from dolfinx.mesh import Mesh
from mpi4py import MPI

from hyper_contact_fem.utils import Expr, assemble_vec


def lagrange_space(domain: Mesh, degree: int, shape: tuple[int, ...] = ()) -> fem.FunctionSpace:
    """Espace de Lagrange autonome (scalaire si shape == (), sinon bloqué de forme 'shape')."""
    element = basix.ufl.element("Lagrange", domain.topology.cell_name(), degree, shape=shape or None)
    return fem.functionspace(domain, element)


def check_same_numbering(comm: MPI.Comm, ref: fem.FunctionSpace, *others: fem.FunctionSpace) -> None:
    """Vérifie que les espaces ont la même numérotation de nœuds (condition d'un export VTX cohérent)."""
    def same(a: fem.FunctionSpace, b: fem.FunctionSpace) -> bool:
        return (
            a.dofmap.index_map.size_local == b.dofmap.index_map.size_local
            and a.dofmap.index_map.num_ghosts == b.dofmap.index_map.num_ghosts
            and np.array_equal(a.dofmap.list, b.dofmap.list)
        )

    ok = all(same(ref, other) for other in others)
    if not all(comm.allgather(ok)):
        raise RuntimeError(
            "Export VTX : les espaces d'export n'ont pas la même numérotation de dofs ; "
            "les champs seraient écrits sur de mauvais nœuds."
        )


def project_p1(domain: Mesh, dx: ufl.Measure, exprs: list[Expr]) -> fem.Function:
    """
    Projection L2 à masse concentrée de plusieurs champs scalaires vers P1, en une seule assemblée.
    Retourne une Function P1 à len(exprs) composantes (une colonne par champ).
    'dx' porte le degré de quadrature choisi dans le YAML.

    Le dénominateur (masse concentrée ∫ φ_i dx) est le même pour toutes les composantes : il est assemblé
    une seule fois dans un espace P1 scalaire, au lieu d'être répété n fois dans l'espace bloqué.
    """
    n = len(exprs)
    W = lagrange_space(domain, 1, (n,))
    S = lagrange_space(domain, 1)
    check_same_numbering(domain.comm, S, W)  # même numérotation de nœuds : colonnes <-> composantes

    num = assemble_vec(fem.form(ufl.inner(ufl.as_vector(exprs), ufl.TestFunction(W)) * dx))
    den = assemble_vec(fem.form(ufl.TestFunction(S) * dx))

    n_nodes = S.dofmap.index_map.size_local
    f = fem.Function(W)
    nodal = num.array[: n_nodes * n].reshape(-1, n) / den.array[:n_nodes, None]
    f.x.array[: n_nodes * n] = nodal.reshape(-1)
    f.x.scatter_forward()
    num.destroy()
    den.destroy()
    return f


def export_spaces(comm: MPI.Comm, domain: Mesh, order: int, gdim: int, n_cols: int) -> dict[int, fem.FunctionSpace]:
    """
    Espaces d'export autonomes, communs à tous les champs : {1: scalaire, 3: vecteur, 9: tenseur 3x3},
    plus l'espace à n_cols composantes (clé 0) qui reçoit la projection nodale interpolée.
    """
    spaces = {
        1: lagrange_space(domain, order),
        3: lagrange_space(domain, order, (gdim,)),
        9: lagrange_space(domain, order, (3, 3)),
        0: lagrange_space(domain, order, (n_cols,)),
    }
    check_same_numbering(comm, spaces[3], spaces[1], spaces[9], spaces[0])
    return spaces


def nodal_columns(nodal: fem.Function, space: fem.FunctionSpace, names: list[str]) -> dict[str, np.ndarray]:
    """Interpole la projection P1 à l'ordre du maillage ; retourne {nom de colonne: valeurs nodales}."""
    nodal_k = fem.Function(space)
    nodal_k.interpolate(nodal)
    nodal_k.x.scatter_forward()
    cols = nodal_k.x.array.reshape(-1, len(names))
    return {name: cols[:, k] for k, name in enumerate(names)}
