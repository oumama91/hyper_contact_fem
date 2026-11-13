from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from dolfinx import fem
from dolfinx.mesh import MeshTags
from mpi4py import MPI

from config import Config, DirichletBoundary
from forms import Forms
from utils import SCALAR, owned_size


@dataclass(frozen=True)
class LocatedBC:
    """Une condition de Dirichlet localisée : composante 'comp' imposée à 'value' sur les dofs 'dofs'."""
    name: str
    comp: int
    dofs: np.ndarray  # numérotation de V (mixte ou non)
    value: float      # valeur de référence (valeur du YAML ; multipliée par lambda à l'application)


def locate_dirichlet(
    forms: Forms, facet_tags: MeshTags, boundary_tags: dict[str, int], config: Config
) -> list[LocatedBC]:
    """Localise UNE fois les dofs de chaque (boundary de Dirichlet, composante)."""
    entries = []
    for bnd in config.boundaries:
        if not isinstance(bnd, DirichletBoundary):
            continue
        facets = facet_tags.find(boundary_tags[bnd.name])
        for comp, val in zip(bnd.components, bnd.value, strict=True):
            entries.append(LocatedBC(bnd.name, comp, forms.component_dofs(comp, facets), val))
    return entries


def check_bc_conflicts(comm: MPI.Comm, entries: list[LocatedBC]) -> None:
    """Avertit si deux Dirichlet imposent des valeurs différentes sur un même dof."""
    pairs, counts = [], []
    for i, a in enumerate(entries):
        for b in entries[i + 1:]:
            if a.comp == b.comp and not np.isclose(a.value, b.value):
                pairs.append((a.name, b.name, a.comp))
                counts.append(np.intersect1d(a.dofs, b.dofs).size)

    totals = comm.allreduce(np.array(counts, dtype=int), op=MPI.SUM)
    if comm.rank == 0:
        for (n1, n2, comp), n in zip(pairs, totals, strict=True):
            if n > 0:
                warnings.warn(
                    f"Dirichlet en conflit sur la composante {comp} : '{n1}' et '{n2}' "
                    f"partagent ~{n} dofs avec des valeurs différentes (la dernière déclarée l'emporte).",
                    stacklevel=2,
                )


class DirichletSet:
    """
    Conditions de Dirichlet localisées une fois.
    Callable : dirichlet(scale) -> liste de DirichletBC dont les valeurs sont multipliées par 'scale'
    (facteur de charge lambda), donc utilisable directement comme 'bcs_at' par solve_newton.

    'entries' et 'owned' servent au calcul des réactions (dofs dans la numérotation de V, mixte ou non).
    'names', 'owner' et 'comp_of' sont précalculés une seule fois ICI (et non à chaque incrément dans
    reactions.py) : 'owner[dof]' est l'index dans 'names' de la boundary qui possède 'dof'
    (-1 si non contraint), 'comp_of[dof]' sa composante. Même règle que l'application des BC :
    la dernière boundary déclarée qui contraint un dof donné l'emporte (pas de double comptage).
    """

    def __init__(
        self,
        entries: list[LocatedBC],
        component_spaces: list[fem.FunctionSpace],
        owned: list[np.ndarray],
        n_flat: int,
    ):
        self.entries = entries
        self._spaces = component_spaces
        self.owned = owned

        self.names: tuple[str, ...] = tuple(dict.fromkeys(bc.name for bc in entries))
        self.owner = np.full(n_flat, -1, dtype=np.int64)
        self.comp_of = np.zeros(n_flat, dtype=np.int64)
        for bc in entries:
            own = np.asarray(bc.dofs[bc.dofs < n_flat], dtype=np.int64)  # exclut les dofs fantômes
            self.owner[own] = self.names.index(bc.name)
            self.comp_of[own] = bc.comp

    def __call__(self, scale: float) -> list[fem.DirichletBC]:
        return [fem.dirichletbc(SCALAR(bc.value * scale), bc.dofs, self._spaces[bc.comp]) for bc in self.entries]


def apply_bcs(
    forms: Forms, facet_tags: MeshTags, boundary_tags: dict[str, int], config: Config
) -> DirichletSet:
    """Localise une fois les dofs de Dirichlet, signale les conflits, et retourne un DirichletSet."""
    entries = locate_dirichlet(forms, facet_tags, boundary_tags, config)
    check_bc_conflicts(forms.V.mesh.comm, entries)

    gdim = forms.V.mesh.geometry.dim
    spaces = [forms.component_space(c) for c in range(gdim)]
    owned = [forms.owned_component_dofs(c) for c in range(gdim)]
    n_flat = owned_size(forms.V)
    return DirichletSet(entries, spaces, owned, n_flat)