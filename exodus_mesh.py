"""
Backend de maillage Exodus II (Cubit) : HEX8 / HEX20 -> maillage dolfinx + tags de facettes.

Contrat : build_mesh(comm, config) renvoie un MeshResult (maillage, tags de facettes, noms de
boundaries) ; l'aval (formes, conditions aux limites, solveur, post-traitement) ignore la
provenance du maillage.

Éléments supportés (v1) : HEX8 (géométrie linéaire) et HEX20 (géométrie quadratique : élément serendipity
de basix, 20 nœuds). Les sidesets Exodus deviennent les boundaries nommées du YAML ; leurs faces (1..6)
sont taguées sur les facettes du maillage dolfinx.

Numérotation locale des nœuds : la convention Cubit/Exodus n'est pas celle de basix. La permutation est
calculée une fois pour toutes, à l'import, de façon purement combinatoire (EXODUS_ORDER) : chaque ddl de
l'élément basix est porté par une entité de la cellule de référence (sommet ou arête) ; le point de
référence de cette entité (sommet, ou milieu d'arête) désigne sans ambiguïté le nœud Exodus occupant la
même position dans la cellule de référence Exodus. Aucune géométrie du maillage réel n'intervient : le
résultat est donc insensible à l'orientation et à la courbure des éléments (maillages tournés type
O-ring compris), et il est vérifié par tests/test_exodus.py sur un cube HEX8 et HEX20.

Parallélisme (dolfinx 0.11) :
  * create_mesh reçoit la connectivité d'un seul rang et la distribue ; dupliquer la connectivité sur
    tous les rangs provoque une corruption du tas (SIGABRT) dès que n > 1. Le fichier .exo n'est donc lu
    et validé que sur le rang 0 (points, connectivité, jacobien > 0, facette portée par au plus deux
    cellules) ; seuls les sidesets — petits — sont diffusés aux autres rangs, qui ne conservent ni les
    coordonnées ni la connectivité du fichier.
  * toute erreur détectée par un seul rang est relayée à tous avant d'être levée (allgather, même motif
    que check_same_numbering dans projection.py) : une levée locale laisserait les autres rangs
    bloqués dans le collectif suivant (deadlock MPI).

Pièges traités explicitement (une erreur claire vaut mieux qu'un maillage silencieusement faux) :
  * un sideset qui référence une face interne est refusé (un sideset doit être un bord) ;
  * deux sidesets ne peuvent pas partager une facette : le tag serait ambigu ;
  * contrôle optionnel par points sonde (probe_points dans le YAML) : chaque point doit se trouver à
    moins de probe_tol de la face effectivement taguée par son sideset.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import basix
import basix.ufl
import numpy as np
import ufl
from dolfinx import geometry
from dolfinx import mesh as dm
from dolfinx.mesh import Mesh, MeshTags, entities_to_geometry
from mpi4py import MPI

from config import Config
from exodus_reader import ExodusMesh, read_exodus
from utils import integrate

logger = logging.getLogger(__name__)


@dataclass
class MeshResult:
    """
    Résultat du backend de maillage Exodus, consommé par tout l'aval.

    domain / facet_tags / boundary_tags : ce que consomment formes, conditions aux limites,
    solveur et post-traitement. volume : volume du domaine en configuration de référence
    [mm³], mesuré sur le maillage. source / geom_degree / description : traçabilité (résumé
    de l'étude).
    """
    domain: Mesh
    facet_tags: MeshTags
    boundary_tags: dict[str, int]
    volume: float
    source: str
    geom_degree: int
    description: str


CELL_FAMILIES = {"HEX8": ("Lagrange", 1), "HEX20": ("serendipity", 2)}
BASIX_FAMILIES = {"Lagrange": basix.ElementFamily.P, "serendipity": basix.ElementFamily.serendipity}

# Faces locales d'un hexaèdre en numérotation Exodus (coins locaux 0-based) : side 1..6.
EXODUS_HEX_FACES = {1: (0, 1, 5, 4), 2: (1, 2, 6, 5), 3: (2, 3, 7, 6), 4: (3, 0, 4, 7), 5: (0, 3, 2, 1), 6: (4, 5, 6, 7)}

# Cellule de référence Exodus (convention Exodus II, SAND92-2137) : coins 1..8 (0-based ici), puis, pour
# HEX20, les milieux d'arête 9..20 dans l'ordre EXODUS_HEX_EDGES (arêtes de la face basse, arêtes
# verticales, arêtes de la face haute).
EXODUS_REFERENCE_CORNERS = np.array(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.float64
)
EXODUS_HEX_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 5), (2, 6), (3, 7), (4, 5), (5, 6), (6, 7), (7, 4))

# Levée collective : types d'erreur de lecture préservés tels quels sur tous les rangs (le message et le
# type de l'erreur du rang 0 sont diffusés, voir _read_mesh_data).
_ERROR_TYPES = {"FileNotFoundError": FileNotFoundError, "ValueError": ValueError, "RuntimeError": RuntimeError}


# -----------------------------------------------------------------------------
# Permutation Exodus -> basix (numérotation des nœuds d'un élément)
# -----------------------------------------------------------------------------
def _dof_entities(cell_type: str) -> tuple[basix.CellType, dict[int, tuple[int, int]]]:
    """Cellule basix et, pour chaque ddl, l'entité (dimension, numéro local) qui le porte."""
    family, degree = CELL_FAMILIES[cell_type]
    ct = basix.CellType.hexahedron
    element = basix.create_element(BASIX_FAMILIES[family], ct, degree, basix.LagrangeVariant.equispaced)
    entities = element.entity_dofs
    entities = list(entities.items()) if hasattr(entities, "items") else list(enumerate(entities))
    dof_entity: dict[int, tuple[int, int]] = {}
    for dim, per_entity in entities:
        for ent, dofs in enumerate(per_entity):
            for dof in dofs:
                dof_entity[int(dof)] = (int(dim), int(ent))
    return ct, dof_entity


def _entity_reference_point(ct: basix.CellType, dim: int, entity: int) -> np.ndarray:
    """Point de la cellule de référence basix porté par une entité : le sommet, ou le milieu de l'arête."""
    reference = np.asarray(basix.geometry(ct), dtype=np.float64)
    if dim == 0:
        return reference[entity]
    if dim == 1:
        sub = basix.cell.sub_entity_connectivity(ct)
        a, b = (int(v) for v in sub[1][entity][0])
        return (reference[a] + reference[b]) / 2.0
    raise RuntimeError(f"Entité de dimension {dim} non prévue (sommets et arêtes uniquement).")


def _exodus_reference_points(cell_type: str) -> np.ndarray:
    """Positions des nœuds locaux Exodus dans la cellule de référence : coins, puis milieux d'arête (HEX20)."""
    if cell_type == "HEX8":
        return EXODUS_REFERENCE_CORNERS
    midpoints = np.array([(EXODUS_REFERENCE_CORNERS[a] + EXODUS_REFERENCE_CORNERS[b]) / 2.0 for a, b in EXODUS_HEX_EDGES])
    return np.vstack([EXODUS_REFERENCE_CORNERS, midpoints])


def _compute_order(cell_type: str) -> np.ndarray:
    """
    Permutation Exodus -> basix : order[k] = nœud local Exodus à placer à la position k de basix.

    Purement combinatoire : le point de référence de l'entité porteuse de chaque ddl basix (sommet ou
    milieu d'arête) est apparié au nœud Exodus de même position dans la cellule de référence Exodus.
    La géométrie du maillage réel n'intervient pas ; le résultat est calculé à l'import et n'a pas à être
    redérivé (ni validé élément par élément) à chaque maillage.
    """
    ct, dof_entity = _dof_entities(cell_type)
    reference_points = _exodus_reference_points(cell_type)
    order = np.empty(len(dof_entity), dtype=np.int64)
    for dof, (dim, entity) in dof_entity.items():
        point = _entity_reference_point(ct, dim, entity)
        distances = np.linalg.norm(reference_points - point, axis=1)
        node = int(np.argmin(distances))
        if distances[node] > 1e-9:
            raise RuntimeError(f"{cell_type} : aucun nœud Exodus à la position de référence du ddl {dof}.")
        order[dof] = node
    if not np.array_equal(np.sort(order), np.arange(len(order))):
        raise RuntimeError(f"{cell_type} : permutation Exodus -> basix non bijective.")
    return order


# Permutations calculées à l'import (coût négligeable, résultat indépendant du maillage) :
# HEX8 = [0,1,3,2,4,5,7,6], HEX20 = [0,1,3,2,4,5,7,6,8,11,12,9,13,10,15,14,16,19,17,18]
# (valeurs vérifiées par tests/test_exodus.py).
EXODUS_ORDER: dict[str, np.ndarray] = {cell_type: _compute_order(cell_type) for cell_type in CELL_FAMILIES}


# -----------------------------------------------------------------------------
# Lecture et validation sur le rang 0, diffusion des sidesets
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class _MeshData:
    """Ce que tous les rangs doivent connaître du fichier : type d'élément, tailles, sidesets (petits)."""
    cell_type: str
    n_elems: int
    n_nodes: int
    sidesets: dict[str, tuple[np.ndarray, np.ndarray]]


def _check_mesh(ex: ExodusMesh) -> None:
    """
    Contrôles globaux du fichier, sur le rang 0 (seul détenteur des coordonnées et de la connectivité) :
    jacobien des coins strictement positif partout (maillage ni inversé ni dégénéré) et chaque facette
    portée par une ou deux cellules, jamais plus (maillage conforme). Chaque échec nomme le premier
    élément fautif, numéroté à partir de 1 comme dans Cubit.
    """
    order = EXODUS_ORDER[ex.cell_type]
    corners = ex.points[ex.cells[:, order[:8]]]                      # (n, 8, 3), coins en numérotation basix
    jacobian = np.linalg.det(
        np.stack([corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0], corners[:, 4] - corners[:, 0]], axis=-1)
    )
    inverted = np.where(jacobian <= 0.0)[0]
    if inverted.size:
        raise RuntimeError(
            f"{int(inverted.size)} élément(s) de jacobien négatif ou nul (premier : élément {int(inverted[0]) + 1}) : "
            "maillage inversé ou dégénéré ?"
        )
    sub = basix.cell.sub_entity_connectivity(basix.CellType.hexahedron)
    reordered = ex.cells[:, order[:8]]
    facets = np.concatenate([np.sort(reordered[:, list(sub[2][f][0])], axis=1) for f in range(6)], axis=0)
    _, carried_by = np.unique(facets, axis=0, return_counts=True)
    if carried_by.max() > 2:
        raise RuntimeError(
            f"{ex.cell_type} : une facette est portée par {int(carried_by.max())} éléments : maillage non conforme."
        )


def _read_mesh_data(comm: MPI.Comm, path: Path) -> tuple[_MeshData, ExodusMesh | None]:
    """
    Lit et valide le fichier sur le rang 0, diffuse l'en-tête (_MeshData, sidesets compris) aux autres
    rangs. Renvoie (data, ex) ; ex n'est non nul que sur le rang 0, seul à alimenter create_mesh.

    Une erreur de lecture/validation est diffusée avec son type et son message, puis levée à l'identique
    sur tous les rangs : sans cela, les autres rangs resteraient bloqués dans create_mesh.
    """
    data: _MeshData | None = None
    ex: ExodusMesh | None = None
    error: tuple[str, str] | None = None
    if comm.rank == 0:
        try:
            ex = read_exodus(path)
            _check_mesh(ex)
            data = _MeshData(cell_type=ex.cell_type, n_elems=ex.n_elems, n_nodes=ex.n_nodes, sidesets=ex.sidesets)
        except Exception as exc:  # noqa: BLE001 - l'erreur est relayée à tous les rangs ci-dessous
            ex, data = None, None
            error = (type(exc).__name__, str(exc))
    error, data = comm.bcast((error, data), root=0)
    if error is not None:
        raise _ERROR_TYPES.get(error[0], RuntimeError)(error[1])
    if data is None:  # inatteignable : sans erreur, le rang 0 a fourni les données
        raise RuntimeError("Lecture du maillage Exodus : aucune donnée reçue du rang 0.")
    return data, ex


def _raise_collective(comm: MPI.Comm, error: Exception | None) -> None:
    """
    Lève sur TOUS les rangs si au moins un rang a échoué (même motif que check_same_numbering) : une
    levée locale laisserait les autres rangs bloqués dans le prochain collectif (deadlock MPI). Le
    message du premier rang fautif est repris dans l'erreur commune.
    """
    messages = comm.allgather(None if error is None else f"{type(error).__name__}: {error}")
    failures = [(rank, message) for rank, message in enumerate(messages) if message is not None]
    if failures:
        rank, message = failures[0]
        raise RuntimeError(f"Échec sur le rang {rank} (levé sur tous les rangs) : {message}")


# -----------------------------------------------------------------------------
# Sidesets -> facettes taguées
# -----------------------------------------------------------------------------
def _local_facet_of_side(cell_type: str, order: np.ndarray) -> dict[int, int]:
    """Side Exodus (1..6) -> numéro de facette locale basix, via l'ensemble de ses coins."""
    sub = basix.cell.sub_entity_connectivity(basix.CellType.hexahedron)
    basix_of_corner = np.argsort(order[:8])                 # coin Exodus -> coin basix (inverse de l'ordre)
    out: dict[int, int] = {}
    for side, corners in EXODUS_HEX_FACES.items():
        target = frozenset(int(basix_of_corner[c]) for c in corners)
        for facet in range(6):
            if frozenset(int(v) for v in sub[2][facet][0]) == target:
                out[side] = facet
                break
        else:
            raise RuntimeError(f"Side Exodus {side} : facette locale introuvable ({cell_type}).")
    if len(set(out.values())) != 6:
        raise RuntimeError("Correspondance sides / facettes non bijective.")
    return out


def _sideset_facets(domain: dm.Mesh, data: _MeshData, order: np.ndarray) -> dict[str, np.ndarray]:
    """
    Facettes (numérotation locale dolfinx) de chaque sideset. En parallèle, chaque facette est traitée par
    le rang qui possède l'élément (topology.original_cell_index), donc taguée une seule fois au total.
    Une face interne ou une facette partagée par deux sidesets est refusée (tags ambigus).
    """
    tdim, fdim = domain.topology.dim, domain.topology.dim - 1
    domain.topology.create_entities(fdim)
    domain.topology.create_connectivity(tdim, fdim)
    domain.topology.create_connectivity(fdim, tdim)
    cell_facets = domain.topology.connectivity(tdim, fdim)
    facet_cells = domain.topology.connectivity(fdim, tdim)
    local_of_file = {int(global_cell): local for local, global_cell in enumerate(domain.topology.original_cell_index)}
    facet_of_side = _local_facet_of_side(data.cell_type, order)

    owner: dict[int, str] = {}
    facets: dict[str, np.ndarray] = {}
    for name, (elems, sides) in data.sidesets.items():
        tagged: set[int] = set()
        for elem, side in zip(elems, sides, strict=True):
            cell = local_of_file.get(int(elem))
            if cell is None:
                continue                                   # élément d'un autre rang : son propriétaire le traite
            facet = int(cell_facets.links(cell)[facet_of_side[int(side)]])
            if len(facet_cells.links(facet)) != 1:
                raise ValueError(
                    f"Sideset '{name}' : la face {int(side)} de l'élément {int(elem) + 1} est INTERNE au maillage "
                    "(un sideset doit référencer des faces de bord)."
                )
            earlier = owner.get(facet)
            if earlier is not None and earlier != name:
                raise ValueError(f"Facette partagée par les sidesets '{earlier}' et '{name}' : tags ambigus.")
            owner[facet] = name
            tagged.add(facet)
        facets[name] = np.array(sorted(tagged), dtype=np.int32)
    return facets


def _check_boundary_names(data: _MeshData, config: Config, mesh_file: Path) -> None:
    """Chaque boundary du YAML doit être un sideset du fichier ; l'inverse n'est qu'une information."""
    if not data.sidesets:
        raise ValueError("Le fichier Exodus ne contient aucun sideset : impossible de définir des conditions aux limites.")
    names = [bnd.name for bnd in config.boundaries]
    missing = [name for name in names if name not in data.sidesets]
    if missing:
        raise ValueError(
            f"boundaries : sideset(s) introuvable(s) dans {mesh_file.name} : {missing} "
            f"(sidesets disponibles : {', '.join(data.sidesets)})."
        )
    unused = [name for name in data.sidesets if name not in names]
    if unused:
        logger.debug("Sidesets non utilisés par l'étude : %s", ", ".join(unused))


def _facet_vertices(domain: dm.Mesh, facet: int) -> np.ndarray:
    """Coordonnées des nœuds d'une facette (4 pour HEX8, 8 pour HEX20), dans la configuration de référence."""
    fdim = domain.topology.dim - 1
    geometry_dofs = entities_to_geometry(domain, fdim, np.array([facet], dtype=np.int32), False)
    return np.asarray(domain.geometry.x[geometry_dofs[0]][:, :3], dtype=np.float64)


def _check_probe_points(domain: dm.Mesh, facets: dict[str, np.ndarray], config: Config) -> None:
    """
    Contrôle optionnel (probe_points fournis dans le YAML) : chaque point sonde doit se trouver à moins de
    probe_tol de la face effectivement taguée par son sideset. Détecte une inversion de sidesets dans le YAML.

    Le contrôle est GLOBAL : en parallèle, un rang ne possède qu'une partie des facettes du sideset (et
    parfois aucune), donc la distance est réduite sur l'ensemble des rangs avant comparaison à la tolérance.
    """
    fdim = domain.topology.dim - 1
    comm = domain.comm
    for bnd in config.boundaries:
        points = bnd.probe_points
        if not points:
            continue
        idx = np.asarray(facets[bnd.name], dtype=np.int32)
        for point in points:
            p = np.asarray(point, dtype=np.float64)
            best = np.inf
            if idx.size:
                midpoints = dm.compute_midpoints(domain, fdim, idx)
                nearest = np.argsort(np.linalg.norm(midpoints - p, axis=1))[:8]   # pré-filtre, puis distance exacte
                best = min(
                    float(np.linalg.norm(geometry.compute_distance_gjk(_facet_vertices(domain, int(idx[j])), p.reshape(1, 3))))
                    for j in nearest
                )
            gap = comm.allreduce(best, op=MPI.MIN)                # distance minimale sur l'ensemble des rangs
            if not np.isfinite(gap):
                raise ValueError(
                    f"Boundary '{bnd.name}' : aucune facette taguée (sideset vide sur tout le maillage ?)."
                )
            if gap > bnd.probe_tol:
                raise ValueError(
                    f"Boundary '{bnd.name}' : le point sonde {point} est à {gap:.4g} mm de la face taguée "
                    f"(tolérance {bnd.probe_tol} mm) : mauvais sideset ?"
                )


# -----------------------------------------------------------------------------
# Point d'entrée du backend
# -----------------------------------------------------------------------------
def build_mesh(comm: MPI.Comm, config: Config) -> MeshResult:
    """Construit le maillage dolfinx et les tags de facettes depuis un fichier Exodus II (Cubit)."""
    mesh_file = Path(config.mesh.file)
    data, ex = _read_mesh_data(comm, mesh_file)
    family, degree = CELL_FAMILIES[data.cell_type]
    order = EXODUS_ORDER[data.cell_type]
    element = basix.ufl.element(family, "hexahedron", degree, shape=(3,))

    # create_mesh attend la connectivité d'UN SEUL rang : dolfinx distribue ensuite maillage et nœuds
    # depuis ce rang d'entrée (« mesh nodes are only distributed from the input rank »). Passer
    # 'cells' sur tous les rangs ne lève aucune erreur mais corrompt le tas : plantage
    # malloc()/SIGABRT dès n > 1 (mesuré sur dolfinx 0.11.0, quel que soit le type d'élément). Ici,
    # seul le rang 0 détient points et connectivité (lus du fichier) : les autres rangs passent donc
    # naturellement des tableaux vides.
    if comm.rank == 0:
        if ex is None:  # inatteignable : sans erreur de lecture, le rang 0 détient le maillage complet
            raise RuntimeError("Lecture du maillage Exodus : maillage absent sur le rang 0.")
        cells = np.ascontiguousarray(ex.cells[:, order], dtype=np.int64)
        points = np.ascontiguousarray(ex.points, dtype=np.float64)
    else:
        cells = np.empty((0, len(order)), dtype=np.int64)
        points = np.empty((0, 3), dtype=np.float64)
    domain = dm.create_mesh(comm, cells, element, points)

    _check_boundary_names(data, config, mesh_file)

    facets: dict[str, np.ndarray] | None = None
    error: Exception | None = None
    try:
        facets = _sideset_facets(domain, data, order)
    except Exception as exc:  # noqa: BLE001 - erreur relayée à tous les rangs par _raise_collective
        error = exc
    _raise_collective(comm, error)
    if facets is None:  # inatteignable : sans erreur, les facettes ont été calculées
        raise RuntimeError("Taggage des sidesets : aucune facette calculée.")

    error = None
    try:
        _check_probe_points(domain, facets, config)
    except Exception as exc:  # noqa: BLE001 - erreur relayée à tous les rangs par _raise_collective
        error = exc
    _raise_collective(comm, error)

    fdim = domain.topology.dim - 1
    boundary_tags = {name: 2 + i for i, name in enumerate(facets)}       # tags 2, 3, ... dans l'ordre des sidesets
    indices = np.concatenate([facets[name] for name in boundary_tags])
    values = np.concatenate([np.full(facets[name].size, tag, dtype=np.int32) for name, tag in boundary_tags.items()])
    sort_idx = np.argsort(indices)
    facet_tags = dm.meshtags(domain, fdim, indices[sort_idx], values[sort_idx])

    volume = integrate(domain, ufl.dx)
    logger.debug(
        "Maillage Exodus %s : %s, %d éléments, %d nœuds", mesh_file.name, data.cell_type, data.n_elems, data.n_nodes
    )
    for name, tag in boundary_tags.items():
        total = comm.allreduce(int(facets[name].size), op=MPI.SUM)
        logger.debug("  boundary '%s' -> %d facette(s) au total, tag %d (%d sur ce rang)", name, total, tag, facets[name].size)

    return MeshResult(
        domain=domain,
        facet_tags=facet_tags,
        boundary_tags=boundary_tags,
        volume=volume,
        source="exodus",
        geom_degree=degree,
        description=f"hexaèdres {data.cell_type} (Cubit Exodus II)",
    )
