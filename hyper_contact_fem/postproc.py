"""
Post-traitement : champs aux points de quadrature -> projection nodale P1 -> export VTX (ParaView).

Pourquoi tous les champs d'export passent par les MÊMES espaces (scalaire, vecteur, tenseur) :
VTXWriter construit le maillage à partir de l'espace du PREMIER champ de la liste et écrit les autres
champs tels quels, sans réordonner. Un champ issu d'un autre espace (par exemple un sous-espace « collapsé »
d'un espace mixte, numéroté autrement) donne des valeurs justes placées sur de mauvais nœuds : un champ
apparemment bruité. Tout est donc interpolé vers des espaces d'export autonomes (voir projection.py),
dont la numérotation commune est vérifiée avant l'écriture.

Projection nodale : L2 à masse concentrée vers P1 (moyenne pondérée aux sommets), en une seule passe pour
tous les champs. Pression moyenne, J, composantes de contrainte et de déformation sont évalués aux points
de quadrature ; les valeurs principales (et les allongements principaux) sont calculées ensuite par
diagonalisation des tenseurs nodaux (algèbre tensorielle ci-dessous). Les tenseurs
sortent en champs 3x3 (9 composantes) et les valeurs principales en champs vectoriels : un seul menu
ParaView par grandeur. Catalogue et conventions : config.py.

postprocess() enchaîne les étapes suivantes :
    _max_displacement        -> métrique u_max
    _quadrature_expressions  -> expressions UFL à projeter (une par colonne interne)
    project_p1 (projection.py)   -> projection nodale P1 (une passe)
    nodal_columns (projection.py) -> interpolation vers l'ordre du maillage, colonnes nommées
    _export_values            -> valeurs d'export (scalaires, vecteurs, tenseurs, valeurs principales)
    write_vtx                -> écriture VTX

L'indicateur de contrainte du résumé est la plus grande contrainte principale de Cauchy (max de
sigma_1 sur les nœuds) : en hyperélasticité quasi incompressible, sigma_1 est la grandeur pertinente
pour un élastomère (traction maximale), et elle est déjà exportée.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.io import VTXWriter
from dolfinx.mesh import Mesh
from mpi4py import MPI

from hyper_contact_fem.config import COMPONENTS, Config, resolve_fields
from hyper_contact_fem.forms import Forms
from hyper_contact_fem.projection import export_spaces, nodal_columns, project_p1
from hyper_contact_fem.utils import Expr, ensure_dir

# Champs d'export qui exigent le tenseur des contraintes / des déformations
STRESS_FIELDS = frozenset({"cauchy_stress", "principal_stress"})
STRAIN_FIELDS = frozenset({"strain", "principal_stretch"})

# Point d'observation optionnel de l'instrumentation (tools/capture_metrics.py) : il reçoit le
# déplacement RÉELLEMENT exporté (u_out après interpolation), jamais réinjecté dans le calcul.
# None en usage normal : aucune instruction numérique supplémentaire n'est exécutée.
displacement_observer: Callable[[fem.Function], None] | None = None


def _owned_rows(f: fem.Function, n_cols: int) -> np.ndarray:
    """Lignes (nœuds, composantes) d'une Function à n_cols composantes, nœuds possédés par ce processus."""
    n_own = f.function_space.dofmap.index_map.size_local
    return f.x.array[: n_own * n_cols].reshape(-1, n_cols)


def _owned_max(comm: MPI.Comm, values: np.ndarray) -> float:
    """Maximum global d'un tableau 1D de valeurs possédées (réduit sur tous les rangs)."""
    # Neutre d'un maximum : -inf (et non 0), sinon un rang sans valeur relèverait un maximum négatif à 0.
    return comm.allreduce(float(np.max(values)) if len(values) else -np.inf, op=MPI.MAX)


def _max_displacement(comm: MPI.Comm, u_h: fem.Function, gdim: int) -> float:
    """Norme maximale du déplacement sur les nœuds possédés, réduite sur tous les rangs."""
    u = _owned_rows(u_h, gdim)
    return _owned_max(comm, np.linalg.norm(u, axis=1))


# -----------------------------------------------------------------------------
# Algèbre tensorielle nodale (numpy pur, aucun accès maillage)
# -----------------------------------------------------------------------------
def symmetric(cols: dict[str, np.ndarray], prefix: str) -> np.ndarray:
    """Assemble les tenseurs symétriques 3x3 nodaux (m, 3, 3) depuis leurs 6 colonnes '<prefix>_xx'..."""
    m = len(cols[f"{prefix}_xx"])
    T = np.empty((m, 3, 3))
    for tag, i, j in COMPONENTS:
        T[:, i, j] = T[:, j, i] = cols[f"{prefix}_{tag}"]
    return T


def principal_values(T: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Valeurs principales décroissantes (v1 >= v2 >= v3) de tenseurs symétriques nodaux."""
    w = np.linalg.eigvalsh(T)  # croissant
    return w[:, 2], w[:, 1], w[:, 0]


# -----------------------------------------------------------------------------
# Post-traitement
# -----------------------------------------------------------------------------
def _quadrature_expressions(forms: Forms, wanted: set[str]) -> dict[str, Expr]:
    """
    Expressions UFL scalaires aux points de quadrature : une par colonne interne du post-traitement.

    Le tenseur de Cauchy est toujours projeté : il porte l'indicateur sigma_1_max du résumé (et les
    champs cauchy_stress / principal_stress s'ils sont demandés), même si l'utilisateur restreint
    output.fields.
    """
    sigma = forms.cauchy_ufl()
    u = forms.displacement_ufl()

    exprs: dict[str, Expr] = {}
    if "mean_stress" in wanted:
        exprs["mean_stress"] = ufl.tr(sigma) / 3.0
    if "J" in wanted:
        exprs["J"] = forms.material.volume_ratio(u)
    for tag, i, j in COMPONENTS:
        exprs[f"sigma_{tag}"] = sigma[i, j]
    if STRAIN_FIELDS & wanted:
        strain = forms.material.strain(u)
        for tag, i, j in COMPONENTS:
            exprs[f"strain_{tag}"] = strain[i, j]
    return exprs


def _export_values(cols: dict[str, np.ndarray], wanted: set[str]) -> dict[str, np.ndarray]:
    """
    Valeurs d'export : nom -> tableau (nœuds, composantes) ;
    1 = scalaire, 3 = vecteur, 9 = tenseur 3x3.
    """
    values: dict[str, np.ndarray] = {}
    for scalar in ("mean_stress", "J"):
        if scalar in wanted:
            values[scalar] = cols[scalar][:, None]
    if STRESS_FIELDS & wanted:
        S = symmetric(cols, "sigma")
        s1, s2, s3 = principal_values(S)
        if "cauchy_stress" in wanted:
            values["cauchy_stress"] = S.reshape(-1, 9)
        if "principal_stress" in wanted:
            values["principal_stress"] = np.column_stack((s1, s2, s3))
    if STRAIN_FIELDS & wanted:
        E = symmetric(cols, "strain")
        if "strain" in wanted:
            values["strain"] = E.reshape(-1, 9)
        if "principal_stretch" in wanted:
            e1, e2, e3 = principal_values(E)
            values["principal_stretch"] = np.sqrt(np.maximum(1.0 + 2.0 * np.column_stack((e1, e2, e3)), 0.0))
    return values


@dataclass(frozen=True)
class PostprocResult:
    """Résultat du post-traitement."""
    u_max: float        # norme de déplacement maximale [mm]
    sigma_1_max: float  # plus grande contrainte principale de Cauchy, lissée P1 [MPa]
    vtx_path: str       # chemin du fichier VTX écrit (ParaView)
    fields: list[str]   # champs effectivement exportés, dans l'ordre d'écriture


def write_vtx(
    comm: MPI.Comm,
    config: Config,
    spaces: dict[int, fem.FunctionSpace],
    u_h: fem.Function,
    requested: tuple[str, ...],
    values: dict[str, np.ndarray],
) -> tuple[str, list[str]]:
    """Export VTX : le déplacement (vectoriel autonome) en premier, puis les champs demandés."""
    u_out = fem.Function(spaces[3], name="displacement")
    u_out.interpolate(u_h)
    u_out.x.scatter_forward()
    if displacement_observer is not None:
        displacement_observer(u_out)  # instrumentation de capture : ne modifie ni u_out ni l'export
    fields = [u_out]
    exported = ["displacement"]

    for name in requested:
        if name == "displacement":
            continue
        data = values[name]
        f = fem.Function(spaces[data.shape[1]], name=name)
        f.x.array[:] = data.reshape(-1)
        fields.append(f)
        exported.append(name)

    outdir = ensure_dir(comm, Path(config.output.outdir))

    vtx_path = outdir / f"{config.study_name}.bp"
    with VTXWriter(comm, vtx_path, fields, engine="BP4") as w:
        w.write(0.0)
    return str(vtx_path), exported


def postprocess(comm: MPI.Comm, domain: Mesh, forms: Forms, config: Config) -> PostprocResult:
    """
    Calcule les champs demandés (output.fields, voir config.py), extrait les métriques globales
    et exporte en VTX. 'forms' porte la solution (uh), la loi de comportement et les accès symboliques
    (displacement_ufl, cauchy_ufl).
    Retourne un PostprocResult (u_max, sigma_1_max, fichier VTX, champs exportés).
    """
    gdim = domain.geometry.dim
    if gdim != 3:
        raise NotImplementedError("post-traitement : maillages 3D uniquement.")
    requested = resolve_fields(config.output.fields)
    wanted = set(requested)

    u_h = forms.displacement()
    u_max = _max_displacement(comm, u_h, gdim)

    exprs = _quadrature_expressions(forms, wanted)
    names = list(exprs)
    nodal = project_p1(domain, forms.dx, list(exprs.values()))

    spaces = export_spaces(comm, domain, config.mesh.order, gdim, len(names))
    cols = nodal_columns(nodal, spaces[0], names)
    values = _export_values(cols, wanted)

    # Indicateur de contrainte : plus grande contrainte principale de Cauchy (nœuds possédés,
    # la projection P1 est réutilisée telle quelle pour les colonnes sigma).
    n_own = spaces[1].dofmap.index_map.size_local
    sigma_1 = principal_values(symmetric(cols, "sigma"))[0]
    sigma_1_max = _owned_max(comm, sigma_1[:n_own])

    vtx_path, exported = write_vtx(comm, config, spaces, u_h, requested, values)

    return PostprocResult(u_max=u_max, sigma_1_max=sigma_1_max, vtx_path=vtx_path, fields=exported)
