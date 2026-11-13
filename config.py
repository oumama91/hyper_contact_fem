"""
Configuration des études : modèles pydantic v2 du YAML, point d'entrée load_config().

Le YAML est validé déclarativement par les modèles ci-dessous (un modèle par section, union
discriminée sur le type des boundaries), là où l'ancien code enchaînait trois phases manuelles
(structure / défauts / valeurs). Conventions conservées :

  * extra="forbid" partout : une faute de frappe dans une clé est rejetée, jamais ignorée ;
  * types « stricts » au sens de l'ancienne validation manuelle : un nombre est un int ou un float
    (jamais un booléen, jamais une chaîne comme "20", que le mode laxiste de pydantic accepterait),
    un entier est un vrai int ; les bornes (> 0, >= 1, ...) sont portées par les champs ;
  * une seule loi de comportement : le néo-hookéen en formulation mixte (u, p) Taylor-Hood
    P_k - P_(k-1), donc mesh.order >= 2 ;
  * après validation, load_config résout les chemins relatifs par rapport au dossier du YAML
    (fichier .exo, outdir) et les réécrit dans le modèle, seul endroit où la configuration est mutée.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import Field, field_validator, model_validator

from config_types import (
    IntGe0,
    IntGe1,
    Integer,
    NonNegativeNumber,
    Number,
    PositiveNumber,
    Vec3,
    _Section,
)
from material import NeoHookean
from obstacles import CylinderObstacle, Obstacle, PlaneObstacle, SphereObstacle  # noqa: F401 (réexport)


# -----------------------------------------------------------------------------
# Catalogue des champs exportables vers ParaView (sans dépendance à dolfinx)
#
# Ce catalogue vit ici, et non dans postproc.py, pour éviter un import circulaire :
# config.py valide output.fields dès le chargement du YAML, alors que postproc.py importe
# config.py (et forms.py, qui importe config.py). postproc.py importe donc ces symboles.
#
# Le YAML choisit les champs dans `output` :
#     fields: all                                       # tous les champs du catalogue (défaut si absent)
#     fields: [von_mises, principal_stress, strain]     # sélection ('displacement' est toujours exporté)
#
# Chaque champ apparaît UNE fois dans le menu ParaView, avec ses composantes dans le second menu :
#     scalaire   : von_mises, mean_stress, tresca, equivalent_strain, J
#     vecteur    : displacement (X, Y, Z) ; principal_stress, principal_strain, principal_stretch
#                  (X, Y, Z = valeur 1, 2, 3 par ordre décroissant ; la « Magnitude » n'a pas de sens)
#     tenseur    : cauchy_stress, strain (9 composantes d'un tenseur symétrique 3x3)
#
# Contraintes en MPa, positives en traction. Toutes sont des contraintes de Cauchy (vraies).
# `mean_stress` vaut p (contrainte moyenne de la formulation mixte), `strain` est la déformation
# de Green-Lagrange E = (F^T F - I)/2, `principal_stretch` donne lambda_i = sqrt(1 + 2 E_i) et J
# le changement de volume (J ~ 1 si quasi incompressible). Les composantes de cisaillement sont
# tensorielles (pas la distorsion d'ingénieur).
#
# Catalogue provisoire de l'Étape 1 : tous les champs de l'ancien projet sont conservés. La
# réduction aux seuls champs utiles à un élastomère est prévue à l'Étape 2 (décision D3).
# -----------------------------------------------------------------------------
# (suffixe, i, j) des 6 composantes indépendantes d'un tenseur symétrique 3x3 (colonnes internes du post-traitement)
COMPONENTS = (("xx", 0, 0), ("yy", 1, 1), ("zz", 2, 2), ("xy", 0, 1), ("yz", 1, 2), ("xz", 0, 2))

# Champs disponibles pour toutes les lois, dans l'ordre d'export
COMMON_FIELDS = (
    "displacement", "von_mises", "cauchy_stress", "principal_stress", "mean_stress", "tresca",
    "strain", "principal_strain", "equivalent_strain",
)
# Champs qui n'ont de sens qu'en grandes déformations (hyperélasticité)
HYPERELASTIC_FIELDS = ("principal_stretch", "J")


def available_fields() -> tuple[str, ...]:
    """Champs exportables pour la loi hyperélastique."""
    return COMMON_FIELDS + HYPERELASTIC_FIELDS


def resolve_fields(spec: str | list[str] | None) -> tuple[str, ...]:
    """
    Convertit output.fields ('all', absent/None, ou liste de noms) en tuple de champs, dans l'ordre du catalogue.
    'displacement' est toujours inclus. Lève ValueError pour une spécification invalide.
    """
    available = available_fields()
    if spec is None or spec == "all":
        return available
    if not (isinstance(spec, list) and spec and all(isinstance(f, str) for f in spec)):
        raise ValueError(
            f"output.fields doit valoir 'all' ou une liste non vide de champs parmi {list(available)} (reçu : {spec!r})."
        )
    known = COMMON_FIELDS + HYPERELASTIC_FIELDS
    unknown = [f for f in spec if f not in known]
    if unknown:
        raise ValueError(f"output.fields : champs inconnus {unknown} (disponibles : {list(available)}).")
    wanted = set(spec) | {"displacement"}
    return tuple(f for f in available if f in wanted)


# -----------------------------------------------------------------------------
# Solveur
# -----------------------------------------------------------------------------
class NewtonConfig(_Section):
    increments: IntGe1 = 1        # nombre d'incréments de charge nominal
    max_iter: IntGe1 = 25         # itérations de Newton max par incrément
    rtol: PositiveNumber = 1.0e-6  # résidu libre relatif au résidu initial de l'incrément
    atol: NonNegativeNumber = 1.0e-10  # résidu libre absolu [N]
    max_cutbacks: IntGe0 = 8      # réductions de pas successives max


class SolverConfig(_Section):
    # rtol est conservé dans le schéma : le solveur linéaire est un LU direct MUMPS, qui ne
    # réalise pas d'itérations de Krylov, mais la clé existe toujours dans les études.
    rtol: PositiveNumber = 1.0e-8
    newton: NewtonConfig = Field(default_factory=NewtonConfig)


# -----------------------------------------------------------------------------
# Maillage (Exodus II uniquement)
# -----------------------------------------------------------------------------
class ExodusMeshConfig(_Section):
    source: Literal["exodus"] = "exodus"
    file: str
    order: IntGe1 = 1                      # ordre des éléments (espace de fonctions)
    quadrature_degree: IntGe1 | None = None  # None = estimation automatique d'UFL


# -----------------------------------------------------------------------------
# Matériau : néo-hookéen mixte uniquement
# -----------------------------------------------------------------------------
class MaterialConfig(_Section):
    model: Literal["neo_hookean"] = "neo_hookean"
    E: Number
    nu: Number
    rho: NonNegativeNumber

    def params(self) -> dict[str, float | str]:
        """Paramètres transmis à la loi néo-hookéenne (instanciée par forms.py)."""
        return {"model": self.model, "E": self.E, "nu": self.nu, "rho": self.rho}


# -----------------------------------------------------------------------------
# Boundaries (union discriminée sur le type)
# -----------------------------------------------------------------------------
class _BoundaryBase(_Section):
    name: str
    probe_points: list[Vec3] = Field(default_factory=list)
    probe_tol: PositiveNumber = 0.1


class DirichletBoundary(_BoundaryBase):
    type: Literal["dirichlet"]
    components: list[Integer]
    value: list[Number]

    @field_validator("components")
    @classmethod
    def _check_components(cls, value: list[int]) -> list[int]:
        if not value or len(set(value)) != len(value) or any(not 0 <= comp < 3 for comp in value):
            raise ValueError("'components' doit être une liste non vide d'entiers distincts dans {0, 1, 2}.")
        return value

    @model_validator(mode="after")
    def _check_value_length(self) -> DirichletBoundary:
        if len(self.value) != len(self.components):
            raise ValueError(f"'value' doit contenir autant de nombres que 'components' ({len(self.components)}).")
        return self


# -----------------------------------------------------------------------------
# Contact unilatéral contre un obstacle rigide analytique : la géométrie de l'obstacle
# (PlaneObstacle/CylinderObstacle/SphereObstacle/Obstacle) vit dans obstacles.py (réexportée
# ci-dessus) ; seule la boundary elle-même (pénalité, frottement) est déclarée ici.
# -----------------------------------------------------------------------------
class ContactBoundary(_BoundaryBase):
    """
    Contact unilatéral (pénalité) contre un obstacle rigide analytique, avec frottement de Coulomb
    régularisé (formulation holonome, voir contact.py : valide pour un chargement monotone,
    sans décharge/inversion du glissement).
    """
    type: Literal["contact"]
    obstacle: Obstacle
    epsilon_n: PositiveNumber              # pénalité normale [MPa/mm] (= N/mm^3)
    epsilon_t: PositiveNumber | None = None  # pénalité tangentielle [MPa/mm] ; défaut : epsilon_n
    friction: NonNegativeNumber = 0.0      # coefficient de Coulomb (0 = contact glissant sans frottement)


Boundary = Annotated[DirichletBoundary | ContactBoundary, Field(discriminator="type")]


class PhysicsConfig(_Section):
    gravity: Vec3 = Field(default_factory=lambda: [0.0, 0.0, 0.0])


class OutputConfig(_Section):
    fields: str | list[str] = "all"  # 'all' ou liste de champs (voir le catalogue ci-dessus)
    outdir: str

    @field_validator("fields", mode="before")
    @classmethod
    def _default_fields(cls, value: object) -> object:
        return "all" if value is None else value


# -----------------------------------------------------------------------------
# Racine
# -----------------------------------------------------------------------------
class StudyConfig(_Section):
    """Configuration complète d'une étude (contenu du YAML, valeurs par défaut appliquées)."""
    study_name: str
    mesh: ExodusMeshConfig
    material: MaterialConfig
    physics: PhysicsConfig = Field(default_factory=PhysicsConfig)
    boundaries: list[Boundary]
    solver: SolverConfig
    output: OutputConfig

    @model_validator(mode="after")
    def _check_cross_rules(self) -> StudyConfig:
        # Matériau : bornes propres au néo-hookéen (messages français historiques).
        NeoHookean.check_values(self.material.params())

        # Formulation mixte u-p : Taylor-Hood P_k - P_(k-1) impose l'ordre 2 au minimum.
        if self.mesh.order < 2:
            raise ValueError(
                "mesh.order >= 2 requis pour une loi hyperélastique (formulation mixte u-p : Taylor-Hood P_k - P_(k-1))."
            )

        # Catalogue des champs exportables.
        resolve_fields(self.output.fields)

        # Garde-fou : le terme J^(-2/3) fait exploser le degré de quadrature estimé automatiquement
        # par UFL si on ne le fixe pas explicitement (assemblage très lent, voire bloquant).
        if self.mesh.quadrature_degree is None:
            warnings.warn(
                "mesh.quadrature_degree n'est pas fourni pour une loi hyperélastique : le degré de quadrature "
                "estimé automatiquement par UFL (à cause de J^(-2/3)) peut être très élevé et ralentir "
                "fortement l'assemblage. Il est recommandé de le fixer explicitement "
                "(voir studies/O-ring_compression.yaml).",
                stacklevel=2,
            )

        if not self.boundaries:
            raise ValueError("La section 'boundaries' doit être une liste non vide.")
        names: set[str] = set()
        n_dirichlet = 0
        for bnd in self.boundaries:
            if bnd.name in names:
                raise ValueError(f"Nom de boundary dupliqué : '{bnd.name}'.")
            names.add(bnd.name)
            if isinstance(bnd, DirichletBoundary):
                n_dirichlet += 1
        if n_dirichlet == 0:
            raise ValueError("Au moins une condition de Dirichlet est nécessaire (problème singulier sinon).")
        return self


# Alias historique : les modules annotent leurs paramètres 'config: Config'.
Config = StudyConfig


def parse_config(raw: object) -> StudyConfig:
    """Valide le contenu d'un YAML (déjà chargé) et renvoie la configuration typée, défauts appliqués."""
    if not isinstance(raw, dict):
        raise ValueError("Le YAML doit contenir un dictionnaire à la racine.")
    return StudyConfig.model_validate(raw)


# -----------------------------------------------------------------------------
# Point d'entrée
# -----------------------------------------------------------------------------
def load_config(yaml_path: str) -> StudyConfig:
    """Charge le YAML, le valide (modèles ci-dessus), puis résout les chemins relatifs au dossier du YAML."""
    path = Path(yaml_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Fichier de configuration introuvable : {path}")

    with open(path, encoding="utf-8") as f:
        config = parse_config(yaml.safe_load(f))

    base_dir = path.parent
    mesh_path = Path(config.mesh.file)
    if not mesh_path.is_absolute():
        mesh_path = (base_dir / mesh_path).resolve()
    # Existence vérifiée à la lecture par exodus_reader.read_exodus (seul lecteur du fichier).
    config.mesh.file = str(mesh_path)

    out_path = Path(config.output.outdir)
    if not out_path.is_absolute():
        out_path = (base_dir / out_path).resolve()
    config.output.outdir = str(out_path)
    return config
