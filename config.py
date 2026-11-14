"""
Configuration des études : modèles pydantic v2 du YAML, point d'entrée load_config().

Le YAML est validé déclarativement par les modèles ci-dessous (un modèle par section, union
discriminée sur le type des boundaries, union discriminée sur le type des obstacles).
Règles :

  * extra="forbid" partout : une faute de frappe dans une clé est rejetée, jamais ignorée ;
  * types natifs pydantic v2 : bornes portées par les champs (PositiveFloat, Field(gt=...)),
    entiers stricts (StrictInt : ni booléen, ni chaîne comme "20") ;
  * une seule loi de comportement : le néo-hookéen en formulation mixte (u, p) Taylor-Hood
    P_k - P_(k-1), donc mesh.order >= 2 ;
  * chaque clé a un effet : un seul format de maillage (Exodus II), loi et solveur linéaire
    (LU direct MUMPS) fixes, sans paramètre de Krylov ;
  * après validation, load_config résout les chemins relatifs par rapport au dossier du YAML
    (fichier .exo, outdir) et les réécrit dans le modèle, seul endroit où la configuration est mutée.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    StrictInt,
    field_validator,
    model_validator,
)


class _Section(BaseModel):
    """Base des sections : aucune clé inconnue n'est tolérée (faute de frappe = erreur)."""
    model_config = ConfigDict(extra="forbid")


# -----------------------------------------------------------------------------
# Catalogue des champs exportables vers ParaView (sans dépendance à dolfinx)
#
# Ce catalogue vit ici, et non dans postproc.py, pour éviter un import circulaire :
# config.py valide output.fields dès le chargement du YAML, alors que postproc.py importe
# config.py (et forms.py, qui importe config.py). postproc.py importe donc ces symboles.
#
# Le YAML choisit les champs dans `output` :
#     fields: all                                       # tous les champs du catalogue (défaut si absent)
#     fields: [cauchy_stress, J]                        # sélection ('displacement' est toujours exporté)
#
# Décision de conception : export restreint aux grandeurs utiles à un élastomère.
#     scalaire   : mean_stress (= p de la formulation mixte), J
#     vecteur    : displacement (X, Y, Z) ; principal_stress, principal_stretch
#                  (X, Y, Z = valeur 1, 2, 3 par ordre décroissant ; la « Magnitude » n'a pas de sens)
#     tenseur    : cauchy_stress, strain (9 composantes d'un tenseur symétrique 3x3)
#
# Contraintes en MPa, positives en traction. Toutes sont des contraintes de Cauchy (vraies).
# `strain` est la déformation de Green-Lagrange E = (F^T F - I)/2, `principal_stretch` donne
# lambda_i = sqrt(1 + 2 E_i) et J le changement de volume (J ~ 1 si quasi incompressible). Les
# composantes de cisaillement sont tensorielles (pas la distorsion d'ingénieur).
# -----------------------------------------------------------------------------
# (suffixe, i, j) des 6 composantes indépendantes d'un tenseur symétrique 3x3 (colonnes internes du post-traitement)
COMPONENTS = (("xx", 0, 0), ("yy", 1, 1), ("zz", 2, 2), ("xy", 0, 1), ("yz", 1, 2), ("xz", 0, 2))

# Champs disponibles, dans l'ordre d'export
COMMON_FIELDS = ("displacement", "cauchy_stress", "principal_stress", "mean_stress", "strain")
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
    increments: StrictInt = Field(default=1, ge=1)   # nombre d'incréments de charge nominal
    max_iter: StrictInt = Field(default=25, ge=1)    # itérations de Newton max par incrément
    rtol: PositiveFloat = 1.0e-6                     # résidu libre relatif au résidu initial de l'incrément
    atol: NonNegativeFloat = 1.0e-10                 # résidu libre absolu [N]
    max_cutbacks: StrictInt = Field(default=8, ge=0)  # réductions de pas successives max


class SolverConfig(_Section):
    newton: NewtonConfig = Field(default_factory=NewtonConfig)


# -----------------------------------------------------------------------------
# Maillage (Exodus II uniquement)
# -----------------------------------------------------------------------------
class ExodusMeshConfig(_Section):
    file: str
    order: StrictInt = Field(default=2, ge=2)             # ordre des éléments (Taylor-Hood P_k - P_(k-1))
    quadrature_degree: StrictInt = Field(default=4, ge=1)  # degré de quadrature du volume


# -----------------------------------------------------------------------------
# Matériau : néo-hookéen mixte uniquement
# -----------------------------------------------------------------------------
class MaterialConfig(_Section):
    E: PositiveFloat
    nu: float = Field(gt=-1.0, le=0.5)  # mixte : nu = 0.5 (incompressible) admis
    rho: NonNegativeFloat = 0.0

    def params(self) -> dict[str, float]:
        """Paramètres transmis à la loi néo-hookéenne (instanciée par forms.py)."""
        return {"E": self.E, "nu": self.nu, "rho": self.rho}


# -----------------------------------------------------------------------------
# Obstacles rigides analytiques : trois formes, union discriminée sur 'kind'. Chacune ne décrit
# QUE la géométrie de l'obstacle rigide (pas la pénalité ni le frottement, portés par
# ContactBoundary) ; contact.py::gap_and_normal sait calculer jauge signée et normale pour chacune.
# -----------------------------------------------------------------------------
Vec3 = Annotated[list[float], Field(min_length=3, max_length=3)]


def _check_nonzero_direction(value: list[float], name: str) -> None:
    """Une normale ou un axe d'obstacle doit être non nul (son sens seul compte, il est normalisé ensuite)."""
    if math.sqrt(sum(c * c for c in value)) <= 1.0e-12:
        raise ValueError(f"'{name}' doit être un vecteur non nul")


class PlaneObstacle(_Section):
    """Demi-espace rigide : {x : (x - point)·normal >= 0} est le côté admissible (hors matière rigide)."""
    kind: Literal["plane"] = "plane"
    point: Vec3            # un point du plan, en configuration de référence [mm]
    normal: Vec3           # normale sortante (côté admissible) ; normalisée en interne, sens seul compte

    @model_validator(mode="after")
    def _check_normal(self) -> PlaneObstacle:
        _check_nonzero_direction(self.normal, "normal")
        return self


class CylinderObstacle(_Section):
    """Cylindre rigide infini (axe + rayon). 'outside' : le solide est à l'extérieur (axe = matière rigide) ;
    'inside' : le solide est à l'intérieur (alésage rigide, matière rigide à l'extérieur du rayon)."""
    kind: Literal["cylinder"]
    axis_point: Vec3
    axis_dir: Vec3
    radius: PositiveFloat
    orientation: Literal["outside", "inside"] = "outside"

    @model_validator(mode="after")
    def _check_axis(self) -> CylinderObstacle:
        _check_nonzero_direction(self.axis_dir, "axis_dir")
        return self


class SphereObstacle(_Section):
    """Sphère rigide. 'outside' : le solide est à l'extérieur de la sphère ; 'inside' : cavité sphérique rigide."""
    kind: Literal["sphere"]
    center: Vec3
    radius: PositiveFloat
    orientation: Literal["outside", "inside"] = "outside"


Obstacle = Annotated[PlaneObstacle | CylinderObstacle | SphereObstacle, Field(discriminator="kind")]


# -----------------------------------------------------------------------------
# Boundaries (union discriminée sur le type)
# -----------------------------------------------------------------------------
class _BoundaryBase(_Section):
    name: str
    probe_points: list[Vec3] = Field(default_factory=list)
    probe_tol: PositiveFloat = 0.1


class DirichletBoundary(_BoundaryBase):
    type: Literal["dirichlet"]
    components: list[StrictInt]
    value: list[float]

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


class ContactBoundary(_BoundaryBase):
    """
    Contact unilatéral (pénalité) contre un obstacle rigide analytique, avec frottement de Coulomb
    régularisé (formulation holonome, voir contact.py : valide pour un chargement monotone,
    sans décharge/inversion du glissement).
    """
    type: Literal["contact"]
    obstacle: Obstacle
    epsilon_n: PositiveFloat              # pénalité normale [MPa/mm] (= N/mm^3)
    epsilon_t: PositiveFloat | None = None  # pénalité tangentielle [MPa/mm] ; défaut : epsilon_n
    friction: NonNegativeFloat = 0.0      # coefficient de Coulomb (0 = contact glissant sans frottement)


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
        # Catalogue des champs exportables.
        resolve_fields(self.output.fields)

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


# Alias utilisé par les annotations des modules ('config: Config').
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
