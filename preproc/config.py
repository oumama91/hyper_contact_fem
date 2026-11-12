"""
Configuration des études : modèles pydantic v2 du YAML, point d'entrée load_config().

Le YAML est validé déclarativement par les modèles ci-dessous (un modèle par section, unions
discriminées sur mesh.source et sur le type des boundaries et des raffinements), là où l'ancien code
enchaînait trois phases manuelles (structure / défauts / valeurs). Conventions conservées :

  * extra="forbid" partout : une faute de frappe dans une clé est rejetée, jamais ignorée ;
  * types « stricts » au sens de l'ancienne validation manuelle : un nombre est un int ou un float
    (jamais un booléen, jamais une chaîne comme "20", que le mode laxiste de pydantic accepterait),
    un entier est un vrai int ; les bornes (> 0, >= 1, ...) sont portées par les champs ;
  * les messages des règles métier (sections croisées, règles propres à une loi, champs disponibles)
    restent les messages français historiques, émis par des validateurs : pydantic les emballe dans
    ValidationError (sous-classe de ValueError) avec le chemin du champ fautif (ex. mesh.size_max) ;
  * la section 'material' n'est PAS fermée par un modèle par loi : seuls 'model' et 'rho' sont
    déclarés ici, les paramètres propres à la loi (E, nu, et ceux d'une future loi) sont validés par
    Material.validate via le registre MATERIAL_MODELS. Ajouter une loi ne touche donc pas ce module ;
  * après validation, load_config résout les chemins relatifs par rapport au dossier du YAML
    (cad_file ou fichier .exo selon la source, outdir) et les réécrit dans le modèle, seul endroit
    où la configuration est mutée.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import ConfigDict, Field, field_validator, model_validator
from pydantic.types import Strict

from preproc.config_types import (
    IntGe0,
    IntGe1,
    Integer,
    NonNegativeNumber,
    Number,
    PositiveNumber,
    Vec3,
    _Section,
)
from preproc.fields import resolve_fields
from preproc.materials import DEFAULT_MODEL, MATERIAL_MODELS
from preproc.obstacles import CylinderObstacle, Obstacle, PlaneObstacle, SphereObstacle  # noqa: F401 (réexport)

# Source du maillage : CAO STEP maillée par gmsh, ou fichier Exodus II (.exo) exporté par Cubit.
# Registre UNIQUE des backends (source -> module, source -> dépendance externe) :
# preproc/mesher.py l'importe, au lieu de redéfinir sa propre correspondance.
MESH_BACKENDS = {"gmsh": "preproc.mesher_gmsh", "exodus": "preproc.mesher_exodus"}
MESH_REQUIREMENTS = {"gmsh": "gmsh", "exodus": "netCDF4"}
MESH_SOURCES = tuple(MESH_BACKENDS)
DEFAULT_SOURCE = "gmsh"

# Clés de génération du maillage, sans objet avec un maillage Exodus (le maillage est fait dans Cubit) :
# présentes par erreur, elles sont rejetées plutôt qu'ignorées.
GMSH_ONLY_MESH_KEYS = ("size_max", "geom_order", "curvature_nodes", "refinements")

SOLVER_METHODS = ("gamg", "direct")
KRYLOV_TYPES = ("cg", "gmres")


# -----------------------------------------------------------------------------
# Types stricts (Number, Vec3, _Section, ...) : voir preproc/config_types.py (extraits dans ce module
# pour être partagés sans dépendance circulaire avec preproc/obstacles.py).
# -----------------------------------------------------------------------------


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
    method: str  # validé ci-dessous (message français historique)
    rtol: PositiveNumber = 1.0e-8  # tolérance du solveur linéaire interne (KSP)
    krylov: str = "cg"            # utilisé avec method: gamg ('gmres' si tangente non SPD)
    check_nullspace: Annotated[bool, Strict()] = False  # debug : un assemblage supplémentaire
    newton: NewtonConfig = Field(default_factory=NewtonConfig)

    @field_validator("method", mode="before")
    @classmethod
    def _check_method(cls, value: object) -> object:
        if value not in SOLVER_METHODS:
            raise ValueError(f"solver.method inconnu : {value!r} (attendu : {', '.join(SOLVER_METHODS)}).")
        return value

    @field_validator("krylov", mode="before")
    @classmethod
    def _check_krylov(cls, value: object) -> object:
        if value not in KRYLOV_TYPES:
            raise ValueError(f"solver.krylov inconnu : {value!r} (attendu : {', '.join(KRYLOV_TYPES)}).")
        return value


# -----------------------------------------------------------------------------
# Maillage (union discriminée sur la source)
# -----------------------------------------------------------------------------
class BoundaryRefinement(_Section):
    type: Literal["boundary"] = "boundary"
    boundary_name: str
    refine_factor: PositiveNumber = 2.0
    dist_max: PositiveNumber = 100.0  # distance de retour à size_max [mm]


class SphereRefinement(_Section):
    type: Literal["sphere"]
    center: Vec3
    radius: PositiveNumber
    transition: NonNegativeNumber | None = None  # épaisseur de transition [mm] ; défaut : radius
    refine_factor: PositiveNumber = 2.0

    @model_validator(mode="after")
    def _default_transition(self) -> SphereRefinement:
        if self.transition is None:
            self.transition = self.radius
        return self


Refinement = Annotated[BoundaryRefinement | SphereRefinement, Field(discriminator="type")]


class _MeshBase(_Section):
    order: IntGe1 = 1                      # ordre des éléments (espace de fonctions), commun aux deux sources
    quadrature_degree: IntGe1 | None = None  # None = estimation automatique d'UFL


class GmshMeshConfig(_MeshBase):
    source: Literal["gmsh"] = "gmsh"
    size_max: PositiveNumber
    geom_order: IntGe1 = 1                 # ordre de la géométrie du maillage
    curvature_nodes: IntGe0 = 0
    refinements: list[Refinement] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _reject_exodus_file(cls, data: object) -> object:
        if isinstance(data, dict) and "file" in data:
            raise ValueError("mesh.file n'est valide qu'avec mesh.source: exodus.")
        return data

    @field_validator("refinements", mode="before")
    @classmethod
    def _default_refinement_type(cls, value: object) -> object:
        if isinstance(value, list):
            return [{**ref, "type": ref.get("type", "boundary")} if isinstance(ref, dict) else ref for ref in value]
        return value


class ExodusMeshConfig(_MeshBase):
    source: Literal["exodus"]
    file: str

    @model_validator(mode="before")
    @classmethod
    def _reject_gmsh_only_keys(cls, data: object) -> object:
        if isinstance(data, dict):
            for key in GMSH_ONLY_MESH_KEYS:
                if key in data:
                    raise ValueError(
                        f"mesh.{key} n'est pas applicable avec mesh.source: exodus : générez le maillage "
                        "(taille, raffinements, ordre géométrique) dans Cubit puis réexportez le .exo."
                    )
        return data


MeshConfig = Annotated[GmshMeshConfig | ExodusMeshConfig, Field(discriminator="source")]


class GeometryConfig(_Section):
    cad_file: str


# -----------------------------------------------------------------------------
# Matériau : paramètres génériques ici, paramètres de loi via le registre
# -----------------------------------------------------------------------------
class MaterialConfig(_Section):
    # Les paramètres propres à la loi (E, nu, ...) sont acceptés ici (extra="allow") et validés par
    # Material.validate : fermer ce modèle par loi casserait l'extensibilité du registre.
    model_config = ConfigDict(extra="allow")

    model: str = DEFAULT_MODEL
    rho: NonNegativeNumber

    @field_validator("model", mode="before")
    @classmethod
    def _check_model(cls, value: object) -> object:
        if value not in MATERIAL_MODELS:
            raise ValueError(f"material.model inconnu : {value!r} (attendu : {', '.join(MATERIAL_MODELS)}).")
        return value

    def params(self) -> dict[str, Any]:
        """Dictionnaire complet des paramètres (contrat historique de Material.validate / build_material)."""
        return {"model": self.model, "rho": self.rho, **(self.__pydantic_extra__ or {})}


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


class NeumannBoundary(_BoundaryBase):
    type: Literal["neumann"]
    force_vector: Vec3 | None = None     # [N] force totale, répartie uniformément sur la face
    traction_vector: Vec3 | None = None  # [MPa] traction uniforme

    @model_validator(mode="after")
    def _check_exactly_one_loading(self) -> NeumannBoundary:
        if (self.force_vector is None) == (self.traction_vector is None):
            raise ValueError(
                "fournir exactement un parmi 'force_vector' (N, force totale répartie uniformément) "
                "ou 'traction_vector' (MPa, traction uniforme)."
            )
        return self


# -----------------------------------------------------------------------------
# Contact unilatéral contre un obstacle rigide analytique : la géométrie de l'obstacle
# (PlaneObstacle/CylinderObstacle/SphereObstacle/Obstacle) vit dans preproc/obstacles.py (réexportée
# ci-dessus) ; seule la boundary elle-même (pénalité, frottement) est déclarée ici, à côté des autres
# types de boundary.
# -----------------------------------------------------------------------------
class ContactBoundary(_BoundaryBase):
    """
    Contact unilatéral (pénalité) contre un obstacle rigide analytique, avec frottement de Coulomb
    régularisé (formulation holonome, voir preproc/contact.py : valide pour un chargement monotone,
    sans décharge/inversion du glissement — limitation à lever si besoin par un historique aux points
    de quadrature, cf. DESIGN_CONTACT.md).
    """
    type: Literal["contact"]
    obstacle: Obstacle
    epsilon_n: PositiveNumber              # pénalité normale [MPa/mm] (= N/mm^3)
    epsilon_t: PositiveNumber | None = None  # pénalité tangentielle [MPa/mm] ; défaut : epsilon_n
    friction: NonNegativeNumber = 0.0      # coefficient de Coulomb (0 = contact glissant sans frottement)


Boundary = Annotated[DirichletBoundary | NeumannBoundary | ContactBoundary, Field(discriminator="type")]


class PhysicsConfig(_Section):
    gravity: Vec3 = Field(default_factory=lambda: [0.0, 0.0, 0.0])


class OutputConfig(_Section):
    fields: str | list[str] = "all"  # 'all' ou liste de champs (voir preproc/fields.py)
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
    geometry: GeometryConfig | None = None
    mesh: MeshConfig
    material: MaterialConfig
    physics: PhysicsConfig = Field(default_factory=PhysicsConfig)
    boundaries: list[Boundary]
    solver: SolverConfig
    output: OutputConfig

    @field_validator("mesh", mode="before")
    @classmethod
    def _default_mesh_source(cls, value: object) -> object:
        if isinstance(value, dict) and "source" not in value:
            return {**value, "source": DEFAULT_SOURCE}
        return value

    @model_validator(mode="after")
    def _check_cross_rules(self) -> StudyConfig:
        source = self.mesh.source
        if source == "gmsh" and self.geometry is None:
            raise ValueError("Clé manquante dans 'racine' : 'geometry'")
        if source == "exodus" and self.geometry is not None:
            raise ValueError(
                "La section 'geometry' n'est pas applicable avec mesh.source: exodus "
                "(la géométrie et le maillage proviennent du fichier .exo)."
            )

        # Matériau : présence puis validation des paramètres propres à la loi choisie (registre).
        material_class = MATERIAL_MODELS[self.material.model]
        params = self.material.params()
        missing = [key for key in material_class.REQUIRED_PARAMS if key not in params]
        if missing:
            raise ValueError(f"Clé manquante dans 'material' : '{missing[0]}'")
        material_class.validate(params)
        # Règles propres à la formulation de la loi (ordre minimal, méthode de solveur).
        material_class.check_setup(self.mesh, self.solver)

        # Catalogue des champs exportables pour la famille de loi.
        resolve_fields(self.output.fields, material_class.MIXED)

        # Garde-fou : en hyperélasticité, le terme J^(-2/3) fait exploser le degré de quadrature estimé
        # automatiquement par UFL si on ne le fixe pas explicitement (assemblage très lent, voire bloquant).
        if material_class.MIXED and self.mesh.quadrature_degree is None:
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
        for i, bnd in enumerate(self.boundaries):
            if bnd.name in names:
                raise ValueError(f"Nom de boundary dupliqué : '{bnd.name}'.")
            names.add(bnd.name)
            if isinstance(bnd, DirichletBoundary):
                n_dirichlet += 1
            if source == "gmsh" and not bnd.probe_points:
                raise ValueError(f"boundaries[{i}] ('{bnd.name}') : 'probe_points' doit être une liste non vide.")
        if n_dirichlet == 0:
            raise ValueError("Au moins une condition de Dirichlet est nécessaire (problème singulier sinon).")

        if isinstance(self.mesh, GmshMeshConfig):
            for i, ref in enumerate(self.mesh.refinements):
                if isinstance(ref, BoundaryRefinement) and ref.boundary_name not in names:
                    raise ValueError(f"mesh.refinements[{i}] : boundary '{ref.boundary_name}' introuvable.")
        return self


# Alias historique : les modules annotent leurs paramètres 'config: Config'.
Config = StudyConfig


def mesh_source(config: StudyConfig) -> str:
    """Source du maillage déclarée par mesh.source ('gmsh' par défaut), validée. Point d'accès unique."""
    return config.mesh.source


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
    if isinstance(config.mesh, ExodusMeshConfig):
        mesh_path = Path(config.mesh.file)
        if not mesh_path.is_absolute():
            mesh_path = (base_dir / mesh_path).resolve()
        # Existence vérifiée à la lecture par preproc.exodus.read_exodus (seul lecteur du fichier).
        config.mesh.file = str(mesh_path)
    else:
        if config.geometry is None:  # garanti par la validation croisée (source gmsh)
            raise RuntimeError("Configuration gmsh sans section 'geometry' après validation.")
        cad_path = Path(config.geometry.cad_file)
        if not cad_path.is_absolute():
            cad_path = (base_dir / cad_path).resolve()
        if not cad_path.exists():
            raise FileNotFoundError(f"Fichier CAO introuvable : {cad_path}")
        config.geometry.cad_file = str(cad_path)

    out_path = Path(config.output.outdir)
    if not out_path.is_absolute():
        out_path = (base_dir / out_path).resolve()
    config.output.outdir = str(out_path)
    return config
