"""
Obstacles rigides analytiques pour le contact unilatéral (contact.py), découplés de
config.py (qui ne fait plus que les référencer dans ContactBoundary.obstacle).

Trois formes, union discriminée sur 'kind' : plan, cylindre, sphère. Chacune ne décrit QUE la
géométrie de l'obstacle rigide (pas la pénalité ni le frottement, portés par ContactBoundary) ;
contact.py::gap_and_normal sait calculer jauge signée et normale pour chacune.

Pourquoi un module séparé (et non des classes dans config.py comme DirichletBoundary/ContactBoundary) :
un obstacle n'est pas un type de boundary, c'est un objet géométrique RÉFÉRENCÉ PAR une ContactBoundary
(champ 'obstacle'). Le regrouper avec le reste de la config (sections YAML, règles croisées, etc.)
aurait mélangé deux préoccupations différentes : la validation de la structure du YAML (config.py) et
la modélisation géométrique du contact (ce module, dont la portée grandira à l'étape 2 de la feuille
de route avec un obstacle NURBS réel — voir DESIGN_CONTACT.md).
"""
from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import AfterValidator, Field

from config_types import PositiveNumber, Vec3, _Section


def _check_nonzero_vec3(value: list[float]) -> list[float]:
    if math.sqrt(sum(c * c for c in value)) <= 1.0e-12:
        raise ValueError("doit être un vecteur non nul")
    return value


UnitableVec3 = Annotated[Vec3, AfterValidator(_check_nonzero_vec3)]

ORIENTATIONS = ("outside", "inside")


class PlaneObstacle(_Section):
    """Demi-espace rigide : {x : (x - point)·normal >= 0} est le côté admissible (hors matière rigide)."""
    kind: Literal["plane"] = "plane"
    point: Vec3            # un point du plan, en configuration de référence [mm]
    normal: UnitableVec3   # normale sortante (côté admissible) ; normalisée en interne, sens seul compte


class CylinderObstacle(_Section):
    """Cylindre rigide infini (axe + rayon). 'outside' : le solide est à l'extérieur (axe = matière rigide) ;
    'inside' : le solide est à l'intérieur (alésage rigide, matière rigide à l'extérieur du rayon)."""
    kind: Literal["cylinder"]
    axis_point: Vec3
    axis_dir: UnitableVec3
    radius: PositiveNumber
    orientation: Literal["outside", "inside"] = "outside"


class SphereObstacle(_Section):
    """Sphère rigide. 'outside' : le solide est à l'extérieur de la sphère ; 'inside' : cavité sphérique rigide."""
    kind: Literal["sphere"]
    center: Vec3
    radius: PositiveNumber
    orientation: Literal["outside", "inside"] = "outside"


Obstacle = Annotated[PlaneObstacle | CylinderObstacle | SphereObstacle, Field(discriminator="kind")]
