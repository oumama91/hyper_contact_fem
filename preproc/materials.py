"""
Lois de comportement. La loi est choisie dans le YAML (material.model).

Deux familles, qui déterminent la formulation (attribut de classe MIXED) :
  * Élasticité linéaire (MIXED = False) : voir preproc/linear_elastic.py
  * Hyperélasticité (MIXED = True) : voir preproc/hyperelasticity.py et preproc/neo_hookean.py

Interface commune (classe abstraite Material), utilisée par forms.py sans test sur la famille :
    internal_work(uh, w, dx) -> forme UFL du travail virtuel interne (uh : inconnue, w : fonction test)
    cauchy(u, p=None)        -> contraintes de Cauchy (p ignoré en déplacement pur)
    strain(u)                -> déformation exportée

Les règles propres à une famille (ordre minimal, méthode de solveur) vivent dans check_setup().

Ajouter une loi : dériver Material (ou HyperelasticMaterial pour une loi hyperélastique, voir
preproc/hyperelasticity.py), déclarer REQUIRED_PARAMS, implémenter l'interface, et décorer la classe
avec @register_material("nom_dans_le_yaml") (voir preproc/linear_elastic.py et preproc/neo_hookean.py).
preproc/__init__.py importe les modules de lois concrètes au chargement du package : c'est ce qui
déclenche l'enregistrement avant toute utilisation de MATERIAL_MODELS, sans ordre d'import à respecter
ni cycle réel (chaque loi concrète ne dépend que de preproc.materials / preproc.hyperelasticity).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol

import ufl
from dolfinx import fem

from preproc.validators import is_number
from utils import Expr

DEFAULT_MODEL = "linear_elastic"

# Registre des lois disponibles : nom YAML (material.model) -> classe. Peuplé par @register_material.
MATERIAL_MODELS: dict[str, type[Material]] = {}


class MeshSetup(Protocol):
    """Vue structurelle minimale de la section mesh lue par Material.check_setup.

    Un Protocol plutôt que l'import de preproc.config : la validation croisée est appelée
    par config.py, qui importe déjà ce module — l'import inverse créerait un cycle.
    """
    order: int


class SolverSetup(Protocol):
    """Vue structurelle minimale de la section solver lue par Material.check_setup."""
    method: str


def register_material(name: str):
    """Décorateur : enregistre une classe de loi de comportement sous 'name' (material.model du YAML)."""

    def wrap(cls: type[Material]) -> type[Material]:
        MATERIAL_MODELS[name] = cls
        return cls

    return wrap


def deformation_gradient(u: Expr) -> Expr:
    """F = I + grad(u) (configuration de référence = maillage)."""
    return ufl.Identity(u.ufl_shape[0]) + ufl.grad(u)


def lame_mu(E: float, nu: float) -> float:
    """Module de cisaillement mu = E / (2(1+nu)) (commun à l'élasticité linéaire et au néo-hookéen)."""
    return E / (2.0 * (1.0 + nu))


class Material(ABC):
    REQUIRED_PARAMS: tuple[str, ...] = ()
    MIXED: bool = False  # True : formulation mixte u-p (hyperélasticité)

    @classmethod
    def validate(cls, params: dict[str, Any]) -> None:
        """Valide les paramètres du bloc 'material' (types, bornes, clés inconnues)."""
        allowed = {"model", "rho", *cls.REQUIRED_PARAMS}
        unknown = set(params) - allowed
        if unknown:
            raise ValueError(
                f"material : paramètres inconnus pour '{params['model']}' : {sorted(unknown)} "
                f"(attendus : {sorted(allowed)})."
            )
        for key in ("rho", *cls.REQUIRED_PARAMS):
            if not is_number(params[key]):
                raise ValueError(f"material.{key} doit être un nombre (reçu : {params[key]!r}).")
        if params["rho"] < 0:
            raise ValueError("material.rho doit être >= 0.")
        cls.check_values(params)

    @classmethod
    def check_values(cls, params: dict[str, Any]) -> None:
        """Bornes spécifiques à la loi (à surcharger)."""

    @classmethod
    def check_setup(cls, mesh: MeshSetup, solver: SolverSetup) -> None:
        """Compatibilité de la loi avec le maillage et le solveur (à surcharger ; défaut : aucune contrainte)."""

    @abstractmethod
    def internal_work(self, uh: fem.Function, w: Expr, dx: ufl.Measure) -> Expr:
        """Travail virtuel interne R_int(uh; w) (forme UFL linéaire en la fonction test w)."""

    @abstractmethod
    def cauchy(self, u: Expr, p: Expr | None = None) -> Expr:
        """Contraintes de Cauchy. 'p' (contrainte moyenne) n'est utilisé que par les lois mixtes."""

    @abstractmethod
    def strain(self, u: Expr) -> Expr:
        """Tenseur des déformations exporté : eps (petites déformations) ou Green-Lagrange E (grandes)."""

    def volume_ratio(self, u: Expr) -> Expr:
        """J = det F (champ exporté en hyperélasticité seulement : voir fields.py)."""
        return ufl.det(deformation_gradient(u))


def _check_young_poisson(params: dict[str, Any], nu_max_included: bool) -> None:
    """E > 0 ; nu dans ]-1, 0.5[ (déplacement pur) ou ]-1, 0.5] (formulation mixte : nu = 0.5 admis)."""
    if params["E"] <= 0:
        raise ValueError("material.E doit être > 0.")
    nu = params["nu"]
    if nu_max_included:
        if not (-1.0 < nu <= 0.5):
            raise ValueError("material.nu doit être dans ]-1, 0.5] (0.5 : incompressible).")
    elif not (-1.0 < nu < 0.5):
        raise ValueError("material.nu doit être dans ]-1, 0.5[ (formulation en déplacement : 0.5 est singulier).")


def build_material(domain, params: dict[str, Any]) -> Material:
    return MATERIAL_MODELS[params["model"]](domain, params)