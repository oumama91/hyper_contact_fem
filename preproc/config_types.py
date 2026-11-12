"""
Types pydantic primitifs partagés par preproc/config.py et preproc/obstacles.py.

Extrait de preproc/config.py (refactor) pour casser la dépendance circulaire qui serait sinon
nécessaire : preproc/obstacles.py a besoin de ces types de base (Vec3, PositiveNumber, _Section) mais
NE DOIT PAS importer preproc.config (qui, lui, importe Obstacle depuis preproc.obstacles). Ce module
n'importe rien de preproc.config ni de preproc.obstacles : c'est la feuille de la dépendance.

Sémantique inchangée par rapport à l'ancienne validation manuelle (voir historique de config.py) :
  * un nombre est un int ou un float (jamais un booléen, jamais une chaîne) ;
  * un entier est un vrai int (ni booléen, ni float même à valeur entière) ;
  * les bornes (> 0, >= 1, ...) sont portées par les champs via Annotated.
"""
from __future__ import annotations

from typing import Annotated

from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict


def _to_number(value: object) -> float:
    """Nombre = int ou float, jamais un booléen ni une chaîne (le laxiste de pydantic les accepterait)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("doit être un nombre")
    return float(value)


def _to_integer(value: object) -> int:
    """Entier = vrai int (ni booléen, ni float, même à valeur entière)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("doit être un entier")
    return value


def _number_gt(bound: float):
    def check(value: float) -> float:
        if not value > bound:
            raise ValueError(f"doit être un nombre > {bound:g}")
        return value

    return check


def _number_ge(bound: float):
    def check(value: float) -> float:
        if not value >= bound:
            raise ValueError(f"doit être un nombre >= {bound:g}")
        return value

    return check


def _int_ge(bound: int):
    def check(value: int) -> int:
        if value < bound:
            raise ValueError(f"doit être un entier >= {bound}")
        return value

    return check


def _to_vec3(value: object) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("doit être une liste de 3 nombres")
    return [_to_number(component) for component in value]


Number = Annotated[float, BeforeValidator(_to_number)]
PositiveNumber = Annotated[Number, AfterValidator(_number_gt(0.0))]
NonNegativeNumber = Annotated[Number, AfterValidator(_number_ge(0.0))]
Integer = Annotated[int, BeforeValidator(_to_integer)]
IntGe0 = Annotated[Integer, AfterValidator(_int_ge(0))]
IntGe1 = Annotated[Integer, AfterValidator(_int_ge(1))]
Vec3 = Annotated[list[float], BeforeValidator(_to_vec3)]


class _Section(BaseModel):
    """Base des sections : aucune clé inconnue n'est tolérée (faute de frappe = erreur)."""
    model_config = ConfigDict(extra="forbid")
