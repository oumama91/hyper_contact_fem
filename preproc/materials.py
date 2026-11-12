"""
Utilitaires partagés par la loi néo-hookéenne : cinématique et paramètres élastiques.

Le projet ne contient qu'une seule loi de comportement (néo-hookéen mixte) : il n'y a donc plus
de registre, d'interface abstraite ni de fabrique. Ce module se limite aux fonctions réellement
utilisées par preproc/hyperelasticity.py et preproc/neo_hookean.py.
"""
from __future__ import annotations

from typing import Any

import ufl

from utils import Expr


def deformation_gradient(u: Expr) -> Expr:
    """F = I + grad(u) (configuration de référence = maillage)."""
    return ufl.Identity(u.ufl_shape[0]) + ufl.grad(u)


def lame_mu(E: float, nu: float) -> float:
    """Module de cisaillement mu = E / (2(1+nu))."""
    return E / (2.0 * (1.0 + nu))


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
