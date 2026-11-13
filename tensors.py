"""Algèbre tensorielle pure (numpy/UFL) pour le post-traitement : aucune dépendance au maillage ni à Forms."""
from __future__ import annotations

import numpy as np
import ufl

from fields import COMPONENTS
from utils import Expr


def von_mises(sigma: Expr, gdim: int) -> Expr:
    dev = sigma - ufl.tr(sigma) / 3.0 * ufl.Identity(gdim)
    return ufl.sqrt(1.5 * ufl.inner(dev, dev))


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