"""
Hyperélasticité : formulation mixte u-p, seule formulation du projet.

Le terme volumétrique est porté par cette classe de base, pas par la loi concrète (voir
neo_hookean.py) :
    Π(u, p) = ∫ ψ_iso(F) + p (J - 1) - p² / (2K) dx - travail des charges
Stationnarité en p : p = K (J - 1). Pour nu = 0.5, 1/K = 0 : incompressibilité stricte.
p est la contrainte moyenne (positive en traction), pas la pression au sens usuel.
P_iso et la tangente sont obtenus par dérivation automatique (psi_iso est fourni par la loi
concrète).
"""
from __future__ import annotations

import ufl
from dolfinx import fem

from materials import deformation_gradient
from utils import Expr


class HyperelasticMaterial:
    """Base d'une loi hyperélastique en formulation mixte (u, p)."""
    inv_K: fem.Constant | None = None  # 1/K, défini par la loi ; 0 = incompressible

    def psi_iso(self, F: Expr) -> Expr:
        """Partie isochore de la densité d'énergie (par volume de référence) en fonction de F."""
        raise NotImplementedError

    def potential(self, u: Expr, p: Expr) -> Expr:
        """Densité d'énergie de la formulation mixte : ψ_iso(F) + p (J - 1) - p² / (2K)."""
        F = deformation_gradient(u)
        J = ufl.det(F)
        return self.psi_iso(F) + p * (J - 1.0) - 0.5 * self.inv_K * p ** 2

    def internal_work(self, uh: fem.Function, w: Expr, dx: ufl.Measure) -> Expr:
        u, p = ufl.split(uh)
        return ufl.derivative(self.potential(u, p) * dx, uh, w)

    def piola_iso(self, u: Expr) -> Expr:
        F = ufl.variable(deformation_gradient(u))
        return ufl.diff(self.psi_iso(F), F)

    def strain(self, u: Expr) -> Expr:
        """Déformation de Green-Lagrange E = (F^T F - I) / 2."""
        F = deformation_gradient(u)
        return 0.5 * (F.T * F - ufl.Identity(u.ufl_shape[0]))

    def cauchy(self, u: Expr, p: Expr | None = None) -> Expr:
        """σ = (1/J) P_iso F^T + p I (P = P_iso + p J F^-T). 'p' est obligatoire (formulation mixte)."""
        if p is None:
            raise ValueError("HyperelasticMaterial.cauchy : la contrainte moyenne 'p' est requise.")
        F = deformation_gradient(u)
        return self.piola_iso(u) * F.T / ufl.det(F) + p * ufl.Identity(u.ufl_shape[0])

    def volume_ratio(self, u: Expr) -> Expr:
        """J = det F (champ exporté)."""
        return ufl.det(deformation_gradient(u))
