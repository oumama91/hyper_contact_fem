"""
Loi de comportement unique du projet : néo-hookéen en formulation mixte (u, p).

Cinématique (F = I + grad u), paramètres élastiques et loi isochore/volumique sont réunis ici.

Terme volumétrique de la formulation mixte :
    Π(u, p) = ∫ ψ_iso(F) + p (J - 1) - p² / (2K) dx - travail des charges
Stationnarité en p : p = K (J - 1). Pour nu = 0.5, 1/K = 0 : incompressibilité stricte.
p est la contrainte moyenne (positive en traction), pas la pression au sens usuel.
P_iso et la tangente sont obtenus par dérivation automatique (psi_iso est fourni par la loi).
"""
from __future__ import annotations

import ufl
from dolfinx import fem

from utils import SCALAR, Expr


def deformation_gradient(u: Expr) -> Expr:
    """F = I + grad(u) (configuration de référence = maillage)."""
    return ufl.Identity(u.ufl_shape[0]) + ufl.grad(u)


def lame_mu(E: float, nu: float) -> float:
    """Module de cisaillement mu = E / (2(1+nu))."""
    return E / (2.0 * (1.0 + nu))


class NeoHookean:
    """
    Néo-hookéen, formulation isochore/volumique (même forme qu'Ansys) :
        W = mu/2 (Ibar1 - 3) + (K/2) (J - 1)^2,  Ibar1 = J^(-2/3) tr(C)
    En formulation mixte, la partie isochore mu/2 (Ibar1 - 3) reste dans la loi ; le terme
    (K/2)(J - 1)^2 est remplacé par p (J - 1) - p²/(2K), équivalent après élimination de p = K (J - 1).
    Paramètres YAML : E, nu -> mu = E / (2(1+nu)), 1/K = 3 (1 - 2 nu) / E (nu = 0.5 : 1/K = 0).
    """

    inv_K: fem.Constant | None = None  # 1/K, défini par la loi ; 0 = incompressible

    def __init__(self, domain, params: dict[str, float]):
        E, nu = params["E"], params["nu"]
        self.mu = fem.Constant(domain, SCALAR(lame_mu(E, nu)))
        self.inv_K = fem.Constant(domain, SCALAR(3.0 * (1.0 - 2.0 * nu) / E))

    def psi_iso(self, F: Expr) -> Expr:
        """Partie isochore de la densité d'énergie (par volume de référence) en fonction de F."""
        J = ufl.det(F)
        Ibar1 = J ** (-2.0 / 3.0) * ufl.tr(F.T * F)
        return 0.5 * self.mu * (Ibar1 - 3.0)

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
            raise ValueError("NeoHookean.cauchy : la contrainte moyenne 'p' est requise.")
        F = deformation_gradient(u)
        return self.piola_iso(u) * F.T / ufl.det(F) + p * ufl.Identity(u.ufl_shape[0])

    def volume_ratio(self, u: Expr) -> Expr:
        """J = det F (champ exporté)."""
        return ufl.det(deformation_gradient(u))
