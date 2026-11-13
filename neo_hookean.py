from __future__ import annotations

from typing import Any

import ufl
from dolfinx import fem

from hyperelasticity import HyperelasticMaterial
from materials import _check_young_poisson, lame_mu
from utils import SCALAR, Expr


class NeoHookean(HyperelasticMaterial):
    """
    Néo-hookéen, formulation isochore/volumique (même forme qu'Ansys) :
        W = mu/2 (Ibar1 - 3) + (K/2) (J - 1)^2,  Ibar1 = J^(-2/3) tr(C)
    En formulation mixte, la partie isochore mu/2 (Ibar1 - 3) reste dans la loi ; le terme
    (K/2)(J - 1)^2 est remplacé par p (J - 1) - p²/(2K), équivalent après élimination de p = K (J - 1).
    Paramètres YAML : E, nu -> mu = E / (2(1+nu)), 1/K = 3 (1 - 2 nu) / E (nu = 0.5 : 1/K = 0).
    """

    @classmethod
    def check_values(cls, params: dict[str, Any]) -> None:
        _check_young_poisson(params, nu_max_included=True)

    def __init__(self, domain, params: dict[str, Any]):
        E, nu = params["E"], params["nu"]
        self.mu = fem.Constant(domain, SCALAR(lame_mu(E, nu)))
        self.inv_K = fem.Constant(domain, SCALAR(3.0 * (1.0 - 2.0 * nu) / E))

    def psi_iso(self, F: Expr) -> Expr:
        J = ufl.det(F)
        Ibar1 = J ** (-2.0 / 3.0) * ufl.tr(F.T * F)
        return 0.5 * self.mu * (Ibar1 - 3.0)
