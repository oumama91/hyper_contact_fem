"""
Contact unilatéral (pénalité) contre un obstacle rigide analytique, avec frottement de Coulomb.

Portée de ce premier jalon (voir DESIGN_CONTACT.md) :
  * obstacle RIGIDE analytique (plan / cylindre / sphère) : pas de deuxième champ mécanique, juste
    une jauge géométrique g_N(x) et une normale n(x) évaluées symboliquement en position courante
    x = X + u, X = coordonnée de référence (spatiale) ;
  * normale : pénalité pure (Alart-Curnier sans multiplicateur persistant) via un crochet de Macaulay
    <.> = max(0, .), non lisse mais Newton-différentiable presque partout (dérivée au sens des sous-
    gradients) : p_N = epsilon_n * <-g_N> ;
  * frottement : régularisation de Coulomb HOLONOME (non incrémentale). Le glissement tangentiel est
    mesuré par rapport à la configuration de référence (u_t = projection tangentielle du déplacement
    TOTAL, pas d'une variable d'historique aux points de quadrature) :
        t_essai = -epsilon_t * u_t                             (ressort de rappel : s'oppose au glissement)
        t_T     = t_essai * min(1, mu * p_N / ||t_essai||)     (retour radial sur le cône de Coulomb)
    C'est l'approximation standard valable pour un chargement MONOTONE sans inversion du sens de
    glissement (le cas d'usage visé ici : solve_newton ne pilote qu'un unique paramètre de charge
    proportionnel lambda in [0, 1]). Pour un chargement cyclique ou multi-séquence (ex. précharge
    normale puis cisaillement, ou glissement alterné), il faut une vraie plasticité de contact avec
    retour radial INCRÉMENTAL et une variable d'état persistante (traction tangentielle au pas
    précédent) stockée à chaque point de quadrature — DESIGN_CONTACT.md détaille l'extension
    (nécessite un espace de fonctions 'Quadrature' et un hook on_increment, à la ReactionHistory).

Le terme de contact est ajouté à la forme résiduelle au même titre qu'un terme interne : comme il
dépend de uh via g_N et u_t, sa tangente est obtenue gratuitement par ufl.derivative (voir forms.py),
exactement comme pour la loi de comportement. Point important : CONTRAIREMENT aux charges de Neumann,
le terme de contact n'est PAS multiplié par le facteur de charge lambda (voir forms.py) : c'est une
réaction de contrainte qui doit être pleinement active dès que g_N < 0, quel que soit lambda.
"""
from __future__ import annotations

import math

import ufl

from obstacles import CylinderObstacle, Obstacle, PlaneObstacle, SphereObstacle
from utils import Expr

# Régularisation de la norme du vecteur tangentiel essai (évite une division par zéro exacte en 0 ;
# l'échelle est fixée par epsilon_t de sorte que le terme soit négligeable dès qu'il y a un glissement
# mesurable, et n'introduise pas de raideur résiduelle artificielle en l'absence de contact).
_NORM_REG_RELATIVE = 1.0e-9


def _unit(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(c * c for c in vec))
    return [c / norm for c in vec]


def _orientation_sign(orientation: str) -> float:
    return 1.0 if orientation == "outside" else -1.0


def gap_and_normal(x: Expr, obstacle: Obstacle) -> tuple[Expr, Expr]:
    """
    Jauge signée g_N(x) et normale sortante n(x) (direction admissible) d'un obstacle analytique,
    en position courante x (expression UFL, typiquement X + déplacement).

    Convention commune aux trois obstacles : g_N > 0 hors contact (admissible), g_N < 0 en
    pénétration, n(x) pointe du côté admissible (loin de la matière rigide).
    """
    if isinstance(obstacle, PlaneObstacle):
        n0 = ufl.as_vector(_unit(obstacle.normal))
        x0 = ufl.as_vector(obstacle.point)
        g = ufl.inner(n0, x - x0)
        return g, n0

    if isinstance(obstacle, CylinderObstacle):
        a = ufl.as_vector(_unit(obstacle.axis_dir))
        x0 = ufl.as_vector(obstacle.axis_point)
        sign = _orientation_sign(obstacle.orientation)
        d = x - x0
        d_perp = d - ufl.inner(d, a) * a
        r = ufl.sqrt(ufl.inner(d_perp, d_perp))
        n = sign * d_perp / r
        g = sign * (r - obstacle.radius)
        return g, n

    if isinstance(obstacle, SphereObstacle):
        c = ufl.as_vector(obstacle.center)
        sign = _orientation_sign(obstacle.orientation)
        d = x - c
        r = ufl.sqrt(ufl.inner(d, d))
        n = sign * d / r
        g = sign * (r - obstacle.radius)
        return g, n

    raise TypeError(f"Obstacle non supporté : {type(obstacle)!r}")  # pragma: no cover - garde-fou


def normal_pressure(gap: Expr, epsilon_n: float) -> Expr:
    """p_N = epsilon_n * <-g_N> (crochet de Macaulay : pénétration seulement, p_N >= 0)."""
    return epsilon_n * ufl.conditional(ufl.lt(gap, 0.0), -gap, 0.0)


def tangential_traction(u: Expr, normal: Expr, p_n: Expr, epsilon_t: float, friction: float, gdim: int) -> Expr:
    """
    t_T : frottement de Coulomb régularisé, holonome (voir docstring du module pour la portée exacte).
    Retourne le vecteur nul si friction == 0.0 (pas de terme ajouté au résidu dans ce cas, voir forms.py).
    """
    if friction <= 0.0:
        return ufl.as_vector([0.0] * gdim)

    P_t = ufl.Identity(gdim) - ufl.outer(normal, normal)
    u_t = P_t * u
    # Ressort de rappel tangentiel : la traction d'essai s'OPPOSE au glissement u_t (signe négatif).
    # Sans ce signe, le "frottement" pousserait le solide dans le sens du glissement au lieu de le
    # freiner (vérifié par tests/test_contact_geometry.py sur un cas 0D).
    t_trial = -epsilon_t * u_t
    reg = _NORM_REG_RELATIVE * epsilon_t
    norm_trial = ufl.sqrt(ufl.inner(t_trial, t_trial) + reg * reg)
    cap = friction * p_n
    factor = ufl.conditional(ufl.gt(norm_trial, cap), cap / norm_trial, 1.0)
    return factor * t_trial


def contact_traction(u: Expr, x: Expr, bnd, gdim: int) -> Expr:
    """
    Traction de contact t_c = p_N(g_N) * n + t_T, exercée par l'obstacle rigide SUR le solide
    (direction n : pousse le solide hors de la matière rigide). 'bnd' : config.ContactBoundary.
    """
    gap, normal = gap_and_normal(x, bnd.obstacle)
    p_n = normal_pressure(gap, bnd.epsilon_n)
    epsilon_t = bnd.epsilon_t if bnd.epsilon_t is not None else bnd.epsilon_n
    t_t = tangential_traction(u, normal, p_n, epsilon_t, bnd.friction, gdim)
    return p_n * normal + t_t
