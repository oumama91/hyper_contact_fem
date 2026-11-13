"""
Contact rigide analytique : jauge signée, normale et frottement, sur des cas 0D exacts.

Les expressions UFL sont évaluées numériquement par dolfinx sur un unique tétraèdre dont les
sommets sont (0,0,0), (1,0,0), (0,1,0), (0,0,1) : son application de référence est l'identité,
donc `fem.Expression` évalue au point physique voulu, même hors du tétraèdre (extrapolation
affine exacte). Aucune solution mécanique n'est calculée ici : les valeurs attendues sont
analytiques (jauge, pression de pénalité, cône de Coulomb).
"""
from __future__ import annotations

import basix.ufl
import numpy as np
import pytest
import ufl
from dolfinx import fem
from dolfinx import mesh as dm
from mpi4py import MPI

from config import CylinderObstacle, PlaneObstacle, SphereObstacle
from contact import gap_and_normal, normal_pressure, tangential_traction

_POINTS = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
_CELLS = np.array([[0, 1, 2, 3]], dtype=np.int64)
# COMM_SELF : chaque rang voit le même maillage, les tests restent valables sous mpiexec.
_MESH_DOMAIN = ufl.Mesh(basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,)))
MESH = dm.create_mesh(MPI.COMM_SELF, _CELLS, _MESH_DOMAIN, _POINTS)


def evaluate(expr, point: tuple[float, float, float]) -> np.ndarray:
    """
    Évalue une expression UFL au point physique (cube unité 1 cellule : référence = physique).

    Une expression purement constante (normale d'un plan, traction de frottement nulle) ne porte
    aucune dépendance au maillage, donc aucun domaine UFL : `fem.Expression` refuserait de
    l'évaluer. On lui ajoute une perturbation proportionnelle aux coordonnées, d'amplitude 1e-30 :
    elle est négligeable devant les valeurs testées mais UFL ne peut pas la simplifier, donc le
    domaine du maillage est conservé.
    """
    coords = ufl.SpatialCoordinate(MESH)
    if expr.ufl_shape:
        bound = expr + 1.0e-30 * coords
    else:
        bound = expr + 1.0e-30 * coords[0]
    compiled = fem.Expression(bound, np.asarray([point], dtype=np.float64))
    values = compiled.eval(MESH, np.array([0], dtype=np.int32))
    return np.asarray(values[0, 0])


def plane() -> PlaneObstacle:
    return PlaneObstacle.model_validate(
        {"point": [0.0, 0.0, 1.8], "normal": [0.0, 0.0, -4.0]}  # normale non unitaire : normalisée en interne
    )


def test_plane_gap_sign_and_normal() -> None:
    x = ufl.SpatialCoordinate(MESH)
    gap, normal = gap_and_normal(x, plane())

    below = float(evaluate(gap, (0.3, 0.4, 2.0)))   # au-dessus du plan : pénétration
    above = float(evaluate(gap, (0.3, 0.4, 1.6)))   # sous le plan : admissible
    assert below == pytest.approx(-0.2)
    assert above == pytest.approx(0.2)
    assert evaluate(normal, (0.3, 0.4, 2.0)) == pytest.approx([0.0, 0.0, -1.0])


def test_normal_pressure_is_macaulay() -> None:
    x = ufl.SpatialCoordinate(MESH)
    gap, _ = gap_and_normal(x, plane())
    p_n = normal_pressure(gap, epsilon_n=1.0e3)
    assert float(evaluate(p_n, (0.3, 0.4, 2.0))) == pytest.approx(200.0)  # 1e3 x 0.2
    assert float(evaluate(p_n, (0.3, 0.4, 1.6))) == pytest.approx(0.0)    # hors contact


@pytest.mark.parametrize(
    "obstacle, point, expected_gap, expected_normal",
    [
        (
            CylinderObstacle.model_validate(
                {
                    "kind": "cylinder",
                    "axis_point": [0.0, 0.0, 0.0],
                    "axis_dir": [0.0, 0.0, 1.0],
                    "radius": 1.0,
                }
            ),
            (2.0, 0.0, 0.5),
            1.0,
            [1.0, 0.0, 0.0],
        ),
        (
            CylinderObstacle.model_validate(
                {
                    "kind": "cylinder",
                    "axis_point": [0.0, 0.0, 0.0],
                    "axis_dir": [0.0, 0.0, 1.0],
                    "radius": 1.0,
                    "orientation": "inside",
                }
            ),
            (2.0, 0.0, 0.5),
            -1.0,
            [-1.0, 0.0, 0.0],
        ),
        (
            SphereObstacle.model_validate({"kind": "sphere", "center": [0.0, 0.0, 0.0], "radius": 2.0}),
            (0.0, 0.0, 3.0),
            1.0,
            [0.0, 0.0, 1.0],
        ),
    ],
)
def test_curved_obstacle_gap_and_normal(obstacle, point, expected_gap, expected_normal) -> None:
    x = ufl.SpatialCoordinate(MESH)
    gap, normal = gap_and_normal(x, obstacle)
    assert float(evaluate(gap, point)) == pytest.approx(expected_gap)
    assert evaluate(normal, point) == pytest.approx(expected_normal)


def test_friction_opposes_sliding_below_cone() -> None:
    x = ufl.SpatialCoordinate(MESH)
    u = ufl.as_vector([0.01, 0.0, 0.0])
    normal = ufl.as_vector([0.0, 0.0, 1.0])
    t_t = tangential_traction(u, normal, p_n=100.0, epsilon_t=1.0e3, friction=0.3, gdim=3)
    # t_essai = -epsilon_t u_t = -10 e_x ; ||t_essai|| = 10 < mu p_N = 30 : pas de glissement
    assert evaluate(t_t, (0.5, 0.5, 0.5)) == pytest.approx([-10.0, 0.0, 0.0])


def test_friction_saturates_on_coulomb_cone() -> None:
    u = ufl.as_vector([0.1, 0.0, 0.0])
    normal = ufl.as_vector([0.0, 0.0, 1.0])
    t_t = tangential_traction(u, normal, p_n=100.0, epsilon_t=1.0e3, friction=0.3, gdim=3)
    # t_essai = -100 e_x ; ||t_essai|| = 100 > 30 : retour radial, ||t_T|| = mu p_N = 30
    assert evaluate(t_t, (0.5, 0.5, 0.5)) == pytest.approx([-30.0, 0.0, 0.0])


def test_zero_friction_gives_zero_traction() -> None:
    u = ufl.as_vector([0.1, 0.0, 0.0])
    normal = ufl.as_vector([0.0, 0.0, 1.0])
    t_t = tangential_traction(u, normal, p_n=100.0, epsilon_t=1.0e3, friction=0.0, gdim=3)
    assert evaluate(t_t, (0.5, 0.5, 0.5)) == pytest.approx([0.0, 0.0, 0.0])
