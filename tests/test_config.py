"""
Validation du schéma YAML : clés inconnues, bornes matériau,
ordre de maillage et degré de quadrature.

Chaque test part d'une configuration minimale valide et n'en modifie qu'un point, de sorte
qu'un échec désigne sans ambiguïté la règle testée. Les messages d'erreur de pydantic ne sont
jamais comparés : seule l'acceptation ou le rejet est vérifié.
"""
from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from hyper_contact_fem.config import parse_config

BASE = {
    "study_name": "test",
    "mesh": {"file": "mesh/O-ring.exo", "order": 2, "quadrature_degree": 4},
    "material": {"E": 6.0, "nu": 0.4999, "rho": 1.2e-9},
    "boundaries": [
        {"name": "fix", "type": "dirichlet", "components": [0], "value": [0.0]},
        {
            "name": "press",
            "type": "contact",
            "obstacle": {"kind": "plane", "point": [0.0, 0.0, 1.8], "normal": [0.0, 0.0, -1.0]},
            "epsilon_n": 1.0e3,
        },
    ],
    "solver": {"newton": {"increments": 1}},
    "output": {"outdir": "results"},
}


def make(**overrides) -> dict:
    """Copie profonde de BASE avec surcharge de sections entières (mesh=..., material=...)."""
    raw = copy.deepcopy(BASE)
    raw.update(overrides)
    return raw


def test_valid_config_and_defaults() -> None:
    config = parse_config(make())
    assert config.mesh.order == 2
    assert config.mesh.quadrature_degree == 4
    assert config.material.rho == pytest.approx(1.2e-9)
    assert config.output.fields == "all"
    assert config.solver.newton.max_cutbacks == 8


def test_unknown_key_rejected() -> None:
    with pytest.raises(ValidationError):
        parse_config(make(study_nam="faute de frappe"))


def test_unknown_key_in_section_rejected() -> None:
    raw = make()
    raw["material"]["young"] = 6.0
    with pytest.raises(ValidationError):
        parse_config(raw)


@pytest.mark.parametrize(
    "section, key, value",
    [
        ("mesh", "source", "exodus"),
        ("material", "model", "neo_hookean"),
        ("solver", "rtol", 1.0e-8),
        ("solver", "method", "mumps"),
    ],
)
def test_removed_keys_rejected(section: str, key: str, value: object) -> None:
    raw = make()
    raw[section][key] = value
    with pytest.raises(ValidationError):
        parse_config(raw)


@pytest.mark.parametrize("young", [0.0, -1.0])
def test_young_modulus_must_be_positive(young: float) -> None:
    raw = make()
    raw["material"]["E"] = young
    with pytest.raises(ValidationError):
        parse_config(raw)


@pytest.mark.parametrize("poisson", [-1.0, -1.5, 0.5001, 0.6])
def test_poisson_ratio_out_of_range_rejected(poisson: float) -> None:
    raw = make()
    raw["material"]["nu"] = poisson
    with pytest.raises(ValidationError):
        parse_config(raw)


def test_poisson_ratio_half_accepted() -> None:
    raw = make()
    raw["material"]["nu"] = 0.5
    assert parse_config(raw).material.nu == 0.5


@pytest.mark.parametrize("order", [1, 0, -2])
def test_mesh_order_below_two_rejected(order: int) -> None:
    raw = make()
    raw["mesh"]["order"] = order
    with pytest.raises(ValidationError):
        parse_config(raw)


def test_non_integer_order_rejected() -> None:
    raw = make()
    raw["mesh"]["order"] = 2.0
    with pytest.raises(ValidationError):
        parse_config(raw)


def test_quadrature_degree_defaults_and_rejects_none() -> None:
    raw = make()
    del raw["mesh"]["quadrature_degree"]
    assert parse_config(raw).mesh.quadrature_degree == 4

    raw["mesh"]["quadrature_degree"] = None
    with pytest.raises(ValidationError):
        parse_config(raw)


def test_negative_density_rejected() -> None:
    raw = make()
    raw["material"]["rho"] = -1.0e-9
    with pytest.raises(ValidationError):
        parse_config(raw)


@pytest.mark.parametrize(
    "components",
    [[0, 0], [], [3], [0, 1, 2, 0]],
)
def test_invalid_dirichlet_components_rejected(components: list[int]) -> None:
    raw = make()
    raw["boundaries"][0]["components"] = components
    raw["boundaries"][0]["value"] = [0.0] * len(components)
    with pytest.raises(ValidationError):
        parse_config(raw)


def test_mismatched_dirichlet_values_rejected() -> None:
    raw = make()
    raw["boundaries"][0]["components"] = [0, 1]
    with pytest.raises(ValidationError):
        parse_config(raw)


def test_zero_obstacle_normal_rejected() -> None:
    raw = make()
    raw["boundaries"][1]["obstacle"]["normal"] = [0.0, 0.0, 0.0]
    with pytest.raises(ValidationError):
        parse_config(raw)


def test_unknown_obstacle_kind_rejected() -> None:
    raw = make()
    raw["boundaries"][1]["obstacle"] = {"kind": "torus", "radius": 1.0}
    with pytest.raises(ValidationError):
        parse_config(raw)
