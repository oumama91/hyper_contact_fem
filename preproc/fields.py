"""
Catalogue des champs exportables vers ParaView (sans dépendance à dolfinx : utilisable par config.py).

Le YAML choisit les champs dans `output` :
    fields: all                                       # tous les champs disponibles pour la loi (défaut si absent)
    fields: [von_mises, principal_stress, strain]     # sélection ('displacement' est toujours exporté)

Chaque champ apparaît UNE fois dans le menu ParaView, avec ses composantes dans le second menu :
    scalaire   : von_mises, mean_stress, tresca, equivalent_strain, J
    vecteur    : displacement (X, Y, Z) ; principal_stress, principal_strain, principal_stretch
                 (X, Y, Z = valeur 1, 2, 3 par ordre décroissant ; la « Magnitude » n'a pas de sens)
    tenseur    : cauchy_stress, strain (9 composantes d'un tenseur symétrique 3x3)

Contraintes en MPa, positives en traction. Toutes sont des contraintes de Cauchy (vraies).

Champs utiles selon la loi de comportement :
                      élasticité linéaire        hyperélasticité
    displacement      oui                        oui
    von_mises         critère de dimensionnement  indicateur ; peu parlant sur un élastomère
    cauchy_stress     oui                        oui (contrainte vraie)
    principal_stress  oui (rupture fragile, fatigue)   oui (rupture des élastomères)
    tresca            oui (cisaillement max)     rarement
    mean_stress       triaxialité, rupture ductile     oui : = p ; pression de contact, vérif. d'incompressibilité
    strain            eps = sym(grad u)          Green-Lagrange E = (F^T F - I)/2
    principal_strain  jauges de déformation      valeurs principales de E
    equivalent_strain début de plasticité        rarement
    principal_stretch -                          oui : élongations principales lambda_i = sqrt(1 + 2 E_i)
    J                 -  (J = 1 + tr eps ~ 1)    oui : changement de volume, ~ 1 si quasi incompressible
Les champs « - » n'existent pas pour la loi concernée : les demander explicitement est une erreur,
`all` les écarte. Les composantes de cisaillement sont tensorielles (pas la distorsion d'ingénieur).
"""
from __future__ import annotations

# (suffixe, i, j) des 6 composantes indépendantes d'un tenseur symétrique 3x3 (colonnes internes du post-traitement)
COMPONENTS = (("xx", 0, 0), ("yy", 1, 1), ("zz", 2, 2), ("xy", 0, 1), ("yz", 1, 2), ("xz", 0, 2))

# Champs disponibles pour toutes les lois, dans l'ordre d'export
COMMON_FIELDS = (
    "displacement", "von_mises", "cauchy_stress", "principal_stress", "mean_stress", "tresca",
    "strain", "principal_strain", "equivalent_strain",
)
# Champs qui n'ont de sens qu'en grandes déformations (hyperélasticité)
HYPERELASTIC_FIELDS = ("principal_stretch", "J")


def available_fields(hyperelastic: bool) -> tuple[str, ...]:
    """Champs exportables pour la famille de loi (hyperélastique ou élasticité linéaire)."""
    return COMMON_FIELDS + (HYPERELASTIC_FIELDS if hyperelastic else ())


def resolve_fields(spec: str | list[str] | None, hyperelastic: bool) -> tuple[str, ...]:
    """
    Convertit output.fields ('all', absent/None, ou liste de noms) en tuple de champs, dans l'ordre du catalogue.
    'displacement' est toujours inclus. Lève ValueError pour une spécification invalide ou un champ
    indisponible pour la loi.
    """
    available = available_fields(hyperelastic)
    if spec is None or spec == "all":
        return available
    if not (isinstance(spec, list) and spec and all(isinstance(f, str) for f in spec)):
        raise ValueError(
            f"output.fields doit valoir 'all' ou une liste non vide de champs parmi {list(available)} (reçu : {spec!r})."
        )
    known = COMMON_FIELDS + HYPERELASTIC_FIELDS
    unknown = [f for f in spec if f not in known]
    if unknown:
        raise ValueError(f"output.fields : champs inconnus {unknown} (disponibles : {list(available)}).")
    unavailable = [f for f in spec if f not in available]
    if unavailable:
        raise ValueError(
            f"output.fields : {unavailable} n'existe(nt) pas pour une loi en élasticité linéaire "
            f"(réservé à l'hyperélasticité). Disponibles : {list(available)}."
        )
    wanted = set(spec) | {"displacement"}
    return tuple(f for f in available if f in wanted)