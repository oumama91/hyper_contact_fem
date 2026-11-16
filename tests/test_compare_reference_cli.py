"""
Caractérisation de la sortie de la CLI ``tools/compare_reference.py``.

Les scénarios partent de copies des JSON de ``reference/`` écrites dans ``tmp_path`` ; la sortie
standard complète et le code retour sont figés après observation du code d'origine. Ces tests ne
dépendent ni des chemins temporaires ni des durées et doivent rester inchangés lors du découpage
de ``main``.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reference"
_SPEC = importlib.util.spec_from_file_location(
    "compare_reference", ROOT / "tools" / "compare_reference.py"
)
assert _SPEC is not None and _SPEC.loader is not None
compare_reference = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(compare_reference)


def _copy(value: Any) -> Any:
    """Copie JSON indépendante d'une structure chargée depuis ``reference/``."""
    return json.loads(json.dumps(value))


def _base_metrics() -> dict[str, Any]:
    return json.loads(
        (REFERENCE / "O-ring_compression_series_metrics.json").read_text(encoding="utf-8")
    )


def _base_fields() -> dict[str, Any]:
    return json.loads(
        (REFERENCE / "O-ring_compression_series_fields.json").read_text(encoding="utf-8")
    )


def _run(
    tmp_path: Path,
    monkeypatch,
    capsys,
    run: dict[str, Any],
    ref: dict[str, Any],
    fields_run: dict[str, Any] | None = None,
    fields_ref: dict[str, Any] | None = None,
) -> tuple[int, str]:
    """Écrit les JSON, lance ``main`` comme la CLI et retourne (code, sortie standard)."""
    run_path = tmp_path / "run_metrics.json"
    ref_path = tmp_path / "reference_metrics.json"
    run_path.write_text(json.dumps(run), encoding="utf-8")
    ref_path.write_text(json.dumps(ref), encoding="utf-8")
    argv = ["compare_reference.py", "--run", str(run_path), "--reference", str(ref_path)]
    if fields_run is not None:
        fields_run_path = tmp_path / "run_fields.json"
        fields_ref_path = tmp_path / "reference_fields.json"
        fields_run_path.write_text(json.dumps(fields_run), encoding="utf-8")
        fields_ref_path.write_text(json.dumps(fields_ref), encoding="utf-8")
        argv += [
            "--fields-run", str(fields_run_path),
            "--fields-reference", str(fields_ref_path),
        ]
    monkeypatch.setattr(sys, "argv", argv)
    code = compare_reference.main()
    captured = capsys.readouterr()
    assert captured.err == ""
    return code, captured.out



EXPECTED_REFERENCE_COMPARED_TO_ITSELF = """\
Référence : reference_metrics.json (MPI = 1)
Run       : run_metrics.json (MPI = 1)
Tolérance : 1e-12

Grandeur                                 référence              run  écart rel.       tol.  statut
--- scalaires globaux ---
u_max                              +4.664172049663e-01  +4.664172049663e-01   0.000e+00      1e-12  OK
sigma_1_max                        +1.599963399299e-01  +1.599963399299e-01   0.000e+00      1e-12  OK
rel_res                            +4.564018773083e-07  +4.564018773083e-07   0.000e+00      1e-12  OK
imbalance                          +9.950141512499e-06  +9.950141512499e-06   0.000e+00      1e-12  OK
--- compteurs (égalité stricte) ---
increments                                       1                1           0      exact  OK
newton_iterations                               10               10           0      exact  OK
cutbacks                                         0                0           0      exact  OK
line_search_evaluations                         10               10           0      exact  OK
--- réactions [N] (référence, run, écart) ---
reactions.symmetry_y[0]            +0.000000000000e+00  +0.000000000000e+00   0.000e+00      1e-12  OK
reactions.symmetry_y[1]            +3.950921197626e-09  +3.950921197626e-09   0.000e+00      1e-12  OK
reactions.symmetry_y[2]            +0.000000000000e+00  +0.000000000000e+00   0.000e+00      1e-12  OK
reactions.symmetry_x[0]            +3.950920746381e-09  +3.950920746381e-09   0.000e+00      1e-12  OK
reactions.symmetry_x[1]            +0.000000000000e+00  +0.000000000000e+00   0.000e+00      1e-12  OK
reactions.symmetry_x[2]            +0.000000000000e+00  +0.000000000000e+00   0.000e+00      1e-12  OK
reactions.symmetry_z[0]            +0.000000000000e+00  +0.000000000000e+00   0.000e+00      1e-12  OK
reactions.symmetry_z[1]            +0.000000000000e+00  +0.000000000000e+00   0.000e+00      1e-12  OK
reactions.symmetry_z[2]            +2.480275894546e+01  +2.480275894546e+01   0.000e+00      1e-12  OK
reaction_total[0]                  +3.950920746381e-09  +3.950920746381e-09   0.000e+00      1e-12  OK
reaction_total[1]                  +3.950921197626e-09  +3.950921197626e-09   0.000e+00      1e-12  OK
reaction_total[2]                  +2.480275894546e+01  +2.480275894546e+01   0.000e+00      1e-12  OK
external[0]                        +0.000000000000e+00  +0.000000000000e+00   0.000e+00      1e-12  OK
external[1]                        +0.000000000000e+00  +0.000000000000e+00   0.000e+00      1e-12  OK
external[2]                        -2.480325253230e+01  -2.480325253230e+01   0.000e+00      1e-12  OK
--- statistiques de champs (min, max, l2 nodale) ---
fields.displacement.min            -1.988721244047e-01  -1.988721244047e-01   0.000e+00      1e-12  OK
fields.displacement.max            +4.664095424640e-01  +4.664095424640e-01   0.000e+00      1e-12  OK
fields.displacement.l2             +6.538288128404e+01  +6.538288128404e+01   0.000e+00      1e-12  OK
fields.cauchy_stress.min           -1.130996647459e+00  -1.130996647459e+00   0.000e+00      1e-12  OK
fields.cauchy_stress.max           +2.614143267162e-01  +2.614143267162e-01   0.000e+00      1e-12  OK
fields.cauchy_stress.l2            +1.357569611544e+02  +1.357569611544e+02   0.000e+00      1e-12  OK
fields.principal_stress.min        -1.131013106470e+00  -1.131013106470e+00   0.000e+00      1e-12  OK
fields.principal_stress.max        +1.599963399299e-01  +1.599963399299e-01   0.000e+00      1e-12  OK
fields.principal_stress.l2         +1.357569611544e+02  +1.357569611544e+02   0.000e+00      1e-12  OK
fields.mean_stress.min             -9.861788244298e-01  -9.861788244298e-01   0.000e+00      1e-12  OK
fields.mean_stress.max             +4.911321764148e-02  +4.911321764148e-02   0.000e+00      1e-12  OK
fields.mean_stress.l2              +7.166689331144e+01  +7.166689331144e+01   0.000e+00      1e-12  OK
fields.strain.min                  -1.081167857774e-01  -1.081167857774e-01   0.000e+00      1e-12  OK
fields.strain.max                  +1.051976911150e-01  +1.051976911150e-01   0.000e+00      1e-12  OK
fields.strain.l2                   +1.380973803409e+01  +1.380973803409e+01   0.000e+00      1e-12  OK
fields.principal_stretch.min       +8.846011758451e-01  +8.846011758451e-01   0.000e+00      1e-12  OK
fields.principal_stretch.max       +1.100535959595e+00  +1.100535959595e+00   0.000e+00      1e-12  OK
fields.principal_stretch.l2        +2.644713736107e+02  +2.644713736107e+02   0.000e+00      1e-12  OK
fields.J.min                       +9.999013821102e-01  +9.999013821102e-01   0.000e+00      1e-12  OK
fields.J.max                       +1.000004912414e+00  +1.000004912414e+00   0.000e+00      1e-12  OK
fields.J.l2                        +1.522347698310e+02  +1.522347698310e+02   0.000e+00      1e-12  OK

Plancher des réactions : 2.480e-07 N (1e-8 x 2.480325e+01 N)

VALIDATION RÉUSSIE : toutes les grandeurs sont dans la tolérance.
"""

EXPECTED_SCALAR_MISMATCH = """\
Référence : reference_metrics.json (MPI = 1)
Run       : run_metrics.json (MPI = 1)
Tolérance : 1e-12

Grandeur                                 référence              run  écart rel.       tol.  statut
--- scalaires globaux ---
u_max                              +4.664172049663e-01  +5.664172049663e-01   2.144e-01      1e-12  ÉCHEC
sigma_1_max                        +1.599963399299e-01  +1.599963399299e-01   0.000e+00      1e-12  OK
rel_res                            +4.564018773083e-07  +4.564018773083e-07   0.000e+00      1e-12  OK
imbalance                          +9.950141512499e-06  +9.950141512499e-06   0.000e+00      1e-12  OK
--- compteurs (égalité stricte) ---
increments                                       1                1           0      exact  OK
newton_iterations                               10               10           0      exact  OK
cutbacks                                         0                0           0      exact  OK
line_search_evaluations                         10               10           0      exact  OK
--- réactions [N] (référence, run, écart) ---
reactions.symmetry_y[0]            +0.000000000000e+00  +0.000000000000e+00   0.000e+00      1e-12  OK
reactions.symmetry_y[1]            +3.950921197626e-09  +3.950921197626e-09   0.000e+00      1e-12  OK
reactions.symmetry_y[2]            +0.000000000000e+00  +0.000000000000e+00   0.000e+00      1e-12  OK
reactions.symmetry_x[0]            +3.950920746381e-09  +3.950920746381e-09   0.000e+00      1e-12  OK
reactions.symmetry_x[1]            +0.000000000000e+00  +0.000000000000e+00   0.000e+00      1e-12  OK
reactions.symmetry_x[2]            +0.000000000000e+00  +0.000000000000e+00   0.000e+00      1e-12  OK
reactions.symmetry_z[0]            +0.000000000000e+00  +0.000000000000e+00   0.000e+00      1e-12  OK
reactions.symmetry_z[1]            +0.000000000000e+00  +0.000000000000e+00   0.000e+00      1e-12  OK
reactions.symmetry_z[2]            +2.480275894546e+01  +2.480275894546e+01   0.000e+00      1e-12  OK
reaction_total[0]                  +3.950920746381e-09  +3.950920746381e-09   0.000e+00      1e-12  OK
reaction_total[1]                  +3.950921197626e-09  +3.950921197626e-09   0.000e+00      1e-12  OK
reaction_total[2]                  +2.480275894546e+01  +2.480275894546e+01   0.000e+00      1e-12  OK
external[0]                        +0.000000000000e+00  +0.000000000000e+00   0.000e+00      1e-12  OK
external[1]                        +0.000000000000e+00  +0.000000000000e+00   0.000e+00      1e-12  OK
external[2]                        -2.480325253230e+01  -2.480325253230e+01   0.000e+00      1e-12  OK

Plancher des réactions : 2.480e-07 N (1e-8 x 2.480325e+01 N)

VALIDATION ÉCHOUÉE : 1 grandeur(s) hors tolérance : u_max
"""

EXPECTED_REACTION_MISMATCH = """\
Référence : reference_metrics.json (MPI = 1)
Run       : run_metrics.json (MPI = 1)
Tolérance : 1e-12

Grandeur                                 référence              run  écart rel.       tol.  statut
--- scalaires globaux ---
u_max                              +4.664172049663e-01  +4.664172049663e-01   0.000e+00      1e-12  OK
sigma_1_max                        +1.599963399299e-01  +1.599963399299e-01   0.000e+00      1e-12  OK
rel_res                            +4.564018773083e-07  +4.564018773083e-07   0.000e+00      1e-12  OK
imbalance                          +9.950141512499e-06  +9.950141512499e-06   0.000e+00      1e-12  OK
--- compteurs (égalité stricte) ---
increments                                       1                1           0      exact  OK
newton_iterations                               10               10           0      exact  OK
cutbacks                                         0                0           0      exact  OK
line_search_evaluations                         10               10           0      exact  OK
--- réactions [N] (référence, run, écart) ---
reactions.symmetry_y[0]            +0.000000000000e+00  +0.000000000000e+00   0.000e+00      1e-12  OK
reactions.symmetry_y[1]            +3.950921197626e-09  +3.950921197626e-09   0.000e+00      1e-12  OK
reactions.symmetry_y[2]            +0.000000000000e+00  +0.000000000000e+00   0.000e+00      1e-12  OK
reactions.symmetry_x[0]            +3.950920746381e-09  +3.950920746381e-09   0.000e+00      1e-12  OK
reactions.symmetry_x[1]            +0.000000000000e+00  +0.000000000000e+00   0.000e+00      1e-12  OK
reactions.symmetry_x[2]            +0.000000000000e+00  +0.000000000000e+00   0.000e+00      1e-12  OK
reactions.symmetry_z[0]            +0.000000000000e+00  +0.000000000000e+00   0.000e+00      1e-12  OK
reactions.symmetry_z[1]            +0.000000000000e+00  +0.000000000000e+00   0.000e+00      1e-12  OK
reactions.symmetry_z[2]            +2.480275894546e+01  +2.480285894546e+01   4.032e-06      1e-12  ÉCHEC
reaction_total[0]                  +3.950920746381e-09  +3.950920746381e-09   0.000e+00      1e-12  OK
reaction_total[1]                  +3.950921197626e-09  +3.950921197626e-09   0.000e+00      1e-12  OK
reaction_total[2]                  +2.480275894546e+01  +2.480275894546e+01   0.000e+00      1e-12  OK
external[0]                        +0.000000000000e+00  +0.000000000000e+00   0.000e+00      1e-12  OK
external[1]                        +0.000000000000e+00  +0.000000000000e+00   0.000e+00      1e-12  OK
external[2]                        -2.480325253230e+01  -2.480325253230e+01   0.000e+00      1e-12  OK

Plancher des réactions : 2.480e-07 N (1e-8 x 2.480325e+01 N)

VALIDATION ÉCHOUÉE : 1 grandeur(s) hors tolérance : reactions.symmetry_z[2]
"""

EXPECTED_COUNTER_MISMATCH = """\
Référence : reference_metrics.json (MPI = 1)
Run       : run_metrics.json (MPI = 1)
Tolérance : 1e-12

Grandeur                                 référence              run  écart rel.       tol.  statut
--- scalaires globaux ---
u_max                              +4.664172049663e-01  +4.664172049663e-01   0.000e+00      1e-12  OK
sigma_1_max                        +1.599963399299e-01  +1.599963399299e-01   0.000e+00      1e-12  OK
rel_res                            +4.564018773083e-07  +4.564018773083e-07   0.000e+00      1e-12  OK
imbalance                          +9.950141512499e-06  +9.950141512499e-06   0.000e+00      1e-12  OK
--- compteurs (égalité stricte) ---
increments                                       1                1           0      exact  OK
newton_iterations                               10                9           0      exact  ÉCHEC
cutbacks                                         0                0           0      exact  OK
line_search_evaluations                         10               10           0      exact  OK
--- réactions [N] (référence, run, écart) ---
reactions.symmetry_y[0]            +0.000000000000e+00  +0.000000000000e+00   0.000e+00      1e-12  OK
reactions.symmetry_y[1]            +3.950921197626e-09  +3.950921197626e-09   0.000e+00      1e-12  OK
reactions.symmetry_y[2]            +0.000000000000e+00  +0.000000000000e+00   0.000e+00      1e-12  OK
reactions.symmetry_x[0]            +3.950920746381e-09  +3.950920746381e-09   0.000e+00      1e-12  OK
reactions.symmetry_x[1]            +0.000000000000e+00  +0.000000000000e+00   0.000e+00      1e-12  OK
reactions.symmetry_x[2]            +0.000000000000e+00  +0.000000000000e+00   0.000e+00      1e-12  OK
reactions.symmetry_z[0]            +0.000000000000e+00  +0.000000000000e+00   0.000e+00      1e-12  OK
reactions.symmetry_z[1]            +0.000000000000e+00  +0.000000000000e+00   0.000e+00      1e-12  OK
reactions.symmetry_z[2]            +2.480275894546e+01  +2.480275894546e+01   0.000e+00      1e-12  OK
reaction_total[0]                  +3.950920746381e-09  +3.950920746381e-09   0.000e+00      1e-12  OK
reaction_total[1]                  +3.950921197626e-09  +3.950921197626e-09   0.000e+00      1e-12  OK
reaction_total[2]                  +2.480275894546e+01  +2.480275894546e+01   0.000e+00      1e-12  OK
external[0]                        +0.000000000000e+00  +0.000000000000e+00   0.000e+00      1e-12  OK
external[1]                        +0.000000000000e+00  +0.000000000000e+00   0.000e+00      1e-12  OK
external[2]                        -2.480325253230e+01  -2.480325253230e+01   0.000e+00      1e-12  OK

Plancher des réactions : 2.480e-07 N (1e-8 x 2.480325e+01 N)

VALIDATION ÉCHOUÉE : 1 grandeur(s) hors tolérance : newton_iterations
"""

EXPECTED_FIELD_MISMATCH = """\
Référence : reference_metrics.json (MPI = 1)
Run       : run_metrics.json (MPI = 1)
Tolérance : 1e-12

Grandeur                                 référence              run  écart rel.       tol.  statut
--- scalaires globaux ---
u_max                              +4.664172049663e-01  +4.664172049663e-01   0.000e+00      1e-12  OK
sigma_1_max                        +1.599963399299e-01  +1.599963399299e-01   0.000e+00      1e-12  OK
rel_res                            +4.564018773083e-07  +4.564018773083e-07   0.000e+00      1e-12  OK
imbalance                          +9.950141512499e-06  +9.950141512499e-06   0.000e+00      1e-12  OK
--- compteurs (égalité stricte) ---
increments                                       1                1           0      exact  OK
newton_iterations                               10               10           0      exact  OK
cutbacks                                         0                0           0      exact  OK
line_search_evaluations                         10               10           0      exact  OK
--- réactions [N] (référence, run, écart) ---
reactions.symmetry_y[0]            +0.000000000000e+00  +0.000000000000e+00   0.000e+00      1e-12  OK
reactions.symmetry_y[1]            +3.950921197626e-09  +3.950921197626e-09   0.000e+00      1e-12  OK
reactions.symmetry_y[2]            +0.000000000000e+00  +0.000000000000e+00   0.000e+00      1e-12  OK
reactions.symmetry_x[0]            +3.950920746381e-09  +3.950920746381e-09   0.000e+00      1e-12  OK
reactions.symmetry_x[1]            +0.000000000000e+00  +0.000000000000e+00   0.000e+00      1e-12  OK
reactions.symmetry_x[2]            +0.000000000000e+00  +0.000000000000e+00   0.000e+00      1e-12  OK
reactions.symmetry_z[0]            +0.000000000000e+00  +0.000000000000e+00   0.000e+00      1e-12  OK
reactions.symmetry_z[1]            +0.000000000000e+00  +0.000000000000e+00   0.000e+00      1e-12  OK
reactions.symmetry_z[2]            +2.480275894546e+01  +2.480275894546e+01   0.000e+00      1e-12  OK
reaction_total[0]                  +3.950920746381e-09  +3.950920746381e-09   0.000e+00      1e-12  OK
reaction_total[1]                  +3.950921197626e-09  +3.950921197626e-09   0.000e+00      1e-12  OK
reaction_total[2]                  +2.480275894546e+01  +2.480275894546e+01   0.000e+00      1e-12  OK
external[0]                        +0.000000000000e+00  +0.000000000000e+00   0.000e+00      1e-12  OK
external[1]                        +0.000000000000e+00  +0.000000000000e+00   0.000e+00      1e-12  OK
external[2]                        -2.480325253230e+01  -2.480325253230e+01   0.000e+00      1e-12  OK
--- statistiques de champs (min, max, l2 nodale) ---
fields.displacement.min            -1.988721244047e-01  -1.988721244047e-01   0.000e+00      1e-12  OK
fields.displacement.max            +4.664095424640e-01  +4.674095424640e-01   2.144e-03      1e-12  ÉCHEC
fields.displacement.l2             +6.538288128404e+01  +6.538288128404e+01   0.000e+00      1e-12  OK
fields.cauchy_stress.min           -1.130996647459e+00  -1.130996647459e+00   0.000e+00      1e-12  OK
fields.cauchy_stress.max           +2.614143267162e-01  +2.614143267162e-01   0.000e+00      1e-12  OK
fields.cauchy_stress.l2            +1.357569611544e+02  +1.357569611544e+02   0.000e+00      1e-12  OK
fields.principal_stress.min        -1.131013106470e+00  -1.131013106470e+00   0.000e+00      1e-12  OK
fields.principal_stress.max        +1.599963399299e-01  +1.599963399299e-01   0.000e+00      1e-12  OK
fields.principal_stress.l2         +1.357569611544e+02  +1.357569611544e+02   0.000e+00      1e-12  OK
fields.mean_stress.min             -9.861788244298e-01  -9.861788244298e-01   0.000e+00      1e-12  OK
fields.mean_stress.max             +4.911321764148e-02  +4.911321764148e-02   0.000e+00      1e-12  OK
fields.mean_stress.l2              +7.166689331144e+01  +7.166689331144e+01   0.000e+00      1e-12  OK
fields.strain.min                  -1.081167857774e-01  -1.081167857774e-01   0.000e+00      1e-12  OK
fields.strain.max                  +1.051976911150e-01  +1.051976911150e-01   0.000e+00      1e-12  OK
fields.strain.l2                   +1.380973803409e+01  +1.380973803409e+01   0.000e+00      1e-12  OK
fields.principal_stretch.min       +8.846011758451e-01  +8.846011758451e-01   0.000e+00      1e-12  OK
fields.principal_stretch.max       +1.100535959595e+00  +1.100535959595e+00   0.000e+00      1e-12  OK
fields.principal_stretch.l2        +2.644713736107e+02  +2.644713736107e+02   0.000e+00      1e-12  OK
fields.J.min                       +9.999013821102e-01  +9.999013821102e-01   0.000e+00      1e-12  OK
fields.J.max                       +1.000004912414e+00  +1.000004912414e+00   0.000e+00      1e-12  OK
fields.J.l2                        +1.522347698310e+02  +1.522347698310e+02   0.000e+00      1e-12  OK

Plancher des réactions : 2.480e-07 N (1e-8 x 2.480325e+01 N)

VALIDATION ÉCHOUÉE : 1 grandeur(s) hors tolérance : fields.displacement.max
"""

EXPECTED_MPI_SIZE_MISMATCH = """\
Référence : reference_metrics.json (MPI = 1)
Run       : run_metrics.json (MPI = 2)

VALIDATION ÉCHOUÉE : mpi_size du run (2) différent de celui de la référence (1).
"""

EXPECTED_MISSING_REACTION = """\
Référence : reference_metrics.json (MPI = 1)
Run       : run_metrics.json (MPI = 1)
Tolérance : 1e-12

Grandeur                                 référence              run  écart rel.       tol.  statut
--- scalaires globaux ---
u_max                              +4.664172049663e-01  +4.664172049663e-01   0.000e+00      1e-12  OK
sigma_1_max                        +1.599963399299e-01  +1.599963399299e-01   0.000e+00      1e-12  OK
rel_res                            +4.564018773083e-07  +4.564018773083e-07   0.000e+00      1e-12  OK
imbalance                          +9.950141512499e-06  +9.950141512499e-06   0.000e+00      1e-12  OK
--- compteurs (égalité stricte) ---
increments                                       1                1           0      exact  OK
newton_iterations                               10               10           0      exact  OK
cutbacks                                         0                0           0      exact  OK
line_search_evaluations                         10               10           0      exact  OK
--- réactions [N] (référence, run, écart) ---
reactions.symmetry_x             absente du run
reactions.symmetry_y[0]            +0.000000000000e+00  +0.000000000000e+00   0.000e+00      1e-12  OK
reactions.symmetry_y[1]            +3.950921197626e-09  +3.950921197626e-09   0.000e+00      1e-12  OK
reactions.symmetry_y[2]            +0.000000000000e+00  +0.000000000000e+00   0.000e+00      1e-12  OK
reactions.symmetry_z[0]            +0.000000000000e+00  +0.000000000000e+00   0.000e+00      1e-12  OK
reactions.symmetry_z[1]            +0.000000000000e+00  +0.000000000000e+00   0.000e+00      1e-12  OK
reactions.symmetry_z[2]            +2.480275894546e+01  +2.480275894546e+01   0.000e+00      1e-12  OK
reaction_total[0]                  +3.950920746381e-09  +3.950920746381e-09   0.000e+00      1e-12  OK
reaction_total[1]                  +3.950921197626e-09  +3.950921197626e-09   0.000e+00      1e-12  OK
reaction_total[2]                  +2.480275894546e+01  +2.480275894546e+01   0.000e+00      1e-12  OK
external[0]                        +0.000000000000e+00  +0.000000000000e+00   0.000e+00      1e-12  OK
external[1]                        +0.000000000000e+00  +0.000000000000e+00   0.000e+00      1e-12  OK
external[2]                        -2.480325253230e+01  -2.480325253230e+01   0.000e+00      1e-12  OK

Plancher des réactions : 2.480e-07 N (1e-8 x 2.480325e+01 N)

VALIDATION ÉCHOUÉE : 1 grandeur(s) hors tolérance : reactions.symmetry_x (absente du run)
"""



def test_reference_compared_to_itself(tmp_path, monkeypatch, capsys) -> None:
    """La référence série comparée à elle-même, statistiques de champs comprises."""
    run_m = _copy(_base_metrics())
    ref_m = _copy(_base_metrics())
    run_f = _copy(_base_fields())
    ref_f = _copy(_base_fields())
    code, out = _run(tmp_path, monkeypatch, capsys, run_m, ref_m, run_f, ref_f)
    assert code == 0
    assert out == EXPECTED_REFERENCE_COMPARED_TO_ITSELF


def test_scalar_mismatch(tmp_path, monkeypatch, capsys) -> None:
    """Un scalaire global hors tolérance produit un échec propre."""
    run_m = _copy(_base_metrics())
    run_m["u_max"] = 0.5664172049663108
    ref_m = _copy(_base_metrics())
    code, out = _run(tmp_path, monkeypatch, capsys, run_m, ref_m)
    assert code == 1
    assert out == EXPECTED_SCALAR_MISMATCH


def test_reaction_mismatch(tmp_path, monkeypatch, capsys) -> None:
    """Une composante de réaction hors tolérance produit un échec propre."""
    run_m = _copy(_base_metrics())
    run_m["reactions"]["symmetry_z"][2] = 24.80285894546313
    ref_m = _copy(_base_metrics())
    code, out = _run(tmp_path, monkeypatch, capsys, run_m, ref_m)
    assert code == 1
    assert out == EXPECTED_REACTION_MISMATCH


def test_counter_mismatch(tmp_path, monkeypatch, capsys) -> None:
    """Un compteur différent produit un échec propre (égalité stricte)."""
    run_m = _copy(_base_metrics())
    run_m["newton_iterations"] = 9
    ref_m = _copy(_base_metrics())
    code, out = _run(tmp_path, monkeypatch, capsys, run_m, ref_m)
    assert code == 1
    assert out == EXPECTED_COUNTER_MISMATCH


def test_field_mismatch(tmp_path, monkeypatch, capsys) -> None:
    """Une statistique de champ hors tolérance produit un échec propre."""
    run_m = _copy(_base_metrics())
    ref_m = _copy(_base_metrics())
    run_f = _copy(_base_fields())
    run_f["fields"]["displacement"]["max"] = 0.46740954246401534
    ref_f = _copy(_base_fields())
    code, out = _run(tmp_path, monkeypatch, capsys, run_m, ref_m, run_f, ref_f)
    assert code == 1
    assert out == EXPECTED_FIELD_MISMATCH


def test_mpi_size_mismatch(tmp_path, monkeypatch, capsys) -> None:
    """Une taille MPI différente arrête la comparaison avant les grandeurs."""
    run_m = _copy(_base_metrics())
    run_m["mpi_size"] = 2
    ref_m = _copy(_base_metrics())
    code, out = _run(tmp_path, monkeypatch, capsys, run_m, ref_m)
    assert code == 1
    assert out == EXPECTED_MPI_SIZE_MISMATCH


def test_missing_reaction(tmp_path, monkeypatch, capsys) -> None:
    """Une réaction présente dans la référence mais absente du run est signalée."""
    run_m = _copy(_base_metrics())
    del run_m["reactions"]["symmetry_x"]
    ref_m = _copy(_base_metrics())
    code, out = _run(tmp_path, monkeypatch, capsys, run_m, ref_m)
    assert code == 1
    assert out == EXPECTED_MISSING_REACTION
