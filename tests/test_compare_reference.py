"""
Tests purs Python du comparateur de références (``tools/compare_reference.py``).

Aucun solveur ni dolfinx : les métriques sont des dictionnaires JSON construits en mémoire
et écrits dans ``tmp_path``. Le module est chargé par son chemin, sans paquet ``tools/``.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "compare_reference", ROOT / "tools" / "compare_reference.py"
)
assert _SPEC is not None and _SPEC.loader is not None
compare_reference = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(compare_reference)


def _metrics(**overrides: Any) -> dict[str, Any]:
    """Métriques minimales valides, surchargées par les cas de chaque test."""
    data: dict[str, Any] = {
        "mpi_size": 1,
        "u_max": 4.0,
        "sigma_1_max": 0.5,
        "rel_res": 1.0e-7,
        "imbalance": 1.0e-6,
        "increments": 1,
        "newton_iterations": 10,
        "cutbacks": 0,
        "line_search_evaluations": 10,
        "reactions": {"fix": [0.0, 1.0, -2.0]},
        "reaction_total": [0.0, 1.0, -2.0],
        "external": [0.0, -1.0, 2.0],
    }
    data.update(overrides)
    return data


def _run(tmp_path: Path, monkeypatch, capsys, run: dict, ref: dict) -> tuple[int, str]:
    """Écrit les deux JSON, lance ``main`` et vérifie qu'aucun traceback n'est produit."""
    run_path = tmp_path / "run.json"
    ref_path = tmp_path / "reference.json"
    run_path.write_text(json.dumps(run), encoding="utf-8")
    ref_path.write_text(json.dumps(ref), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["compare_reference.py", "--run", str(run_path), "--reference", str(ref_path)],
    )
    code = compare_reference.main()
    captured = capsys.readouterr()
    assert "Traceback" not in captured.out
    assert "Traceback" not in captured.err
    return code, captured.out


def test_zero_reference_zero_run_passes(tmp_path, monkeypatch, capsys) -> None:
    code, out = _run(tmp_path, monkeypatch, capsys, _metrics(u_max=0.0), _metrics(u_max=0.0))
    assert code == 0
    assert "VALIDATION RÉUSSIE" in out


def test_zero_reference_nonzero_run_fails_cleanly(tmp_path, monkeypatch, capsys) -> None:
    code, out = _run(tmp_path, monkeypatch, capsys, _metrics(u_max=1.0), _metrics(u_max=0.0))
    assert code == 1
    assert "VALIDATION ÉCHOUÉE" in out


def test_nan_against_nan_is_accepted(tmp_path, monkeypatch, capsys) -> None:
    code, out = _run(
        tmp_path,
        monkeypatch,
        capsys,
        _metrics(u_max=float("nan")),
        _metrics(u_max=float("nan")),
    )
    assert code == 0
    assert "VALIDATION RÉUSSIE" in out


def test_nan_against_value_fails_cleanly(tmp_path, monkeypatch, capsys) -> None:
    code, out = _run(
        tmp_path,
        monkeypatch,
        capsys,
        _metrics(u_max=float("nan")),
        _metrics(u_max=4.0),
    )
    assert code == 1
    assert "VALIDATION ÉCHOUÉE" in out


def test_infinite_value_fails_cleanly(tmp_path, monkeypatch, capsys) -> None:
    code, out = _run(
        tmp_path,
        monkeypatch,
        capsys,
        _metrics(u_max=float("inf")),
        _metrics(u_max=4.0),
    )
    assert code == 1
    assert "VALIDATION ÉCHOUÉE" in out


def test_missing_reaction_fails_cleanly(tmp_path, monkeypatch, capsys) -> None:
    code, out = _run(
        tmp_path,
        monkeypatch,
        capsys,
        _metrics(reactions={}),
        _metrics(reactions={"fix": [0.0, 1.0, -2.0]}),
    )
    assert code == 1
    assert "absente du run" in out


def test_extra_reaction_fails_cleanly(tmp_path, monkeypatch, capsys) -> None:
    code, out = _run(
        tmp_path,
        monkeypatch,
        capsys,
        _metrics(reactions={"extra": [0.0, 1.0, -2.0]}),
        _metrics(reactions={}),
    )
    assert code == 1
    assert "absente de la référence" in out


def test_reaction_length_mismatch_fails_cleanly(tmp_path, monkeypatch, capsys) -> None:
    code, out = _run(
        tmp_path,
        monkeypatch,
        capsys,
        _metrics(reactions={"fix": [0.0, 1.0]}),
        _metrics(reactions={"fix": [0.0, 1.0, -2.0]}),
    )
    assert code == 1
    assert "longueur" in out


def test_total_length_mismatch_fails_cleanly(tmp_path, monkeypatch, capsys) -> None:
    code, out = _run(
        tmp_path,
        monkeypatch,
        capsys,
        _metrics(reaction_total=[0.0, 1.0]),
        _metrics(reaction_total=[0.0, 1.0, -2.0]),
    )
    assert code == 1
    assert "longueur" in out


def test_mpi_size_mismatch_fails_cleanly(tmp_path, monkeypatch, capsys) -> None:
    code, out = _run(
        tmp_path,
        monkeypatch,
        capsys,
        _metrics(mpi_size=2),
        _metrics(mpi_size=1),
    )
    assert code == 1
    assert "mpi_size du run (2)" in out
    assert "référence (1)" in out


def test_nominal_case_passes(tmp_path, monkeypatch, capsys) -> None:
    metrics = _metrics()
    code, out = _run(tmp_path, monkeypatch, capsys, metrics, metrics)
    assert code == 0
    assert "VALIDATION RÉUSSIE" in out
