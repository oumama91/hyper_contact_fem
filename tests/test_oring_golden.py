"""
Test « golden » du cas O-ring : rejoue l'étude et la compare aux références de `reference/`.

Ce test exécute un calcul complet (environ 1 min 30 en série) puis `tools/compare_reference.py`,
qui vérifie les métriques (u_max, sigma_1_max, rel_res, imbalance, réactions, compteurs) et les
statistiques des champs exportés aux tolérances de référence (1e-12 en série).

Le run se fait dans un sous-processus : il reproduit exactement la commande documentée dans
reference/README.md, sans contaminer l'état MPI du processus pytest.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reference" / "O-ring_compression_series"


def test_oring_matches_reference(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.json"
    fields = tmp_path / "fields.json"

    run = subprocess.run(
        [
            sys.executable,
            "tools/capture_metrics.py",
            "studies/O-ring_compression.yaml",
            "--out",
            str(metrics),
            "--fields-out",
            str(fields),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, f"Échec du run :\n{run.stdout}\n{run.stderr}"

    comparison = subprocess.run(
        [
            sys.executable,
            "tools/compare_reference.py",
            "--run",
            str(metrics),
            "--reference",
            f"{REFERENCE}_metrics.json",
            "--fields-run",
            str(fields),
            "--fields-reference",
            f"{REFERENCE}_fields.json",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert comparison.returncode == 0, f"Écarts à la référence :\n{comparison.stdout}\n{comparison.stderr}"
    assert "VALIDATION RÉUSSIE" in comparison.stdout
