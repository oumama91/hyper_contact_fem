"""
Comparaison d'un run aux références de non-régression de l'Étape 1.

Entrées : deux fichiers JSON de métriques produites par ``tools/capture_metrics.py``
(une référence et le run à valider). Affiche les écarts relatifs et vérifie les tolérances
du plan de migration :

    * série  : 1e-12 sur les réactions, |u|max, vm_max, et égalité des compteurs ;
    * np = 2 : 1e-8 (l'ordre des réductions MPI n'est pas garanti bit à bit).

Les réactions ont des composantes exactement nulles par symétrie et d'autres de l'ordre de
1e-9 (bruit numérique) : un écart purement relatif y serait dénué de sens. On normalise donc
chaque composante par ``max(|référence|, plancher)`` avec
``plancher = 1e-8 * max|réaction de référence|`` (échelle globale du chargement). Un écart
absolu de 1e-16 sur une composante nulle est ainsi vu comme négligeable, alors qu'un vrai
écart sur la réaction portante reste pénalisé.

Usage :
    python tools/compare_reference.py --run metrics.json --reference reference/....json
    python tools/compare_reference.py --run metrics.json --reference reference/....json --tol 1e-8
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

COUNTERS = ("increments", "newton_iterations", "cutbacks", "line_search_evaluations")
SCALARS = ("u_max", "vm_max", "rel_res", "imbalance")


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _rel_error(run: float, ref: float, floor: float = 0.0) -> float:
    """Écart relatif |run - ref| / max(|ref|, plancher), 0 si les deux sont identiques."""
    diff = abs(float(run) - float(ref))
    if diff == 0.0:
        return 0.0
    return diff / max(abs(float(ref)), floor)


def _fmt(value: float) -> str:
    return f"{value:+.12e}"


def _row(name: str, run: float, ref: float, error: float, tol: float) -> tuple[str, bool]:
    ok = (error <= tol) or (math.isnan(error) and math.isnan(run) and math.isnan(ref))
    status = "OK" if ok else "ÉCHEC"
    return f"{name:<34} {_fmt(ref)}  {_fmt(run)}  {error:>10.3e}  {tol:>9.0e}  {status}", ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare un run aux références JSON de l'Étape 1.")
    parser.add_argument("--run", required=True, help="Métriques du run à valider (JSON)")
    parser.add_argument("--reference", required=True, help="Métriques de référence (JSON)")
    parser.add_argument("--tol", type=float, default=None, help="Tolérance (défaut : 1e-12 si MPI=1, sinon 1e-8)")
    args = parser.parse_args()

    run = _load(args.run)
    ref = _load(args.reference)
    tol = args.tol if args.tol is not None else (1.0e-12 if int(ref["mpi_size"]) == 1 else 1.0e-8)

    # Échelle des réactions : plus grande composante de référence (réactions + total + forces extérieures).
    scale = 0.0
    for group in ("reactions", "reaction_total", "external"):
        value = ref[group]
        components = [c for vec in value.values() for c in vec] if isinstance(value, dict) else list(value)
        scale = max(scale, max((abs(float(c)) for c in components), default=0.0))
    floor = 1.0e-8 * scale if scale > 0.0 else 1.0

    rows: list[str] = []
    failures: list[str] = []

    print(f"Référence : {Path(args.reference).name} (MPI = {ref['mpi_size']})")
    print(f"Run       : {Path(args.run).name} (MPI = {run.get('mpi_size', '?')})")
    print(f"Tolérance : {tol:.0e}\n")
    print(f"{'Grandeur':<34} {'référence':>15}  {'run':>15}  {'écart rel.':>10}  {'tol.':>9}  statut")

    rows.append("--- scalaires globaux ---")
    for key in SCALARS:
        line, ok = _row(key, run[key], ref[key], _rel_error(run[key], ref[key]), tol)
        rows.append(line)
        if not ok:
            failures.append(key)

    rows.append("--- compteurs (égalité stricte) ---")
    for key in COUNTERS:
        same = int(run[key]) == int(ref[key])
        rows.append(f"{key:<34} {int(ref[key]):>15d}  {int(run[key]):>15d}  {'0':>10}  {'exact':>9}  {'OK' if same else 'ÉCHEC'}")
        if not same:
            failures.append(key)

    rows.append("--- réactions [N] (référence, run, écart) ---")
    for name, vec_ref in ref["reactions"].items():
        for comp, (a, b) in enumerate(zip(vec_ref, run["reactions"][name], strict=True)):
            label = f"reactions.{name}[{comp}]"
            error = _rel_error(b, a, floor)
            line, ok = _row(label, b, a, error, tol)
            rows.append(line)
            if not ok:
                failures.append(label)
    for key in ("reaction_total", "external"):
        for comp, (a, b) in enumerate(zip(ref[key], run[key], strict=True)):
            label = f"{key}[{comp}]"
            error = _rel_error(b, a, floor)
            line, ok = _row(label, b, a, error, tol)
            rows.append(line)
            if not ok:
                failures.append(label)

    print("\n".join(rows))
    print(f"\nPlancher des réactions : {floor:.3e} N (1e-8 x {scale:.6e} N)")
    if failures:
        print(f"\nVALIDATION ÉCHOUÉE : {len(failures)} grandeur(s) hors tolérance : {', '.join(failures)}")
        return 1
    print("\nVALIDATION RÉUSSIE : toutes les grandeurs sont dans la tolérance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
