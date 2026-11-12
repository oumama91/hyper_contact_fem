"""
Capture haute précision des métriques d'une étude, pour la non-régression.

Répète exactement la séquence d'appels de ``main._run`` (configuration, maillage, formes,
conditions aux limites, solveur de Newton, post-traitement) mais écrit les valeurs en
précision complète (repr JSON) au lieu de se limiter au résumé console. Les valeurs
affichées par ``main.py`` sont arrondies (7 chiffres pour |u|max, 5 pour vm_max), ce qui ne
permet pas de vérifier les tolérances 1e-12 / 1e-8 du plan de migration.

Usage :
    python tools/capture_metrics.py studies/O-ring_compression.yaml --out metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mpi4py import MPI

# Le script est lancé par son chemin : la racine du projet (celle de main.py) doit être
# dans sys.path pour importer preproc / solver / postproc quel que soit le dossier courant.
# Le dossier courant est prioritaire : il permet de pointer le code d'un autre dépôt
# (par exemple l'ancien projet) sans déplacer le script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.cwd()))

from postproc.postproc import postprocess
from postproc.reactions import ReactionHistory
from preproc.config import load_config
from preproc.dirichlet import apply_bcs
from preproc.forms import build_forms
from preproc.mesher_exodus import build_mesh
from solver.solver import solve_newton
from utils import configure_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture les métriques d'une étude en précision complète.")
    parser.add_argument("study", help="Chemin vers le fichier YAML de l'étude")
    parser.add_argument("--out", required=True, help="Chemin du fichier JSON de sortie")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    configure_logging(comm)

    config = load_config(args.study)
    mesh = build_mesh(comm, config)
    forms = build_forms(mesh.domain, mesh.facet_tags, mesh.boundary_tags, config)
    dirichlet = apply_bcs(forms, mesh.facet_tags, mesh.boundary_tags, config)

    history = ReactionHistory(comm, forms, dirichlet)
    metrics = solve_newton(comm, forms, dirichlet, config, on_increment=history)
    results = postprocess(comm, mesh.domain, forms, config)

    final = history.last
    payload = {
        "mpi_size": int(comm.size),
        "u_max": float(results.u_max),
        "vm_max": float(results.vm_max),
        "rel_res": float(metrics.rel_res),
        "increments": int(metrics.increments),
        "newton_iterations": int(metrics.newton_iterations),
        "cutbacks": int(metrics.cutbacks),
        "line_search_evaluations": int(metrics.line_search_evaluations),
        "imbalance": float(final.imbalance),
        "reactions": {name: [float(c) for c in f] for name, f in final.reactions.items()},
        "reaction_total": [float(c) for c in final.total],
        "external": [float(c) for c in final.external],
    }

    if comm.rank == 0:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
