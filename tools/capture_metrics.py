"""
Capture haute précision des métriques et des champs d'une étude, pour la non-régression.

Répète exactement la séquence d'appels de ``main._run`` (configuration, maillage, formes,
conditions aux limites, solveur de Newton, post-traitement) mais écrit les valeurs en
précision complète (repr JSON) au lieu de se limiter au résumé console. Les valeurs
affichées par ``main.py`` sont arrondies (7 chiffres pour |u|max, 5 pour sigma_1_max), ce qui ne
permet pas de vérifier les tolérances 1e-12 / 1e-8.

Statistiques de champs : pour chaque champ exporté, ``--fields-out`` enregistre
le minimum, le maximum et la norme l2 des valeurs nodales, réduits sur tous les rangs.
``adios2`` n'est pas installé dans ``fenicsx-env`` et ``dolfinx`` 0.11 n'expose pas de
lecteur VTX : les valeurs sont donc capturées AU MOMENT de l'export, en enveloppant
``postproc.write_vtx`` (les valeurs nodales écrites dans le .bp sont copiées, pas
modifiées). Cette instrumentation ne change ni les calculs ni les fichiers de sortie.

Conventions des statistiques : min/max portent sur toutes les composantes nodales du champ
(composantes confondues, donc sans signification géométrique pour un vecteur ou un tenseur),
la norme l2 est ``sqrt(somme des valeurs²)`` sur les nœuds possédés ; les nœuds fantômes sont
exclus pour ne pas compter deux fois.

Usage :
    python tools/capture_metrics.py studies/O-ring_compression.yaml --out metrics.json
    python tools/capture_metrics.py studies/O-ring_compression.yaml --out metrics.json \
        --fields-out fields.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI

# Le script est lancé par son chemin : la racine du projet (celle de main.py) doit être
# dans sys.path pour importer les modules du projet quel que soit le dossier courant.
# Le dossier courant reste prioritaire dans sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.cwd()))

import postproc as pp  # noqa: E402
from config import load_config  # noqa: E402
from dirichlet import apply_bcs  # noqa: E402
from exodus_mesh import build_mesh  # noqa: E402
from forms import build_forms  # noqa: E402
from postproc import postprocess  # noqa: E402
from reactions import ReactionHistory  # noqa: E402
from solver import solve_newton  # noqa: E402
from utils import configure_logging  # noqa: E402


class _FieldCapture:
    """Valeurs nodales lues au vol dans write_vtx (aucune copie n'est réinjectée dans le calcul)."""

    def __init__(self) -> None:
        self.requested: tuple[str, ...] = ()
        self.values: dict[str, np.ndarray] = {}
        self.u_h = None
        self.spaces: dict = {}


def _install_field_capture(capture: _FieldCapture):
    """Enveloppe postproc.write_vtx : copie ses arguments, puis délègue à l'original."""
    original = pp.write_vtx

    def wrapper(comm, config, spaces, u_h, requested, values):
        capture.requested = tuple(requested)
        capture.values = {name: np.array(val, dtype=np.float64, copy=True) for name, val in values.items()}
        capture.u_h = u_h
        capture.spaces = spaces
        return original(comm, config, spaces, u_h, requested, values)

    pp.write_vtx = wrapper


def _field_statistics(comm: MPI.Comm, flat: np.ndarray, n_cols: int, n_own: int) -> dict[str, float]:
    """
    min, max et norme l2 des valeurs nodales possédées d'un champ à n_cols composantes.
    'flat' peut contenir des nœuds fantômes en fin de tableau : ils sont écartés.
    """
    rows = np.asarray(flat, dtype=np.float64).reshape(-1, n_cols)[:n_own]
    if rows.size:
        local_min, local_max = float(rows.min()), float(rows.max())
        local_sq = float(np.einsum("ij,ij->", rows, rows))
    else:  # rang sans nœud possédé : neutre pour les trois réductions
        local_min, local_max, local_sq = np.inf, -np.inf, 0.0
    return {
        "components": int(n_cols),
        "min": float(comm.allreduce(local_min, op=MPI.MIN)),
        "max": float(comm.allreduce(local_max, op=MPI.MAX)),
        "l2": float(math.sqrt(comm.allreduce(local_sq, op=MPI.SUM))),
    }


def _exported_field_statistics(comm: MPI.Comm, capture: _FieldCapture) -> dict[str, dict[str, float]]:
    """Statistiques de chaque champ exporté, dans l'ordre d'écriture du VTX."""
    u_h = capture.u_h
    if u_h is None:  # inatteignable : write_vtx a été appelé par postprocess
        raise RuntimeError("Aucune valeur de champ capturée : write_vtx n'a pas été appelé.")
    u_space = u_h.function_space
    gdim = u_space.mesh.geometry.dim
    stats = {
        "displacement": _field_statistics(
            comm, u_h.x.array, gdim, u_space.dofmap.index_map.size_local
        )
    }
    for name in capture.requested:
        if name == "displacement":
            continue
        data = capture.values[name]
        n_cols = int(data.shape[1])
        stats[name] = _field_statistics(
            comm, data, n_cols, capture.spaces[n_cols].dofmap.index_map.size_local
        )
    return stats


def _write_json(path: str, payload: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture les métriques d'une étude en précision complète.")
    parser.add_argument("study", help="Chemin vers le fichier YAML de l'étude")
    parser.add_argument("--out", required=True, help="Chemin du fichier JSON de sortie (métriques)")
    parser.add_argument(
        "--fields-out",
        default=None,
        help="Chemin du fichier JSON des statistiques de champs (optionnel)",
    )
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    configure_logging(comm)

    config = load_config(args.study)
    mesh = build_mesh(comm, config)
    forms = build_forms(mesh.domain, mesh.facet_tags, mesh.boundary_tags, config)
    dirichlet = apply_bcs(forms, mesh.facet_tags, mesh.boundary_tags, config)

    history = ReactionHistory(comm, forms, dirichlet)
    metrics = solve_newton(comm, forms, dirichlet, config, on_increment=history)
    capture = _FieldCapture()
    if args.fields_out:
        _install_field_capture(capture)
    results = postprocess(comm, mesh.domain, forms, config)

    final = history.last
    payload = {
        "mpi_size": int(comm.size),
        "u_max": float(results.u_max),
        "sigma_1_max": float(results.sigma_1_max),
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

    if args.fields_out:
        fields_payload = {"mpi_size": int(comm.size), "fields": _exported_field_statistics(comm, capture)}

    if comm.rank == 0:
        _write_json(args.out, payload)
        if args.fields_out:
            _write_json(args.fields_out, fields_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
