from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import dolfinx
from mpi4py import MPI
from petsc4py import PETSc

from preproc.config import Config, load_config
from preproc.forms import Forms, build_forms
from preproc.dirichlet import apply_bcs
from preproc.mesher_exodus import MeshResult, build_mesh
from postproc.postproc import PostprocResult, postprocess
from postproc.reactions import ReactionHistory
from solver.solver import SolverMetrics, solve_newton
from utils import configure_logging, total_size

logger = logging.getLogger(__name__)

RULE = "-" * 64

# gmsh n'est requis que par la source de maillage 'gmsh' (backend importé à la demande) : une étude
# purement Exodus (Cubit) doit pouvoir tourner sans gmsh installé.
try:
    import gmsh

    GMSH_LABEL = f"gmsh {gmsh.__version__}"
except ImportError:  # pragma: no cover - dépend de l'environnement
    GMSH_LABEL = "gmsh absent"

# Erreurs attendues (config invalide, CAO/maillage introuvable, Newton non convergé, ...) : message
# clair sans traceback en usage normal. NewtonDiverged (solver/linear_solver.py) est une RuntimeError,
# déjà couverte, et n'a jamais besoin de remonter jusqu'ici (solve_newton la convertit en RuntimeError
# après épuisement des réductions de pas).
# ImportError inclus : dépendance d'un backend de maillage absente (message clair, voir preproc/mesher.py).
EXPECTED_ERRORS = (ValueError, FileNotFoundError, RuntimeError, ImportError)


def print_summary(
    comm: MPI.Comm,
    config: Config,
    mesh: MeshResult,
    forms: Forms,
    solver_metrics: SolverMetrics,
    history: ReactionHistory,
    results: PostprocResult,
    csv_path: str,
) -> None:
    """Affiche le résumé de l'étude (rang 0 uniquement)."""
    domain = mesh.domain
    V = forms.V
    n_cells = domain.topology.index_map(domain.topology.dim).size_global
    n_dofs = total_size(V)
    n_geom_nodes = domain.geometry.index_map().size_global
    k = config.mesh.order
    final = history.last
    t = final.total

    logger.info("\n" + "=" * 64)
    logger.info(f"dolfinx {dolfinx.__version__} / {GMSH_LABEL} / PETSc {PETSc.Sys.getVersion()}")
    logger.info(f"ÉTUDE : {config.study_name}")
    logger.info(f"Loi de comportement : {config.material.model}")
    logger.info("Formulation : " + (f"mixte u-p (Taylor-Hood P{k}-P{k - 1})" if forms.mixed else "déplacement"))
    logger.info(f"Processus MPI : {comm.size}")
    logger.info(RULE)
    logger.info(f"Maillage : {mesh.description} [source : {mesh.source}]")
    logger.info(f"Cellules / DDLs : {n_cells} / {n_dofs}")
    logger.info(f"Ordre FE / Ordre géom : {k} / {mesh.geom_degree}")
    logger.info(f"Nœuds géométrie : {n_geom_nodes}")
    logger.info(f"Volume : {mesh.volume:.2f} mm³")
    logger.info(RULE)
    logger.info(
        f"Newton : {solver_metrics.newton_iterations} itération(s) / "
        f"{solver_metrics.increments} incrément(s), {solver_metrics.cutbacks} réduction(s) de pas",
    )
    logger.info(
        f"Solveur linéaire : {solver_metrics.method} ({solver_metrics.iterations} itérations KSP cumulées)",
    )
    logger.info(f"Temps assemblage : {solver_metrics.t_asm:.2f} s")
    logger.info(f"Temps résolution : {solver_metrics.t_solve:.2f} s")
    logger.info(f"Résidu relatif final : {solver_metrics.rel_res:.2e}")
    logger.info(f"Recherche linéaire : {solver_metrics.line_search_evaluations} évaluation(s) de résidu")
    logger.info(RULE)
    logger.info("Réactions d'appui [N] (force de l'appui sur la structure)")
    for name, f in final.reactions.items():
        logger.info(f"  {name:<22} Fx = {f[0]:+.4e}  Fy = {f[1]:+.4e}  Fz = {f[2]:+.4e}")
    logger.info(f"  {'TOTAL':<22} Fx = {t[0]:+.4e}  Fy = {t[1]:+.4e}  Fz = {t[2]:+.4e}")
    logger.info(f"Bilan global : {final.imbalance:.2e} (somme réactions + forces ext., relatif)")
    logger.info(f"Historique réactions : {csv_path}")
    logger.info(RULE)
    logger.info(f"Déplacement max |u| : {results.u_max:.6e} mm")
    logger.info(f"Von Mises max (lissé P1) : {results.vm_max:.4e} MPa")
    logger.info(f"Export ParaView : {results.vtx_path}")
    logger.info(f"Champs exportés : {', '.join(results.fields)}")
    logger.info("=" * 64 + "\n")


def _run(args: argparse.Namespace, comm: MPI.Comm) -> int:
    """Pipeline complet d'une étude, isolé de main() pour limiter le try/except aux erreurs utilisateur."""
    # 1. Configuration (n'est plus modifiée ensuite)
    config = load_config(args.study)

    # 2. Maillage
    mesh = build_mesh(comm, config)

    # 3. Physique : formulation (loi choisie dans le YAML) et conditions aux limites
    forms = build_forms(mesh.domain, mesh.facet_tags, mesh.boundary_tags, config)
    dirichlet = apply_bcs(forms, mesh.facet_tags, mesh.boundary_tags, config)

    # 4. Résolution (Newton unique : le cas linéaire converge en une résolution).
    #    Les réactions d'appui sont enregistrées à chaque incrément accepté.
    history = ReactionHistory(comm, forms, dirichlet)
    solver_metrics = solve_newton(comm, forms, dirichlet, config, on_increment=history)

    # 5. Post-traitement
    results = postprocess(comm, mesh.domain, forms, config)
    csv_path = history.write_csv(Path(config.output.outdir) / f"{config.study_name}_reactions.csv")

    print_summary(comm, config, mesh, forms, solver_metrics, history, results, csv_path)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Solveur éléments finis statique 3D piloté par YAML (FEniCSx)")
    parser.add_argument("study", help="Chemin vers le fichier YAML de l'étude (ex: studies/beam_bending.yaml)")
    parser.add_argument("--verbose", action="store_true", help="Affiche les détails du maillage, des tags et des incréments")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    configure_logging(comm, verbose=args.verbose)
    try:
        return _run(args, comm)
    except EXPECTED_ERRORS as exc:
        # Config invalide, CAO/maillage introuvable, Newton non convergé, ... : message clair, sans
        # traceback (--verbose le réaffiche intégralement pour le diagnostic).
        logger.error(f"\nERREUR : {exc}\n")
        if args.verbose:
            raise
        if comm.size > 1:
            # En parallèle, un rang qui sort laisserait les autres bloqués dans une collective : on
            # termine le job entier avec un code de retour non nul.
            comm.Abort(1)
        return 1


if __name__ == "__main__":
    sys.exit(main())