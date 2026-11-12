from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from dolfinx import fem
from dolfinx.fem.petsc import apply_lifting, assemble_matrix, create_matrix, create_vector, set_bc
from mpi4py import MPI
from petsc4py import PETSc

from preproc.config import Config, NewtonConfig
from preproc.forms import Forms
from solver.linear_solver import LinearSolver, NewtonDiverged, check_nullspace
from utils import assemble_vec, petsc_vec

logger = logging.getLogger(__name__)

CONVERGED_EPS = 1.0e-12  # tolérance sur lambda pour considérer le chargement terminé

# Pilotage du pas de charge
CUTBACK_FACTOR = 0.5    # pas après un échec : fraction du pas tenté
EASY_ITERATIONS = 4     # un incrément est « facile » s'il converge en au plus ce nombre d'itérations de Newton
EASY_STREAK = 2         # nombre d'incréments faciles consécutifs avant d'agrandir le pas
STEP_GROWTH = 1.5       # facteur d'agrandissement du pas (plafonné au pas nominal)


@dataclass(frozen=True)
class SolverMetrics:
    """Métriques d'un solve_newton terminé (remplace l'ancien dict à clés)."""
    method: str            # méthode du solveur linéaire (solver.method du YAML)
    iterations: int        # itérations KSP cumulées
    newton_iterations: int  # résolutions de Newton cumulées (une par itération convergée)
    increments: int        # incréments de charge acceptés
    cutbacks: int          # réductions de pas après échec d'un incrément
    t_asm: float           # temps d'assemblage cumulé [s]
    t_solve: float         # temps de résolution linéaire cumulé [s]
    rel_res: float         # résidu relatif final du dernier incrément accepté
    line_search_evaluations: int = 0  # résidus assemblés par la recherche linéaire (backtracking)


# -----------------------------------------------------------------------------
# Newton-Raphson
# -----------------------------------------------------------------------------
# Recherche linéaire par backtracking sur la norme du résidu libre (§3.6) : pas plein d'abord,
# puis pas divisé par deux jusqu'à décroissance stricte, sans descendre sous le pas minimal
# (en dessous, l'incrément est abandonné au mécanisme de cutback du pilotage de charge).
LINE_SEARCH_MIN_STEP = 2.0**-6
LINE_SEARCH_SHRINK = 0.5


@dataclass
class _NewtonWorkspace:
    """Matrice et vecteurs de travail créés UNE fois pour tout le calcul (§3.5).

    Avant : assemble_matrix / assemble_vec / duplicate allouaient une Mat et deux Vec à CHAQUE
    itération de Newton (et pour GAMG, setOperators sur une matrice neuve reconstruisait toute
    la hiérarchie multigrille). Ici : même matrice (zeroEntries + réassemblage), mêmes vecteurs.
    """
    matrix: PETSc.Mat      # jacobienne assemblée en place
    residual: PETSc.Vec    # résidu courant, réassemblé en place
    direction: PETSc.Vec   # correction de Newton du
    free: PETSc.Vec        # copie du résidu pour la norme libre (lignes contraintes exclues)
    backup: PETSc.Vec      # copie de l'état, pour les essais de pas de la recherche linéaire

    @classmethod
    def create(cls, forms: Forms) -> _NewtonWorkspace:
        return cls(
            matrix=create_matrix(forms.jacobian),
            residual=create_vector(forms.V),
            direction=create_vector(forms.V),
            free=create_vector(forms.V),
            backup=create_vector(forms.V),
        )

    def destroy(self) -> None:
        for vec in (self.residual, self.direction, self.free, self.backup):
            vec.destroy()
        self.matrix.destroy()


def _newton_increment(
    forms: Forms, bcs: list[fem.DirichletBC], lin: LinearSolver, nw: NewtonConfig,
    timers: dict[str, float], work: _NewtonWorkspace,
) -> tuple[int, float, int]:
    """
    Itérations de Newton pour un niveau de chargement donné, dans l'espace de travail préalloué.

    Les Dirichlet sont imposés par relèvement (apply_lifting) à la première itération : la correction
    amène u sur la valeur imposée EN RÉPARTISSANT le saut dans tout le domaine (prédicteur tangent).
    Imposer directement la valeur sur le bord avec un intérieur inchangé concentrerait tout le saut
    dans la première couche d'éléments, d'où des éléments retournés (J <= 0) même à faible chargement.

    Chaque correction est suivie d'une recherche linéaire par backtracking : le pas plein est essayé
    d'abord, puis divisé par deux jusqu'à décroissance stricte de la norme du résidu libre (au plus
    jusqu'à LINE_SEARCH_MIN_STEP ; au-delà, NewtonDiverged laisse le cutback réduire le pas de charge).
    En hyperélasticité, c'est ce qui évite les états à J <= 0 qu'un Newton pur traverserait.

    Convergence : au moins une résolution est faite (la première itération porte le saut Dirichlet),
    puis résidu libre (dofs non contraints) <= max(rtol * r_ref, atol), où r_ref est la norme du second
    membre libre de la première itération (déséquilibre dû au pas de charge et au saut imposé).
    Retourne (nombre de résolutions linéaires, résidu relatif final, évaluations de recherche linéaire).
    """
    uh = forms.uh
    x = petsc_vec(uh)

    def lift(b) -> None:
        apply_lifting(b, [forms.jacobian], bcs=[bcs], x0=[x], alpha=-1.0)

    def assemble_residual(*, lifted: bool) -> None:
        t0 = time.perf_counter()
        assemble_vec(forms.residual, before_accumulate=lift if lifted else None, b=work.residual)
        timers["asm"] += time.perf_counter() - t0

    def free_norm() -> float:
        """Norme du résidu libre : les lignes contraintes sont exclues (copie, puis set_bc à 0)."""
        work.residual.copy(work.free)
        set_bc(work.free, bcs, alpha=0.0)
        return work.free.norm()

    def line_search(current: float) -> tuple[float, int]:
        """
        Backtracking depuis l'état courant : essaie u - step * du pour step = 1, 1/2, 1/4, ...
        L'état accepté est laissé dans x et son résidu dans work.residual (pas de réassemblage ensuite).
        """
        x.copy(work.backup)
        step = 1.0
        evaluations = 0
        while True:
            work.backup.copy(x)
            x.axpy(-step, work.direction)  # u <- u_backup - step * du
            uh.x.scatter_forward()
            assemble_residual(lifted=False)
            trial = free_norm()
            evaluations += 1
            if np.isfinite(trial) and (current == 0.0 or trial < current):
                return trial, evaluations
            if step <= LINE_SEARCH_MIN_STEP:
                raise NewtonDiverged(
                    f"recherche linéaire : aucun pas décroissant (||R|| libre reste >= {current:.3e})"
                )
            step *= LINE_SEARCH_SHRINK

    assemble_residual(lifted=True)
    r_norm = free_norm()
    if not np.isfinite(r_norm):
        raise NewtonDiverged("résidu non fini (déformation excessive, J <= 0 ?)")
    r_ref = r_norm
    tolerance = max(nw.rtol * r_ref, nw.atol)
    n_solves = 0
    ls_evaluations = 0

    while True:
        if n_solves > 0 and r_norm <= tolerance:
            return n_solves, (r_norm / r_ref if r_ref > 0.0 else 0.0), ls_evaluations
        if n_solves >= nw.max_iter:
            rel = r_norm / r_ref if r_ref > 0.0 else 0.0
            raise NewtonDiverged(f"{nw.max_iter} itérations atteintes (résidu relatif {rel:.2e})")

        # Second membre de la résolution : à la première, les lignes contraintes portent du = u - g
        # (le saut Dirichlet, réparti par le relèvement) ; ensuite elles sont nulles (u ne bouge plus
        # sur ces dofs). work.residual est ensuite réassemblé par la recherche linéaire.
        if n_solves == 0:
            set_bc(work.residual, bcs, x, -1.0)
        else:
            set_bc(work.residual, bcs, alpha=0.0)

        t0 = time.perf_counter()
        work.matrix.zeroEntries()
        assemble_matrix(work.matrix, forms.jacobian, bcs=bcs)
        work.matrix.assemble()
        timers["asm"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        lin.solve(work.matrix, work.residual, work.direction)  # J du = R
        timers["solve"] += time.perf_counter() - t0
        n_solves += 1

        r_norm, evaluations = line_search(r_norm)
        ls_evaluations += evaluations


def solve_newton(
    comm: MPI.Comm,
    forms: Forms,
    bcs_at: Callable[[float], list[fem.DirichletBC]],
    config: Config,
    on_increment: Callable[[float], None] | None = None,
) -> SolverMetrics:
    """
    Solveur unique pour toutes les lois : chargement proportionnel lambda de 0 à 1 par incréments
    (nominal : 1/solver.newton.increments), Newton (avec recherche linéaire) à chaque incrément,
    réduction automatique du pas en cas d'échec (jusqu'à max_cutbacks réductions successives),
    retour progressif au pas nominal après EASY_STREAK incréments faciles consécutifs.
    Un problème linéaire converge en une résolution : le cas linéaire n'est pas un cas particulier.

    La solution est écrite en place dans forms.uh (lisible via forms.displacement()).
    Retourne les métriques du calcul (SolverMetrics : méthode, itérations, temps, résidu final).
    """
    nw = config.solver.newton
    uh = forms.uh
    timers = {"asm": 0.0, "solve": 0.0}
    n_steps = n_newton = n_cutbacks = n_ls_evaluations = 0
    rel_res = 0.0

    with LinearSolver(comm, forms.V, config) as lin:
        if (
            lin.nullspace is not None
            and config.solver.check_nullspace
            and not check_nullspace(forms.jacobian, lin.nullspace)
        ):
            raise RuntimeError("Les modes rigides ne sont pas dans le noyau de la matrice de raideur.")

        work = _NewtonWorkspace.create(forms)
        try:
            dlam_nominal = 1.0 / nw.increments
            dlam = dlam_nominal
            lam = 0.0
            fails = 0
            easy = 0

            while lam < 1.0 - CONVERGED_EPS:
                lam_try = lam + min(dlam, 1.0 - lam)
                if lam_try > 1.0 - CONVERGED_EPS:
                    lam_try = 1.0

                backup = uh.x.array.copy()
                forms.load.value = lam_try
                bcs = bcs_at(lam_try)  # valeurs imposées de ce niveau de charge (relèvement à la 1re résolution)

                try:
                    its, rel, ls_evals = _newton_increment(forms, bcs, lin, nw, timers, work)
                except NewtonDiverged as exc:
                    uh.x.array[:] = backup
                    uh.x.scatter_forward()
                    fails += 1
                    easy = 0
                    n_cutbacks += 1
                    if fails > nw.max_cutbacks:
                        raise RuntimeError(
                            f"Newton : échec à lambda = {lam_try:.4g} ({exc}) "
                            f"après {nw.max_cutbacks} réductions de pas successives."
                        ) from exc
                    dlam = CUTBACK_FACTOR * (lam_try - lam)
                    logger.info(f"  lambda = {lam_try:.4g} non convergé ({exc}) -> pas réduit à {dlam:.3g}")
                    continue

                lam = lam_try
                fails = 0
                n_steps += 1
                n_newton += its
                n_ls_evaluations += ls_evals
                rel_res = rel
                logger.debug(f"  lambda = {lam:.4g} : {its} itération(s) de Newton, résidu relatif {rel:.2e}")
                if on_increment is not None:
                    on_increment(lam)  # ex : ReactionHistory (uh est à l'équilibre à ce lambda)

                # Retour progressif vers le pas nominal, seulement après EASY_STREAK incréments faciles de suite
                # (évite l'oscillation échec / succès d'un pas agrandi à chaque réussite).
                easy = easy + 1 if its <= EASY_ITERATIONS else 0
                if easy >= EASY_STREAK and dlam < dlam_nominal:
                    dlam = min(STEP_GROWTH * dlam, dlam_nominal)
                    easy = 0

            ksp_its = lin.iterations
        finally:
            work.destroy()

    return SolverMetrics(
        method=config.solver.method,
        iterations=ksp_its,
        newton_iterations=n_newton,
        increments=n_steps,
        cutbacks=n_cutbacks,
        t_asm=timers["asm"],
        t_solve=timers["solve"],
        rel_res=rel_res,
        line_search_evaluations=n_ls_evaluations,
    )
