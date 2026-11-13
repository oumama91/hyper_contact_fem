from __future__ import annotations

from dolfinx import fem
from mpi4py import MPI
from petsc4py import PETSc

from config import Config


class NewtonDiverged(RuntimeError):
    """Échec d'un incrément (résidu non fini, trop d'itérations, échec du solveur linéaire)."""


# -----------------------------------------------------------------------------
# Solveur linéaire interne (KSP PETSc, LU direct MUMPS uniquement)
# -----------------------------------------------------------------------------
class LinearSolver:
    """
    Contexte qui configure un KSP (préfixe 'fea_') pour résoudre A du = b en LU direct MUMPS,
    et retire les options globales PETSc à la sortie (succès ou échec), pour ne pas contaminer
    une étude suivante.

    La tangente de la formulation mixte (u, p) est de type point-selle, indéfinie : un solveur
    direct est obligatoire (pas de multigrille algébrique ni de Krylov préconditionné).
    """
    PREFIX = "fea_"

    def __init__(self, comm: MPI.Comm, V: fem.FunctionSpace, config: Config):
        self.comm = comm
        self.V = V
        self.rtol = config.solver.rtol  # conservé pour la traçabilité ; sans effet en LU direct
        self.iterations = 0  # itérations KSP cumulées
        self._ksp: PETSc.KSP | None = None
        self._keys: list[str] = []

    def _set(self, name: str, value: str | int | float) -> None:
        key = f"{self.PREFIX}{name}"
        PETSc.Options()[key] = value
        self._keys.append(key)

    def __enter__(self) -> LinearSolver:
        try:
            self._set("ksp_type", "preonly")
            self._set("pc_type", "lu")
            self._set("pc_factor_mat_solver_type", "mumps")

            self._ksp = PETSc.KSP().create(self.comm)
            self._ksp.setOptionsPrefix(self.PREFIX)
            self._ksp.setFromOptions()
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        opts = PETSc.Options()
        for key in self._keys:
            opts.delValue(key)
        self._keys = []
        if self._ksp is not None:
            self._ksp.destroy()
        self._ksp = None

    def solve(self, A: PETSc.Mat, b: PETSc.Vec, x: PETSc.Vec) -> None:
        # setOperators est appelé à chaque résolution : c'est lui qui signale les nouvelles
        # valeurs (la factorisation LU est alors refaite).
        self._ksp.setOperators(A)
        x.zeroEntries()  # point de départ déterministe
        self._ksp.solve(b, x)
        reason = self._ksp.getConvergedReason()
        self.iterations += self._ksp.getIterationNumber()
        if reason < 0:
            raise NewtonDiverged(f"solveur linéaire non convergé (raison PETSc = {reason})")
