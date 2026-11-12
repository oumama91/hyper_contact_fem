from __future__ import annotations

from dolfinx import fem, la
from dolfinx.fem.petsc import assemble_matrix
from mpi4py import MPI
from petsc4py import PETSc

from preproc.config import Config
from utils import SCALAR

# Multigrille algébrique (GAMG) : étapes d'estimation de la plus grande valeur propre pour Chebyshev
GAMG_ESTEIG_STEPS = 10

class NewtonDiverged(RuntimeError):
    """Échec d'un incrément (résidu non fini, trop d'itérations, échec du solveur linéaire)."""
    
# -----------------------------------------------------------------------------
# Modes rigides (near-nullspace pour GAMG)
# -----------------------------------------------------------------------------
def build_nullspace(V: fem.FunctionSpace) -> PETSc.NullSpace:
    """
    Construit l'espace nul (modes rigides) pour le solveur algébrique multigrille (GAMG).
    Indispensable pour l'élasticité 3D.
    """
    bs = V.dofmap.index_map_bs
    if bs != 3:
        # Couplage implicite rendu explicite : V.sub(i) et bsize=3 ci-dessous supposent un espace
        # de déplacement vectoriel. En formulation mixte, seule la règle check_setup (hyperélasticité
        # => method: direct) empêchait d'arriver ici ; autant le dire clairement.
        raise RuntimeError(
            "build_nullspace suppose un espace de déplacement vectoriel 3D (bloc de 3) : "
            "GAMG est réservé à la formulation en déplacement pur (voir Material.check_setup)."
        )
    n_local = V.dofmap.index_map.size_local
    basis = [la.vector(V.dofmap.index_map, bs=bs, dtype=SCALAR) for _ in range(6)]
    b = [vec.array for vec in basis]

    dofs = [V.sub(i).dofmap.list.flatten() for i in range(3)]
    for i in range(3):
        b[i][dofs[i]] = 1.0

    xyz = V.tabulate_dof_coordinates()
    dofs_block = V.dofmap.list.flatten()
    x0, x1, x2 = xyz[dofs_block, 0], xyz[dofs_block, 1], xyz[dofs_block, 2]

    b[3][dofs[0]], b[3][dofs[1]] = -x1, x0
    b[4][dofs[0]], b[4][dofs[2]] = x2, -x0
    b[5][dofs[2]], b[5][dofs[1]] = x1, -x2

    la.orthonormalize(basis)
    vecs = [PETSc.Vec().createWithArray(arr[: bs * n_local], bsize=3, comm=V.mesh.comm) for arr in b]
    return PETSc.NullSpace().create(vectors=vecs)


def check_nullspace(jacobian: fem.Form, ns: PETSc.NullSpace) -> bool:
    """
    Vérifie que les modes rigides annulent la tangente SANS conditions aux limites.
    À appeler avec uh = 0 (la tangente en F = I est celle de l'élasticité linéaire).
    """
    A0 = assemble_matrix(jacobian)
    A0.assemble()
    ok = ns.test(A0)
    A0.destroy()
    return ok


# -----------------------------------------------------------------------------
# Solveur linéaire interne (KSP PETSc)
# -----------------------------------------------------------------------------
class LinearSolver:
    """
    Contexte qui configure un KSP (préfixe 'fea_') pour résoudre A du = b et retire
    les options globales PETSc à la sortie (succès ou échec), pour ne pas contaminer
    une étude suivante.
    Les options mg_levels_* ne sont lues qu'au setup du préconditionneur (pendant solve) :
    elles restent donc posées jusqu'à la fermeture du contexte.
    La méthode est validée par la config (solver.method) : 'gamg' ou 'direct'.
    """
    PREFIX = "fea_"

    def __init__(self, comm: MPI.Comm, V: fem.FunctionSpace, config: Config):
        solver = config.solver
        self.comm = comm
        self.V = V
        self.method: str = solver.method
        self.rtol = solver.rtol
        self.krylov: str = solver.krylov
        self.nullspace: PETSc.NullSpace | None = None
        self._operators: PETSc.Mat | None = None  # matrice déjà paramétrée (near-nullspace, réutilisation PC)
        self.iterations = 0  # itérations KSP cumulées
        self._ksp: PETSc.KSP | None = None
        self._keys: list[str] = []

    def _set(self, name: str, value: str | int | float) -> None:
        key = f"{self.PREFIX}{name}"
        PETSc.Options()[key] = value
        self._keys.append(key)

    def __enter__(self) -> LinearSolver:
        try:
            if self.method == "gamg":
                self.nullspace = build_nullspace(self.V)
                self._set("ksp_type", self.krylov)
                self._set("ksp_rtol", self.rtol)
                self._set("pc_type", "gamg")
                self._set("mg_levels_ksp_type", "chebyshev")
                self._set("mg_levels_pc_type", "jacobi")
                self._set("mg_levels_ksp_chebyshev_esteig_steps", GAMG_ESTEIG_STEPS)
            else:  # "direct"
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
        for obj in (self._ksp, self.nullspace):
            if obj is not None:
                obj.destroy()
        self._ksp = None
        self.nullspace = None
        self._operators = None

    def solve(self, A: PETSc.Mat, b: PETSc.Vec, x: PETSc.Vec) -> None:
        if A is not self._operators:
            # Premier appel : paramétrage attaché à la matrice (qui garde ensuite la même identité,
            # préallocation oblige). setOperators reste appelé à chaque résolution ci-dessous :
            # c'est lui qui signale les nouvelles valeurs (factorisation refaite pour un direct).
            if self.method == "gamg":
                A.setNearNullSpace(self.nullspace)
                if self.krylov == "cg":
                    A.setOption(PETSc.Mat.Option.SPD, True)
                # Hiérarchie multigrille réutilisée entre itérations de Newton : le préconditionneur
                # reste bâti sur les anciennes valeurs, mais le Krylov résout avec la matrice courante.
                self._ksp.getPC().setReusePreconditioner(True)  # méthode du PC, pas du KSP
            self._operators = A
        self._ksp.setOperators(A)
        x.zeroEntries()  # point de départ déterministe du Krylov
        self._ksp.solve(b, x)
        reason = self._ksp.getConvergedReason()
        self.iterations += self._ksp.getIterationNumber()
        if reason < 0:
            hint = ""
            if self.method == "gamg" and self.krylov == "cg":
                hint = " ; tangente peut-être non définie positive : essayer solver.krylov: gmres ou solver.method: direct"
            raise NewtonDiverged(f"solveur linéaire non convergé (raison PETSc = {reason}){hint}")