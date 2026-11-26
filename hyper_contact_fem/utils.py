from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from pathlib import Path

import ufl
from dolfinx import default_scalar_type, fem
from dolfinx.fem.petsc import assemble_vector
from dolfinx.la.petsc import create_vector_wrap
from mpi4py import MPI
from petsc4py import PETSc

SCALAR = default_scalar_type

# Alias unique du type des expressions UFL.
Expr = ufl.core.expr.Expr

# Options de compilation JIT des formes. -O3 est propre à gcc/clang : on l'omet sous Windows natif (MSVC).
JIT_OPTIONS: dict[str, list[str]] = {} if sys.platform.startswith("win") else {"cffi_extra_compile_args": ["-O3"]}


class _RankZeroFilter(logging.Filter):
    """Ne laisse passer que les messages du rang 0 (pas de sorties MPI dupliquées)."""

    def __init__(self, rank: int):
        super().__init__()
        self._rank = rank

    def filter(self, record: logging.LogRecord) -> bool:
        return self._rank == 0


def configure_logging(comm: MPI.Comm, verbose: bool = False) -> None:
    """
    Journalisation console du calcul : stdout, message seul (le résumé final lu par le test
    golden garde donc exactement son format), rang 0 uniquement. verbose -> niveau DEBUG
    (détails de maillage et d'incréments), sinon INFO. --verbose ne fait que régler ce niveau.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.addFilter(_RankZeroFilter(comm.rank))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG if verbose else logging.INFO)


def petsc_vec(f: fem.Function) -> PETSc.Vec:
    """
    Vecteur PETSc partageant la mémoire de 'f' (aucune copie).

    Point d'accès unique : on utilise l'API publique 'dolfinx.la.petsc.create_vector_wrap'
    (vérifiée dans dolfinx 0.11.0), qui enveloppe le vecteur DOLFINx, plutôt que la propriété
    'f.x.petsc_vec' (Vec mis en cache dans l'objet Vector, non détruit avec la Function).
    """
    return create_vector_wrap(f.x)


def integrate(domain, measure) -> float:
    """Intégrale globale (réduite sur tous les rangs) de la constante 1 sur 'measure'.

    Factorise le motif volume/aire : assemble_scalar(form(Constant(1) * mesure))
    suivi d'un allreduce, utilisé pour le volume du domaine et l'aire d'une face.
    """
    local = fem.assemble_scalar(fem.form(fem.Constant(domain, SCALAR(1.0)) * measure))
    return float(domain.comm.allreduce(local, op=MPI.SUM))


def owned_size(V: fem.FunctionSpace) -> int:
    """Nombre de dofs possédés par ce rang (size_local x taille de bloc)."""
    return V.dofmap.index_map.size_local * V.dofmap.index_map_bs


def total_size(V: fem.FunctionSpace) -> int:
    """Nombre total de dofs, tous rangs confondus (size_global x taille de bloc)."""
    return V.dofmap.index_map.size_global * V.dofmap.index_map_bs


def ensure_dir(comm: MPI.Comm, path: str | Path) -> Path:
    """Crée 'path' sur le rang 0 puis barrière : le dossier existe pour tous les rangs."""
    out = Path(path)
    if comm.rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    comm.Barrier()
    return out


def assemble_vec(
    form: fem.Form,
    before_accumulate: Callable[[PETSc.Vec], None] | None = None,
    b: PETSc.Vec | None = None,
) -> PETSc.Vec:
    """
    Assemble un vecteur puis cumule les contributions fantômes (ghostUpdate ADD / REVERSE).
    'before_accumulate' (ex : apply_lifting) est appelé entre l'assemblage et le cumul, comme l'exige
    le motif standard de dolfinx. 'b' : vecteur existant à réutiliser (préallocation du solveur) ;
    créé si absent. Le vecteur retourné est à détruire par l'appelant.
    """
    if b is None:
        b = assemble_vector(form)
    else:
        # zeroEntries() ne remet à zéro que les entrées propres d'un vecteur fantôme MPI :
        # les contributions fantômes de l'assemblage précédent seraient recomptées par la
        # mise à jour REVERSE ci-dessous (résidu gonflé à chaque réutilisation, invisible
        # en série). La forme locale couvre propres + fantômes.
        with b.localForm() as local:
            local.zeroEntries()
        assemble_vector(b, form)
    if before_accumulate is not None:
        before_accumulate(b)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    return b