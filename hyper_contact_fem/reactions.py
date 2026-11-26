from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from mpi4py import MPI

from hyper_contact_fem.dirichlet import DirichletSet
from hyper_contact_fem.forms import Forms
from hyper_contact_fem.utils import assemble_vec, ensure_dir, owned_size


@dataclass(frozen=True)
class Reactions:
    """Réactions d'appui à l'état courant."""
    reactions: dict[str, np.ndarray]  # boundary de Dirichlet -> réactions (Fx, Fy, Fz) [N]
    total: np.ndarray                 # somme des réactions [N]
    external: np.ndarray              # somme des forces extérieures appliquées [N]
    imbalance: float                  # bilan global relatif (voir compute_reactions)


def compute_reactions(comm: MPI.Comm, forms: Forms, dirichlet: DirichletSet) -> Reactions:
    """
    Réactions d'appui [N] à l'état courant (uh, forms.load) : force exercée PAR l'appui SUR la structure,
    par boundary de Dirichlet et par composante.

    Méthode des forces nodales : R = f_int - f_ext, sommé sur les dofs de déplacement contraints.
    Le mapping dof -> (boundary, composante)
    est précalculé une seule fois dans DirichletSet (dirichlet.names / .owner / .comp_of), pas recalculé
    ici à chaque appel : chaque dof contraint est attribué à la dernière boundary déclarée qui le
    contraint (même règle que apply_bcs : 'la dernière déclarée l'emporte'), donc sans double comptage.

    Bilan global : somme des réactions + somme des forces extérieures = 0 (la somme des f_int est nulle).
    'imbalance' est cet écart, relatif à la somme des normes des réactions individuelles et de la force
    extérieure.

    Retourne un Reactions (réactions par boundary, total, forces extérieures, bilan relatif).
    """
    V = forms.V
    gdim = V.mesh.geometry.dim
    n_flat = owned_size(V)  # dofs possédés (numérotation de V)

    b = assemble_vec(forms.residual)
    e = assemble_vec(forms.external)
    r = np.array(b.array_r[:n_flat])
    f_ext = np.array(e.array_r[:n_flat])
    b.destroy()
    e.destroy()

    names = dirichlet.names
    constrained = dirichlet.owner >= 0
    flat_index = dirichlet.owner[constrained] * gdim + dirichlet.comp_of[constrained]
    local = np.bincount(flat_index, weights=r[constrained], minlength=len(names) * gdim).reshape(len(names), gdim)
    glob = comm.allreduce(local, op=MPI.SUM)

    # Force extérieure par composante : somme sur les dofs de déplacement (les lignes de p valent 0)
    external = comm.allreduce(np.array([f_ext[own].sum() for own in dirichlet.owned]), op=MPI.SUM)
    total = glob.sum(axis=0)

    # Échelle : somme des normes des réactions individuelles + force extérieure. Elle ne s'annule pas quand
    # les réactions se compensent (ex : traction pure, total ~ 0), contrairement à la norme de leur somme.
    scale = sum(float(np.linalg.norm(glob[k])) for k in range(len(names))) + float(np.linalg.norm(external))
    imbalance = float(np.linalg.norm(total + external)) / scale if scale > 0.0 else 0.0

    return Reactions(
        reactions={name: glob[k] for k, name in enumerate(names)},
        total=total,
        external=external,
        imbalance=imbalance,
    )


class ReactionHistory:
    """
    Callable à passer à solve_newton(on_increment=...) : enregistre lambda et les réactions
    à chaque incrément accepté. Les valeurs imposées valent lambda x (valeur du YAML).
    """

    AXES = ("Fx", "Fy", "Fz")

    def __init__(self, comm: MPI.Comm, forms: Forms, dirichlet: DirichletSet):
        self.comm = comm
        self.forms = forms
        self.dirichlet = dirichlet
        self.rows: list[tuple[float, Reactions]] = []  # [(lambda, résultat de compute_reactions)]

    def __call__(self, lam: float) -> None:
        self.rows.append((float(lam), compute_reactions(self.comm, self.forms, self.dirichlet)))

    @property
    def last(self) -> Reactions:
        return self.rows[-1][1]

    def write_csv(self, path: str | Path) -> str:
        """Écrit lambda, réactions par boundary, total, bilan (rang 0 uniquement)."""
        path = Path(path)
        ensure_dir(self.comm, path.parent)
        if self.comm.rank == 0:
            names = list(self.rows[0][1].reactions)
            header = ["lambda"]
            for name in names:
                header += [f"{name}_{ax}" for ax in self.AXES]
            header += [f"total_{ax}" for ax in self.AXES] + ["imbalance"]
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for lam, res in self.rows:
                    row = [f"{lam:.8g}"]
                    for name in names:
                        row += [f"{v:.8e}" for v in res.reactions[name]]
                    row += [f"{v:.8e}" for v in res.total] + [f"{res.imbalance:.3e}"]
                    writer.writerow(row)
        self.comm.Barrier()
        return str(path)
