"""
Lecture des maillages Exodus II (Cubit) : aucune dépendance à dolfinx (comme le catalogue de config.py).

Un fichier .exo exporté par Cubit contient tout ce qu'il faut au solveur : coordonnées, maillage et
regroupements nommés de faces (sidesets). Ce module en extrait ces informations sans les interpréter ;
c'est exodus_mesh.py qui les convertit en maillage dolfinx.

Limites volontaires de la v1 (erreurs explicites plutôt que comportement implicite) :
  * un seul bloc d'éléments (pas de multi-matériaux) ;
  * HEX8 et HEX20 uniquement ;
  * numérotation d'éléments identitaire (elem_num_map absent ou trivial) : les sidesets référencent
    les indices du fichier, tels que les écrit Cubit ; une numérotation exotique serait ambiguë.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import netCDF4
import numpy as np

SUPPORTED_CELL_TYPES = ("HEX8", "HEX20")
N_HEX_SIDES = 6  # numéros de face valides pour un hexaèdre Exodus (side_ss)


@dataclass(frozen=True)
class ExodusMesh:
    """Contenu utile d'un fichier Exodus : maillage, provenance et regroupements nommés."""
    points: np.ndarray                                  # (n_nodes, 3) coordonnées [mm]
    cells: np.ndarray                                   # (n_elems, n_nodes_par_elem), 0-based, ordre Exodus
    cell_type: str                                      # "HEX8" | "HEX20"
    sidesets: dict[str, tuple[np.ndarray, np.ndarray]]  # nom -> (éléments 0-based, faces 1..6)

    @property
    def n_elems(self) -> int:
        return int(self.cells.shape[0])

    @property
    def n_nodes(self) -> int:
        return int(self.points.shape[0])


# -----------------------------------------------------------------------------
# Utilitaires de lecture netCDF
# -----------------------------------------------------------------------------
def _var(ds: netCDF4.Dataset, name: str) -> np.ndarray:
    """Variable netCDF numérique ; une valeur masquée est une donnée manquante rejetée explicitement."""
    values = ds.variables[name][:]
    if np.ma.getmaskarray(values).any():
        raise ValueError(f"{Path(ds.filepath()).name} : variable '{name}' masquée (données manquantes).")
    return np.asarray(values)


def _dim_len(ds: netCDF4.Dataset, name: str) -> int:
    """Taille d'une dimension (0 si absente : les sidesets sont optionnels en Exodus)."""
    return len(ds.dimensions[name]) if name in ds.dimensions else 0


def _char_var(ds: netCDF4.Dataset, name: str) -> np.ndarray:
    """Variable texte d'Exodus (tableau de char), valeurs manquantes neutralisées, jamais numérisées."""
    values = ds.variables[name][:]
    return np.ma.filled(values, b"\x00") if np.ma.isMaskedArray(values) else np.asarray(values)


def _decode_names(raw: np.ndarray, count: int) -> list[str]:
    """Décode des noms de longueur fixe ('name_el_blk' ou 'eb_names' : tableaux de char d'Exodus)."""
    names = []
    for row in np.atleast_2d(raw)[:count]:
        if row.dtype.kind == "S":
            blob = row.tobytes() if row.dtype.itemsize == 1 else b"".join(bytes(x) for x in row.tolist())
        elif row.dtype.kind == "U":
            blob = b"".join(str(x).encode("latin1", "replace") for x in row.tolist())
        else:
            blob = np.asarray(row, dtype="S1").tobytes()
        names.append(blob.decode("latin1").split("\x00")[0].strip())
    return names


def _names(ds: netCDF4.Dataset, *candidates: str, count: int) -> list[str]:
    """Noms des entités ; deux conventions existent selon l'écrivain ('eb_names' ou 'name_el_blk')."""
    for var in candidates:
        if var in ds.variables:
            return _decode_names(_char_var(ds, var), count)
    return [""] * count


# -----------------------------------------------------------------------------
# Lecture
# -----------------------------------------------------------------------------
def read_exodus(path: str | Path) -> ExodusMesh:
    """
    Lit un fichier Exodus II et retourne son maillage (points, cellules) et ses sidesets nommés.
    Toute limite de la v1 ou incohérence du fichier lève une erreur explicite.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier de maillage Exodus introuvable : {path}")

    with netCDF4.Dataset(str(path), "r") as ds:
        if "num_dim" not in ds.dimensions:
            raise ValueError(f"{path.name} : ce fichier n'est pas un fichier Exodus (dimension 'num_dim' absente).")
        n_dim = _dim_len(ds, "num_dim")
        if n_dim != 3:
            raise ValueError(f"{path.name} : maillage {n_dim}D non supporté (le solveur est 3D).")

        n_nodes = _dim_len(ds, "num_nodes")
        n_elems = _dim_len(ds, "num_elem")
        n_blocks = _dim_len(ds, "num_el_blk")
        if n_blocks != 1:
            raise ValueError(
                f"{path.name} : {n_blocks} blocs d'éléments (multi-matériaux non supporté : "
                "exporter un seul bloc de volume depuis Cubit)."
            )
        if n_nodes <= 0:
            raise ValueError(f"{path.name} : maillage sans nœud (num_nodes = {n_nodes}).")
        if n_elems <= 0:
            raise ValueError(f"{path.name} : maillage sans élément (num_elem = {n_elems}).")
        if "connect1" not in ds.variables:
            raise ValueError(f"{path.name} : bloc d'éléments 'connect1' absent.")

        cell_type = str(getattr(ds.variables["connect1"], "elem_type", "")).strip().upper()
        if cell_type not in SUPPORTED_CELL_TYPES:
            raise ValueError(
                f"{path.name} : type d'élément {cell_type or 'inconnu'!r} non supporté "
                f"(attendu : {' ou '.join(SUPPORTED_CELL_TYPES)})."
            )

        missing = [c for c in "xyz" if f"coord{c}" not in ds.variables]
        if missing:
            raise ValueError(f"{path.name} : coordonnées {missing} absentes du fichier.")
        coords = [np.asarray(_var(ds, f"coord{c}"), dtype=np.float64) for c in "xyz"]
        if any(coord.ndim != 1 for coord in coords) or len({coord.shape[0] for coord in coords}) != 1:
            raise ValueError(
                f"{path.name} : coordonnées de dimensions incohérentes "
                f"({', '.join(str(coord.shape) for coord in coords)})."
            )
        points = np.ascontiguousarray(np.column_stack(coords), dtype=np.float64)
        if points.shape[0] != n_nodes:
            raise ValueError(f"{path.name} : {points.shape[0]} nœuds lus pour 'num_nodes' = {n_nodes}.")
        if not np.isfinite(points).all():
            raise ValueError(f"{path.name} : coordonnées non finies (NaN ou Inf).")

        cells = np.ascontiguousarray(np.asarray(_var(ds, "connect1"), dtype=np.int64) - 1, dtype=np.int64)
        if cells.ndim != 2 or cells.shape[0] != n_elems:
            raise ValueError(f"{path.name} : connectivité de forme {cells.shape} inattendue.")
        n_per_elem = {"HEX8": 8, "HEX20": 20}[cell_type]
        if cells.shape[1] != n_per_elem:
            raise ValueError(
                f"{path.name} : connectivité de {cells.shape[1]} nœuds par élément "
                f"(attendu {n_per_elem} pour {cell_type})."
            )
        if cells.min() < 0 or cells.max() >= n_nodes:
            raise ValueError(f"{path.name} : connectivité hors bornes (nœuds 1..{n_nodes}).")

        # Les sidesets référencent la numérotation des éléments du fichier : une numérotation
        # utilisateur non identitaire rendrait cette référence ambiguë.
        num_map = np.asarray(_var(ds, "elem_num_map") if "elem_num_map" in ds.variables else np.arange(1, n_elems + 1), dtype=np.int64)
        if not np.array_equal(num_map, np.arange(1, n_elems + 1)):
            raise ValueError(
                f"{path.name} : numérotation d'éléments non identitaire (elem_num_map) non supportée : "
                "réexporter le maillage depuis Cubit sans renumérotation."
            )

        n_sidesets = _dim_len(ds, "num_side_sets")
        ss_names = _names(ds, "ss_names", "name_ss", count=n_sidesets)
        sidesets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for s in range(1, n_sidesets + 1):
            name = ss_names[s - 1]
            if not name:
                raise ValueError(f"{path.name} : sideset {s} sans nom (les sidesets doivent être nommés pour le YAML).")
            if name in sidesets:
                raise ValueError(f"{path.name} : sideset '{name}' défini deux fois.")
            elems = np.ascontiguousarray(np.asarray(_var(ds, f"elem_ss{s}"), dtype=np.int64) - 1, dtype=np.int64)
            sides = np.ascontiguousarray(np.asarray(_var(ds, f"side_ss{s}"), dtype=np.int64), dtype=np.int64)
            if elems.ndim != 1 or sides.ndim != 1 or elems.shape != sides.shape:
                raise ValueError(f"{path.name} : sideset '{name}' : listes de longueurs différentes.")
            if elems.size and (elems.min() < 0 or elems.max() >= n_elems):
                raise ValueError(f"{path.name} : sideset '{name}' référence un élément hors bornes.")
            if sides.size and (sides.min() < 1 or sides.max() > N_HEX_SIDES):
                raise ValueError(
                    f"{path.name} : sideset '{name}' : numéro de face hors bornes "
                    f"({sides.min()}..{sides.max()}, attendu 1..{N_HEX_SIDES})."
                )
            sidesets[name] = (elems, sides)

    return ExodusMesh(points=points, cells=cells, cell_type=cell_type, sidesets=sidesets)
