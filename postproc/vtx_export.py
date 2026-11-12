"""Écriture VTX (ParaView) : le déplacement en premier, puis les champs demandés."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from dolfinx import fem
from dolfinx.io import VTXWriter
from mpi4py import MPI

from preproc.config import Config
from utils import ensure_dir


def write_vtx(
    comm: MPI.Comm,
    config: Config,
    spaces: dict[int, fem.FunctionSpace],
    u_h: fem.Function,
    requested: tuple[str, ...],
    values: dict[str, np.ndarray],
) -> tuple[str, list[str]]:
    """Export VTX : le déplacement (vectoriel autonome) en premier, puis les champs demandés."""
    u_out = fem.Function(spaces[3], name="displacement")
    u_out.interpolate(u_h)
    u_out.x.scatter_forward()
    fields = [u_out]
    exported = ["displacement"]

    for name in requested:
        if name == "displacement":
            continue
        data = values[name]
        f = fem.Function(spaces[data.shape[1]], name=name)
        f.x.array[:] = data.reshape(-1)
        fields.append(f)
        exported.append(name)

    outdir = ensure_dir(comm, Path(config.output.outdir))

    vtx_path = outdir / f"{config.study_name}.bp"
    with VTXWriter(comm, vtx_path, fields, engine="BP4") as w:
        w.write(0.0)
    return str(vtx_path), exported