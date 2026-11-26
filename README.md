# hyper_contact_fem

**Version 1.0** — solveur éléments finis 3D FEniCSx pour un corps hyperélastique
quasi-incompressible en contact unilatéral pénalisé avec frottement de Coulomb
contre des obstacles rigides analytiques. Maillage Exodus II, étude YAML.

## Installation rapide

Linux/WSL et conda-forge ; PETSc doit inclure MUMPS.

```bash
conda create -n fenicsx-env -c conda-forge \
  python=3.12 fenics-dolfinx=0.11.0 petsc4py=3.25.6 mpi4py=4.1.2 \
  numpy=2.5.3 netcdf4=1.7.4 pydantic=2.14.1 pyyaml=6.0.3 pytest=9.1.1
conda activate fenicsx-env
```

## Usage

```bash
python main.py studies/O-ring_compression.yaml                  # série
mpiexec -n 2 python main.py studies/O-ring_compression.yaml     # parallèle
python -m pytest tests                                          # tests
```

Sous Windows, exécuter ces commandes dans WSL via `bash tools/wsl_run.sh <commande>`.
`output.outdir` est obligatoire ; `mesh.file` et `output.outdir` sont relatifs au fichier YAML.

## Structure

```text
hyper_contact_fem/   code : configuration, maillage, physique, solveur, post-traitement
main.py              point d'entrée compatible
studies/             cas YAML
mesh/                maillage Exodus II, journal Cubit et géométrie source
reference/           sorties figées de non-régression
tools/               capture/comparaison des métriques, wrapper WSL
tests/               tests pytest
docs/                documentation détaillée
results/             sorties générées (ignoré par Git)
```

## Documentation

- [Modèle mécanique et limites](docs/model.md)
- [Fichier YAML, maillage et sorties](docs/usage.md)
- [Validation et références](docs/validation.md)
- [Architecture du code](docs/architecture.md)
- [Références de non-régression](reference/README.md)

## Licence

MIT — voir [LICENSE](LICENSE).
