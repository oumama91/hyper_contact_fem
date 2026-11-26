# Validation

Le cas de référence `studies/O-ring_compression.yaml` (`HEX20`, $E = 6$ MPa,
$\nu = 0{,}4999$, $P_2$-$P_1$) en série donne : 1 incrément, 10 itérations de Newton,
0 cutback, `u_max` = 4.664172049663e-01 mm, `sigma_1_max` = 1.599963399299e-01 MPa,
`total_Fz` = 2.480275894546e+01 N.

`reference/` fige les journaux, CSV, métriques haute précision et statistiques de champs pour la
série et `np = 2` ; [`reference/README.md`](../reference/README.md) détaille la génération, la
normalisation et les valeurs de référence.

## Commandes de validation

```bash
python tools/capture_metrics.py studies/O-ring_compression.yaml \
    --out results/series_metrics.json \
    --fields-out results/series_fields.json

python tools/compare_reference.py \
    --run results/series_metrics.json \
    --reference reference/O-ring_compression_series_metrics.json \
    --fields-run results/series_fields.json \
    --fields-reference reference/O-ring_compression_series_fields.json
```

Pour `np = 2`, préfixer **uniquement** `capture_metrics.py` par `mpiexec -n 2` et remplacer
`series` par `np2` ; `compare_reference.py` reste série. Tolérances : écart relatif $\le 10^{-12}$
en série (attendu 0), $\le 10^{-8}$ pour `np = 2`, compteurs en égalité stricte ; la
reproductibilité série exige les threads BLAS à 1 fixés par `tools/wsl_run.sh`.

## Tests

La suite `tests/` contient 98 tests rapides et 1 test golden. Le test golden exécute un solveur
complet (environ 1 min 30 en série) puis `tools/compare_reference.py` ; les autres tests couvrent :

- `test_config.py` : schéma strict, bornes et règles croisées ;
- `test_contact.py` : jauge, normale, pénalité et frottement ;
- `test_exodus*.py` : permutations `HEX8`/`HEX20`, sidesets, entrées invalides et caractérisation
  du lecteur ;
- `test_dirichlet_set.py`, `test_postproc.py`, `test_solver_options.py` : localisation des DDL,
  post-traitement et options PETSc ;
- `test_capture_metrics.py`, `test_compare_reference*.py` : outils de capture et de comparaison ;
- `test_oring_golden.py` : run complet comparé à `reference/`.

```bash
python -m pytest tests -q --ignore=tests/test_oring_golden.py   # suite rapide
python -m pytest tests/test_oring_golden.py -q                  # test golden
```

## Wrapper WSL et contrôles statiques

Sous Windows, exécuter les commandes dans WSL via `bash tools/wsl_run.sh <commande>`. Le wrapper
active conda, cherche l'installation sous `$HOME/miniconda3` (surcharge possible avec `CONDA_BASE`)
et fixe par défaut `OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = 1` ; une valeur
déjà fixée par l'appelant reste prioritaire.

Contrôles statiques optionnels : `ruff check .` et `mypy .`, configurés dans `pyproject.toml`.
`mypy` signale encore des erreurs liées aux stubs incomplets de dolfinx/PETSc ; elles sont
préexistantes et ne bloquent pas l'exécution.
