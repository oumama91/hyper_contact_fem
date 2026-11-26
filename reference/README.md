# Références de non-régression

Ce dossier fige la solution du cas de référence `studies/O-ring_compression.yaml` : quart de
joint torique, maillage `HEX20`, formulation mixte $P_2$-$P_1$, $E = 6$ MPa, $\nu = 0{,}4999$,
`quadrature_degree: 4`, contact plan pénalisé sans frottement. Ces fichiers ne sont pas des
sorties de run : ils constituent la vérité de référence pour les tests et pour
`tools/compare_reference.py`.

## Environnement de génération

Toutes les valeurs ont été produites dans l'environnement conda `fenicsx-env` :
`dolfinx` 0.11.0, PETSc 3.25.6 avec MUMPS, `mpi4py` 4.1.2, `numpy` 2.5.3, `netCDF4` 1.7.4,
Python 3.12.

Les commandes passent par `tools/wsl_run.sh`, qui fixe par défaut
`OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = 1`. Ce réglage est indispensable :
MUMPS factorise via un BLAS multi-thread dont les réductions ne sont pas déterministes au
niveau $10^{-15}$ sur $|u|_{\max}$ et de l'ordre de $10^{-9}$ N sur les composantes de
réaction, ce qui rendrait la tolérance série $10^{-12}$ inatteignable. Avec un seul thread, les
runs série sont bit à bit identiques. La valeur reste surchargeable par l'appelant.

## Fichiers

| Fichier | Contenu |
|---|---|
| `O-ring_compression_series.log` | sortie console complète, série |
| `O-ring_compression_np2.log` | sortie console complète, `np = 2` |
| `O-ring_compression_series_reactions.csv` | historique des réactions, série |
| `O-ring_compression_np2_reactions.csv` | historique des réactions, `np = 2` |
| `O-ring_compression_series_metrics.json` | métriques haute précision, série |
| `O-ring_compression_np2_metrics.json` | métriques haute précision, `np = 2` |
| `O-ring_compression_series_fields.json` | statistiques des champs exportés, série |
| `O-ring_compression_np2_fields.json` | statistiques des champs exportés, `np = 2` |

Le résumé console arrondit $|u|_{\max}$ à 7 chiffres significatifs et `sigma_1_max` à 5 (formats
`.6e` et `.4e`) : les métriques en précision complète (repr JSON) sont donc produites par
`tools/capture_metrics.py`, qui rejoue exactement la séquence d'appels de `main._run` et expose
les mêmes appels de post-traitement que le run normal.

## Génération

Depuis la racine du dépôt, avec l'environnement activé :

```bash
# Métriques et champs, série
python tools/capture_metrics.py studies/O-ring_compression.yaml \
    --out reference/O-ring_compression_series_metrics.json \
    --fields-out reference/O-ring_compression_series_fields.json

# Métriques et champs, np = 2
mpiexec -n 2 python tools/capture_metrics.py studies/O-ring_compression.yaml \
    --out reference/O-ring_compression_np2_metrics.json \
    --fields-out reference/O-ring_compression_np2_fields.json
```

Les journaux `*.log` proviennent d'un run de `main.py` sur le même YAML (série et `mpiexec -n 2`)
et les CSV de réactions du fichier `results/<study_name>_reactions.csv` produit par ce run. Ces
runs doivent également passer par `tools/wsl_run.sh`, pour les threads BLAS.

Toute modification volontaire des sorties impose de régénérer ces fichiers : c'est le seul cas
où `reference/` est mis à jour.

## Valeurs de référence (précision complète)

| Grandeur | Série | `np = 2` |
|---|---|---|
| `u_max` [mm] | 4.664172049663108e-01 | 4.664172049663145e-01 |
| `sigma_1_max` [MPa] | 1.599963399298864e-01 | 1.599963399298864e-01 |
| `rel_res` | 4.564018773083237e-07 | 4.564018773114682e-07 |
| incréments | 1 | 1 |
| itérations Newton | 10 | 10 |
| cutbacks | 0 | 0 |
| `imbalance` | 9.950141512499176e-06 | 9.950141513645077e-06 |
| `total_Fz` [N] | 2.480275894546313e+01 | 2.480275894546309e+01 |

## Statistiques de champs

Les fichiers `*_fields.json` figent, pour chaque champ exporté, trois statistiques globales
calculées sur les valeurs nodales réellement écrites dans le `.bp` :

| Statistique | Définition |
|---|---|
| `min`, `max` | minimum / maximum sur **toutes les composantes nodales** du champ (composantes confondues) |
| `l2` | `sqrt(somme des valeurs²)` sur les nœuds possédés par chaque rang, puis réduite par `MPI.SUM` |

Les nœuds fantômes sont exclus (sinon ils seraient comptés deux fois). `components` rappelle
le nombre de composantes du champ, ce qui permet de détecter un changement de nature du champ.

`adios2` n'est pas installé dans `fenicsx-env` et `dolfinx` 0.11 n'expose pas de lecteur VTX :
les statistiques sont capturées pendant le run, en enveloppant `postproc.write_vtx` dans
`tools/capture_metrics.py`. Les valeurs exportées ne sont que copiées, jamais réinjectées dans
le calcul.

## Comparaison

```bash
python tools/compare_reference.py \
    --run <metrics-du-run>.json \
    --reference reference/O-ring_compression_series_metrics.json \
    --fields-run <fields-du-run>.json \
    --fields-reference reference/O-ring_compression_series_fields.json
```

La tolérance est choisie automatiquement d'après le champ `mpi_size` de la référence :
$10^{-12}$ en série, $10^{-8}$ en `np = 2` (l'ordre des réductions MPI n'est pas garanti bit à
bit). Les compteurs (incréments, itérations de Newton, cutbacks, évaluations de recherche
linéaire) sont comparés en égalité stricte. Le script affiche les écarts et renvoie
`VALIDATION RÉUSSIE` ou `VALIDATION ÉCHOUÉE`.

Les réactions ont des composantes exactement nulles par symétrie et d'autres de l'ordre de
`4e-9` N (bruit numérique) face à `Fz = 24,8` N. Un écart purement relatif composante par
composante n'aurait aucun sens : chaque composante est normalisée par `max(|référence|, plancher)`
avec `plancher = 1e-8 * max|composante de référence|` sur les réactions, la réaction totale et
les forces extérieures (soit `2,48e-7` N pour le cas O-ring). Un écart absolu de `1e-16` sur une
composante nulle est ainsi négligeable, alors qu'un écart sur la réaction portante reste
pénalisé.

Les statistiques de champs suivent la même règle, avec un plancher par champ :
`1e-8 * max(|min|, |max|, |l2|)` de la référence, car un minimum peut être nul ou négatif.

## Essai de PETSc.SNES

La variante SNES, les résultats comparés et la conclusion sont archivés dans
[`docs/decisions/snes-evaluation.md`](../docs/decisions/snes-evaluation.md). Le solveur retenu
reste la boucle de Newton avec recherche linéaire par backtracking de `solver.py`.

## Sorties de run

`results/` contient les sorties de chaque exécution (`.bp`, CSV de réactions) et est ignoré par
git : aucun fichier de `results/` n'est une référence. Le `.bp` n'est pas figé dans ce dossier ;
il est entièrement reproduit par le run, et les grandeurs utiles à la non-régression sont figées
sous forme de statistiques dans `*_fields.json`.
