# Références de non-régression — Étape 1

Toutes les références ont été générées avec le code de `<OLD>` (`D:\Recherche\FEM`, en
lecture seule), sans modification de son arborescence ni de son dossier `results/` : le
fichier YAML de lancement a été copié dans un dossier temporaire et seuls `mesh.file` et
`output.outdir` y ont été rendus absolus (vers le maillage de `<OLD>` et un dossier
temporaire de sortie).

Cas de référence : `studies/O-ring_compression.yaml` (HEX20, P2-P1, E = 6 MPa,
nu = 0,4999, `quadrature_degree: 4`, contact plan pénalisé sans frottement).

## Commandes

Depuis `<OLD>`, avec le wrapper `tools/wsl_run.sh` du nouveau projet :

```bash
# Série
python main.py <yaml-temporaire-serie>
# np = 2
mpiexec -n 2 python main.py <yaml-temporaire-np2>
```

Les métriques en précision complète (le résumé console arrondit |u|max à 7 chiffres et
vm_max à 5) sont produites par `tools/capture_metrics.py`, avec la même séquence d'appels
que `main._run` :

```bash
python tools/capture_metrics.py <yaml-temporaire> --out reference/....json
mpiexec -n 2 python tools/capture_metrics.py <yaml-temporaire> --out reference/....json
```

Toutes les commandes passent par `tools/wsl_run.sh`, qui fixe par défaut
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`. Sans ce réglage, la factorisation
MUMPS via un BLAS multi-thread n'est pas reproductible : deux exécutions du **même** code
OLD divergent déjà de ~1e-15 sur |u|max et sur les composantes de réaction de l'ordre de
1e-9 N, ce qui rend la tolérance série 1e-12 inatteignable. Un thread par défaut rend les
runs série bit à bit identiques ; la valeur reste surchargeable par l'appelant.

## Fichiers

| Fichier | Contenu |
|---|---|
| `O-ring_compression_series.log` | sortie console complète, série |
| `O-ring_compression_np2.log` | sortie console complète, np = 2 |
| `O-ring_compression_series_reactions.csv` | historique des réactions, série |
| `O-ring_compression_np2_reactions.csv` | historique des réactions, np = 2 |
| `O-ring_compression_series_metrics.json` | métriques haute précision, série |
| `O-ring_compression_np2_metrics.json` | métriques haute précision, np = 2 |
| `O-ring_compression_series_fields.json` | statistiques des champs exportés, série (Étape 2.0) |
| `O-ring_compression_np2_fields.json` | statistiques des champs exportés, np = 2 (Étape 2.0) |

Seule modification apportée aux journaux bruts : la bannière d'environnement de `<OLD>`,
qui affichait le nom et la version du paquet de maillage CAO supprimé, a été réduite à
`dolfinx 0.11.0 / PETSc (3, 25, 6)` pour respecter la décision D5 (aucune occurrence de
ce nom dans le nouveau dépôt). Le contenu numérique des journaux est inchangé.

## Valeurs de référence (précision complète)

| Grandeur | Série | np = 2 |
|---|---|---|
| `u_max` [mm] | 4.664172049663108e-01 | 4.664172049663042e-01 |
| `vm_max` [MPa] | 7.537310547729685e-01 | 7.537310547729683e-01 |
| `rel_res` | 4.564018773083237e-07 | 4.564018773111702e-07 |
| incréments | 1 | 1 |
| itérations Newton | 10 | 10 |
| cutbacks | 0 | 0 |
| `imbalance` | 9.950141512499176e-06 | 9.950141513215369e-06 |
| `total_Fz` [N] | 2.480275894546313e+01 | 2.480275894546309e+01 |

## Statistiques de champs (Étape 2.0)

Les fichiers `*_fields.json` figent, pour chaque champ exporté, trois statistiques
globales calculées sur les valeurs nodales réellement écrites dans le `.bp` :

| Statistique | Définition |
|---|---|
| `min`, `max` | minimum / maximum sur **toutes les composantes nodales** du champ (composantes confondues) |
| `l2` | `sqrt(somme des valeurs²)` sur les nœuds possédés par chaque rang, puis réduite par `MPI.SUM` |

Les nœuds fantômes sont exclus (sinon ils seraient comptés deux fois). `components` rappelle
le nombre de composantes du champ, ce qui permet de détecter un changement de nature du champ.

`adios2` n'est pas installé dans `fenicsx-env` et `dolfinx` 0.11 n'expose pas de lecteur VTX :
les statistiques sont capturées pendant le run, en enveloppant
`postproc.postproc.write_vtx` dans `tools/capture_metrics.py`. Les valeurs exportées ne sont
que copiées, jamais réinjectées dans le calcul : les sorties numériques sont inchangées
(série bit à bit identique au tag `step1-validated`).

Génération depuis le tag `step1-validated` (le code de production de l'Étape 2.0 est
identique à celui du tag) :

```bash
python tools/capture_metrics.py studies/O-ring_compression.yaml \
    --out reference/O-ring_compression_series_metrics.json \
    --fields-out reference/O-ring_compression_series_fields.json
mpiexec -n 2 python tools/capture_metrics.py studies/O-ring_compression.yaml \
    --out reference/O-ring_compression_np2_metrics.json \
    --fields-out reference/O-ring_compression_np2_fields.json
```

Les métriques régénérées sont bit à bit identiques aux références de l'Étape 1.

## Normalisation des écarts

Les réactions ont des composantes exactement nulles par symétrie et d'autres de l'ordre de
`4e-9` N (bruit numérique) face à `Fz = 24,8` N. Un écart purement relatif composante par
composante n'aurait aucun sens : `tools/compare_reference.py` normalise donc chaque
composante par `max(|référence|, plancher)` avec
`plancher = 1e-8 * max|composante de référence|` sur les réactions, la réaction totale et les
forces extérieures (soit `2,48e-7` N pour le cas O-ring). Un écart absolu de `1e-16` sur une
composante nulle est ainsi négligeable, alors qu'un écart sur la réaction portante reste
pénalisé.

Les statistiques de champs suivent la même règle, avec un plancher par champ :
`1e-8 * max(|min|, |max|, |l2|)` de la référence, car un minimum peut être nul ou négatif.

## Sorties de run

`results/` contient les sorties de chaque exécution (`.bp`, CSV de réactions) et est ignoré
par git depuis l'Étape 2.0 : aucun fichier de `results/` n'est une référence. Le `.bp` qui y
figurait avant l'Étape 2.0 n'a pas été copié dans `reference/` : il est entièrement
reproduit par le run et les grandeurs utiles à la non-régression sont désormais figées sous
forme de statistiques dans `*_fields.json`.

## Comparaison

```bash
python tools/compare_reference.py \
    --run <metrics-du-run>.json \
    --reference reference/O-ring_compression_series_metrics.json \
    --fields-run <fields-du-run>.json \
    --fields-reference reference/O-ring_compression_series_fields.json
```

La tolérance est choisie automatiquement : `1e-12` si la référence est en série, `1e-8`
si elle est en np = 2.
