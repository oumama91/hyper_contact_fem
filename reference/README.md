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

## Fichiers

| Fichier | Contenu |
|---|---|
| `O-ring_compression_series.log` | sortie console complète, série |
| `O-ring_compression_np2.log` | sortie console complète, np = 2 |
| `O-ring_compression_series_reactions.csv` | historique des réactions, série |
| `O-ring_compression_np2_reactions.csv` | historique des réactions, np = 2 |
| `O-ring_compression_series_metrics.json` | métriques haute précision, série |
| `O-ring_compression_np2_metrics.json` | métriques haute précision, np = 2 |

Seule modification apportée aux journaux bruts : la bannière d'environnement de `<OLD>`,
qui affichait le nom et la version du paquet de maillage CAO supprimé, a été réduite à
`dolfinx 0.11.0 / PETSc (3, 25, 6)` pour respecter la décision D5 (aucune occurrence de
ce nom dans le nouveau dépôt). Le contenu numérique des journaux est inchangé.

## Valeurs de référence (précision complète)

| Grandeur | Série | np = 2 |
|---|---|---|
| `u_max` [mm] | 4.664172049663098e-01 | 4.6641720496630973e-01 |
| `vm_max` [MPa] | 7.537310547729684e-01 | 7.537310547729688e-01 |
| `rel_res` | 4.564018773103674e-07 | 4.564018773119079e-07 |
| incréments | 1 | 1 |
| itérations Newton | 10 | 10 |
| cutbacks | 0 | 0 |
| `imbalance` | 9.950141512928887e-06 | 9.950141513000514e-06 |
| `total_Fz` [N] | 2.480275894546312e+01 | 2.4802758945463097e+01 |

## Comparaison

```bash
python tools/compare_reference.py \
    --run <metrics-du-run>.json \
    --reference reference/O-ring_compression_series_metrics.json
```

La tolérance est choisie automatiquement : `1e-12` si la référence est en série, `1e-8`
si elle est en np = 2.
