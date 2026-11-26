# Fichier YAML, maillage et sorties

## Fichier YAML

Validation **Pydantic v2 stricte** : `extra="forbid"`, une clé inconnue est rejetée, les entiers
sont stricts (`StrictInt`), `boundaries` est non vide, sans nom dupliqué et avec au moins un
Dirichlet ; `solver` et `output` sont obligatoires, `physics` optionnel.

| Clé | Type | Défaut | Contrainte |
|---|---|---|---|
| `study_name` | chaîne | — | requis ; sert de nom aux sorties |
| `mesh.file` | chaîne | — | requis ; chemin du `.exo` |
| `mesh.order` | entier | 2 | $\ge 2$ (Taylor-Hood $P_k$-$P_{k-1}$) |
| `mesh.quadrature_degree` | entier | 4 | $\ge 1$ ; quadrature du volume |
| `material.E` | flottant | — | $> 0$ [MPa] |
| `material.nu` | flottant | — | $-1 < \nu \le 0{,}5$ |
| `material.rho` | flottant | 0.0 | $\ge 0$ [t/mm³] |
| `physics.gravity` | 3 flottants | `[0, 0, 0]` | [mm/s²] |
| `boundaries[].name` | chaîne | — | requis, unique, *sideset* existant |
| `boundaries[].type` | `"dirichlet"` / `"contact"` | — | requis (discriminant) |
| `boundaries[].probe_points`, `probe_tol` | liste de `[x,y,z]`, flottant | `[]`, 0.1 | contrôle de sideset [mm], tolérance $> 0$ |
| `boundaries[].components` | liste d'entiers | — | `dirichlet` ; non vide, distincts dans `{0,1,2}` |
| `boundaries[].value` | liste de flottants | — | `dirichlet` ; même longueur que `components` |
| `boundaries[].obstacle` | objet | — | `contact` ; requis, voir `obstacle.kind` |
| `boundaries[].epsilon_n`, `epsilon_t` | flottant, flottant ou `null` | —, `null` | `contact` ; $> 0$ [MPa/mm], défaut `epsilon_n` |
| `boundaries[].friction` | flottant | 0.0 | `contact` ; $\ge 0$, coefficient de Coulomb |
| `obstacle.kind` | `"plane"` / `"cylinder"` / `"sphere"` | — | requis (discriminant) |
| `plane.point`, `plane.normal` | 2 × `[x,y,z]` | — | requis ; `normal` non nulle |
| `cylinder.axis_point`, `axis_dir` | 2 × `[x,y,z]` | — | requis ; `axis_dir` non nulle |
| `cylinder.radius`, `sphere.radius` | flottant | — | $> 0$ [mm] |
| `sphere.center` | `[x,y,z]` | — | requis |
| `orientation` | `"outside"` / `"inside"` | `"outside"` | cylindre, sphère ; côté de la matière rigide |
| `solver.newton.increments` | entier | 1 | $\ge 1$ ; incréments nominaux |
| `solver.newton.max_iter` | entier | 25 | $\ge 1$ ; itérations par incrément |
| `solver.newton.rtol` | flottant | `1.0e-6` | $> 0$ ; résidu libre relatif |
| `solver.newton.atol` | flottant | `1.0e-10` | $\ge 0$ ; résidu libre absolu [N] |
| `solver.newton.max_cutbacks` | entier | 8 | $\ge 0$ ; réductions de pas successives |
| `output.fields` | `"all"` ou liste | `"all"` | sous-ensemble du catalogue des champs |
| `output.outdir` | chaîne | — | requis ; dossier de sortie |

`kind` et `type` sont les discriminants des unions : **obligatoires** même à leur valeur par défaut ;
`orientation` vaut `"outside"` (matière rigide à l'intérieur) ou `"inside"` (cavité rigide) ; la
section `newton` peut être omise. `displacement` est toujours exporté ; une liste `fields` suit l'ordre
`displacement`, `cauchy_stress`, `principal_stress`, `mean_stress`, `strain`, `principal_stretch`, `J`.

Exemple, cas de référence (quart de joint torique, trois symétries, un plan de contact) :

```yaml
study_name: "O-ring_compression"          # nom des sorties
mesh: {file: "../mesh/O-ring.exo", order: 2, quadrature_degree: 4}
material: {E: 6.0, nu: 0.4999, rho: 1.2e-9}   # [MPa], [-], [t/mm^3]
physics: {gravity: [0.0, 0.0, 0.0]}
boundaries:
  - {name: "symmetry_y", type: "dirichlet", components: [1], value: [0.0]}
  - {name: "symmetry_x", type: "dirichlet", components: [0], value: [0.0]}
  - {name: "symmetry_z", type: "dirichlet", components: [2], value: [0.0]}
  - {name: "compression", type: "contact",  # contact plan pénalisé
     obstacle: {kind: "plane", point: [0.0, 0.0, 1.8], normal: [0.0, 0.0, -1.0]},
     epsilon_n: 1.0e+3, epsilon_t: 1.0e+3, friction: 0.0}   # [MPa/mm], Coulomb
solver: {newton: {increments: 1, max_iter: 30, rtol: 1.0e-6}}
output: {fields: all, outdir: "../results"}
```

`studies/O-ring_compression_severe.yaml` abaisse le plan à `z = 0.6 mm` (1.4 mm d'écrasement) :
test de robustesse, hors des références figées.

## Maillage Exodus

Fichier Exodus II fourni par l'utilisateur. Le lecteur exige : **3D** (`num_dim = 3`), coordonnées en
**mm** ; **un seul bloc volumique** (`num_el_blk = 1`), `HEX8` (géométrie $P_1$) ou `HEX20` (serendipity
$P_2$, 20 nœuds) ; numérotation d'éléments identitaire (`elem_num_map` absent ou trivial) ; sidesets
**nommés** référençant des **faces de bord** (face interne refusée, deux sidesets ne peuvent partager
une facette) ; maillage conforme et déterminant strictement positif en chaque élément, calculé sur les
trois arêtes issues de son premier sommet (et non aux coins), contrôlé au chargement.

Les noms de `boundaries` sont ceux des sidesets ; un sideset inutilisé est ignoré (signalé en
`--verbose`). `probe_points` vérifie qu'un sideset est le bon : chaque point sonde doit être à moins
de `probe_tol` de la face taguée. Le fichier n'est lu et validé que sur le rang 0, puis
`dolfinx.mesh.create_mesh` reçoit la connectivité d'un seul rang : la dupliquer sur tous les rangs corrompt la mémoire de dolfinx 0.11.0 en parallèle.

## Sorties

`results/<study_name>_reactions.csv` : une ligne par incrément accepté, colonnes `lambda`, puis
`<frontière>_Fx/Fy/Fz` par condition de Dirichlet, puis `total_Fx/Fy/Fz` et `imbalance` (forces en N,
force exercée par l'appui sur la structure ; `imbalance` = bilan global relatif).

`results/<study_name>.bp` : export VTX (moteur `BP4`) lisible dans ParaView. Les champs sont projetés
aux nœuds par une projection $L^2$ à masse concentrée $P_1$, puis interpolés à l'ordre du maillage.
Catalogue (`output.fields`) : `displacement` (3, [mm]), `cauchy_stress` (9, [MPa]), `principal_stress`
(3, [MPa], $\sigma_1 \ge \sigma_2 \ge \sigma_3$), `mean_stress` (1, [MPa], $=p$), `strain` (9,
Green-Lagrange), `principal_stretch` (3, $\lambda_1 \ge \lambda_2 \ge \lambda_3$), `J` (1, $\det\mathbf F$).

Le résumé console (rang 0) donne les versions dolfinx/PETSc, l'étude, la loi et la formulation, le
nombre de processus MPI, le maillage (type, cellules, DDLs, ordres, nœuds, volume), les compteurs de
Newton, le solveur, les temps, le résidu final, la recherche linéaire, les réactions par frontière et
totales, le bilan, les chemins CSV et `.bp`, $|\mathbf u|_{\max}$ [mm] et `sigma_1_max` [MPa] (plus grande contrainte principale de Cauchy, projection $P_1$).
