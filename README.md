# hyper_contact_fem

**Version actuelle : 1.0**

Solveur éléments finis 3D pour un **corps hyperélastique quasi-incompressible** en **contact
unilatéral pénalisé avec frottement de Coulomb** contre des **obstacles rigides analytiques**.
Maillage **Exodus II**, étude décrite par un fichier **YAML**, code **FEniCSx**.

**Sommaire** : [Présentation](#présentation) · [Modèle mécanique](#modèle-mécanique) ·
[Installation](#installation-et-utilisation) · [Fichier YAML](#fichier-yaml) ·
[Maillage](#maillage-exodus) · [Sorties](#sorties) · [Structure](#structure-du-dépôt) ·
[Validation](#validation) · [Limites](#limites-connues)

## Présentation

Une seule loi (néo-hookéenne), un seul corps déformable, un seul lecteur de maillage, un solveur direct : le projet vise les joints élastomères écrasés entre une pièce rigide et un plan de symétrie.

- **corps hyperélastique** néo-hookéen, grandes déformations, quasi-incompressible, formulation
  mixte $(\mathbf u,p)$ Taylor-Hood $P_k$-$P_{k-1}$ ($k \ge 2$) ;
- **contact unilatéral** pénalisé avec **frottement de Coulomb régularisé** (holonome) contre un
  plan, un cylindre infini ou une sphère analytiques, rigides et non maillés ;
- **pilotage de charge** $\lambda \in [0,1]$ avec cutbacks et recherche linéaire ;
- **maillage Exodus II** (`HEX8`/`HEX20`), un bloc volumique, sidesets nommés = frontières YAML ;
- **post-traitement** : réactions d'appui en CSV, export ParaView `BP4`, résumé console.

Il ne fait pas :

- contact **corps-à-corps** : un seul solide, l'obstacle est analytique ;
- chargement de **Neumann** (pression, effort surfacique) : seule la gravité est pilotée ;
- autre loi que le **néo-hookéen mixte** (ni viscoélasticité, ni plasticité) ;
- dynamique, génération ni adaptation de maillage.

## Modèle mécanique

Système cohérent **mm, t, s, N, MPa** (1 t accélérée à 1 mm/s² donne 1 N ; 1 N/mm² = 1 MPa) : longueurs en mm, temps en s, masse en t, contrainte en MPa, $\rho$ en t/mm³, gravité en mm/s², $\varepsilon$ en MPa/mm.

Avec $\mathbf F = \mathbf I + \nabla\mathbf u$, $J = \det\mathbf F$ et $\mathbf w = (\mathbf v,q)$ un champ test mixte :

$$\Pi(\mathbf u,p) = \int_{\Omega_0}\left[\psi_{\text{iso}}(\mathbf F) + p\,(J-1) - \frac{p^2}{2K}\right]\mathrm d\Omega_0 - W_{\text{ext}}(\mathbf v),\qquad \psi_{\text{iso}} = \frac{\mu}{2}\left(\bar I_1-3\right)$$

$$\bar I_1 = J^{-2/3}\,\mathrm{tr}(\mathbf F^T\mathbf F),\quad \mu = \frac{E}{2(1+\nu)},\quad K = \frac{E}{3(1-2\nu)}$$

La stationnarité en $p$ donne $p = K(J-1)$ : $p$ est une contrainte moyenne (positive en
traction), pas une pression. $E$ [MPa] et $\nu$ [-] sont les paramètres matériau ; $\nu = 0{,}5$
donne $1/K = 0$ (incompressibilité stricte), le cas de référence prend $\nu = 0{,}4999$.

$$\mathbf P = \frac{\partial\psi}{\partial\mathbf F} = \mathbf P_{\text{iso}} + p\,J\,\mathbf F^{-T},\quad \boldsymbol\sigma = \frac{1}{J}\mathbf P\mathbf F^T = \frac{1}{J}\mathbf P_{\text{iso}}\mathbf F^T + p\,\mathbf I,\quad \mathbf E = \tfrac{1}{2}(\mathbf F^T\mathbf F-\mathbf I)$$

$\mathbf P$ est le premier tenseur de Piola-Kirchhoff (PK1), $\boldsymbol\sigma$ la contrainte de
Cauchy, $\mathbf E$ la déformation de Green-Lagrange, $\lambda_i = \sqrt{1+2E_i}$ les allongements
principaux. La partie isochore de $\boldsymbol\sigma$ étant de trace nulle, $\mathrm{tr}(\boldsymbol\sigma)/3 = p$ (champ `mean_stress`).

La position courante est $\mathbf x = \mathbf X + \mathbf u$. Chaque obstacle fournit une jauge
signée $g_N(\mathbf x)$ et une normale sortante $\mathbf n$ (côté admissible) : $g_N > 0$ hors
contact, $g_N < 0$ en pénétration. La pression normale est pénalisée, $p_N = \varepsilon_n\langle -g_N\rangle$
avec $\langle a\rangle = \max(a,0)$ ; la traction tangentielle est un retour radial sur le cône de Coulomb :

$$\mathbf u_t = (\mathbf I-\mathbf n\otimes\mathbf n)\,\mathbf u,\quad \mathbf t_T = -\varepsilon_t\mathbf u_t\min\left(1,\frac{\mu_f\,p_N}{\lVert\varepsilon_t\mathbf u_t\rVert_\eta}\right),\quad \lVert\mathbf t\rVert_\eta = \sqrt{\mathbf t\cdot\mathbf t+\eta^2},\quad \eta = 10^{-9}\varepsilon_t$$

$$\mathbf t_c = p_N\,\mathbf n + \mathbf t_T,\quad W_{\text{ext}}(\mathbf v) = \lambda\int_{\Omega_0}\rho\,\mathbf g\cdot\mathbf v\,\mathrm d\Omega_0 + \int_{\Gamma_c}\mathbf t_c\cdot\mathbf v\,\mathrm ds$$

$\mu_f$ est le coefficient de frottement. La traction de contact n'est **pas** multipliée par
$\lambda$ (réaction de contrainte, active dès $g_N < 0$) ; l'intégrale porte sur la **facette de
référence** (surface non réactualisée) et le frottement holonome suppose un chargement monotone.

Le résidu et sa tangente viennent de la dérivation automatique UFL. $\lambda$ va de 0 à 1 par
incréments ($1/$`increments`) ; seuls Dirichlet et gravité sont multipliés par $\lambda$. Les
Dirichlet sont imposées par **relèvement** (`apply_lifting`), ce qui répartit le saut dans tout le
domaine. La convergence porte sur le résidu **libre** $r_k \le \max(r_{\text{tol}}\,r_{\text{ref}}, a_{\text{tol}})$,
où $r_{\text{ref}}$ est la norme du premier résidu libre de l'incrément. Chaque correction est
essayée avec un pas $\alpha = 1, 1/2, \ldots, 2^{-6}$, accepté si le résidu libre décroît
strictement ; sinon l'état est restauré et le pas réduit de moitié, l'incrément étant retenté au
plus `max_cutbacks` fois. Après deux incréments faciles, le pas est agrandi d'un facteur 1,5
jusqu'au pas nominal. La tangente étant de type point-selle indéfinie, le solveur est un **LU
direct MUMPS** (`preonly`/`lu`/`mumps`).

## Installation et utilisation

Environnement de référence : Linux (natif ou WSL) et **conda-forge**. Le projet n'est pas empaqueté :
les modules Python sont lancés depuis la racine du dépôt. Versions utilisées pour les références :
Python 3.12.13, `fenics-dolfinx` 0.11.0, PETSc 3.25.6 **avec MUMPS** (indispensable à la
factorisation directe de la tangente mixte), `petsc4py` 3.25.6, `mpi4py` 4.1.2, `numpy` 2.5.3,
`netCDF4` 1.7.4, `pydantic` 2.14.1, `pyyaml` 6.0.3 et `pytest` 9.1.1. La configuration PETSc
installée doit inclure MUMPS.

Exemple de création de l'environnement de référence (les versions doivent être disponibles sur
la plateforme conda-forge utilisée) :

```bash
conda create -n fenicsx-env -c conda-forge \
  python=3.12 fenics-dolfinx=0.11.0 petsc4py=3.25.6 mpi4py=4.1.2 \
  numpy=2.5.3 netcdf4=1.7.4 pydantic=2.14.1 pyyaml=6.0.3 pytest=9.1.1
conda activate fenicsx-env
cd <chemin-vers-le-clone>
python main.py studies/O-ring_compression.yaml                 # série
mpiexec -n 2 python main.py studies/O-ring_compression.yaml    # parallèle
python -m pytest tests                                         # tests
```

Contrôles statiques optionnels : `ruff check .` (`ruff.toml`) et `mypy .` (`mypy.ini`).

Sous Windows, FEniCSx et le `mpiexec` PETSc s'exécutent dans WSL. Depuis la racine du dépôt côté
Linux/WSL, le wrapper `tools/wsl_run.sh` active conda et fixe par défaut
`OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = 1` (une valeur déjà fixée par
l'appelant reste prioritaire). Il recherche conda sous `$HOME/miniconda3` ; définir `CONDA_BASE`
si conda est installé ailleurs :

```bash
bash tools/wsl_run.sh python main.py studies/O-ring_compression.yaml
```

Depuis Windows, lancer cette commande avec WSL en positionnant le répertoire courant à la racine
du clone dans le système de fichiers Linux.

`output.outdir` est obligatoire ; `mesh.file` et `output.outdir` sont relatifs au YAML.

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

`studies/O-ring_compression_severe.yaml` abaisse le plan à `z = 0.6 mm` (1.4 mm d'écrasement) : test de robustesse, hors des références figées.

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

## Structure du dépôt

```text
.
├── config.py, contact.py, forms.py, material.py, solver.py, ...  # solveur et modules Python
├── mesh/                                                        # maillage Exodus II fourni
├── studies/                                                     # cas YAML exécutables
├── tests/                                                       # tests pytest, dont le cas golden
├── tools/                                                       # scripts de validation et wrapper WSL
├── reference/                                                   # sorties numériques de référence
├── results/                                                     # résultats générés, ignorés par Git
├── README.md
├── VERSION                                                     # version publiée du projet
├── LICENSE
├── .gitignore
├── ruff.toml
└── mypy.ini
```

Les modules sont actuellement à la racine (pas de répertoire `src/`) : cette disposition est
intentionnelle pour préserver les imports et le lancement depuis le clone. Les fichiers de maillage,
d'études et de référence sont des entrées/tests du projet et sont conservés dans Git ; seuls les
résultats régénérables sont exclus.

## Architecture du code

| Fichier | Rôle |
|---|---|
| `main.py` | CLI (`study`, `--verbose`), enchaînement du calcul, résumé console |
| `config.py` | modèles Pydantic v2 du YAML, obstacles, catalogue des champs exportables |
| `exodus_reader.py` | lecture bas niveau du fichier Exodus II (`netCDF4`) : points, connectivité, sidesets |
| `exodus_mesh.py` | permutations Exodus → basix, maillage et `MeshTags` dolfinx, motif MPI rang 0, contrôles |
| `material.py` | cinématique ($\mathbf F$, $\mathbf E$, $J$), loi néo-hookéenne mixte, PK1 et Cauchy |
| `contact.py` | jauge et normale des obstacles, pénalité normale, Coulomb régularisé |
| `forms.py` | espace Taylor-Hood, résidu et tangente UFL, assemblage des frontières |
| `dirichlet.py` | localisation des DDL de Dirichlet, valeurs multipliées par $\lambda$ |
| `solver.py` | pilotage de charge, Newton avec backtracking, solveur LU MUMPS, métriques |
| `reactions.py` | réactions par frontière et historique CSV |
| `postproc.py` | champs aux points de quadrature, valeurs principales, métriques, export VTX |
| `projection.py` | projection $L^2$ à masse concentrée $P_1$ et espaces d'export |
| `utils.py` | journalisation, vecteur PETSc partagé, intégrales, tailles de DDL, assemblage |

`studies/` contient les études YAML, `mesh/` le maillage Exodus, `reference/` les références figées,
`tools/` les scripts de capture/comparaison et le wrapper WSL, `tests/` la suite pytest. Les erreurs
attendues (YAML invalide, maillage introuvable, non-convergence) s'affichent sans traceback. Flux de `main.py` :

1. `argparse` et journalisation (rang 0) ; `config.load_config` : validation Pydantic et résolution des chemins relatifs au YAML.
2. `exodus_mesh.build_mesh` : lecture et validation rang 0, distribution MPI, tags de facettes.
3. `forms.build_forms` puis `dirichlet.apply_bcs` : formulation mixte, résidu, tangente, DDL.
4. `solver.solve_newton` avec `reactions.ReactionHistory` : incréments, Newton, cutbacks.
5. `postproc.postprocess`, `write_csv`, `print_summary` : projection, métriques, VTX, CSV.

## Validation

Cas de référence `studies/O-ring_compression.yaml` (`HEX20`, $E = 6$ MPa, $\nu = 0{,}4999$, $P_2$-$P_1$)
en série : 1 incrément, 10 itérations de Newton, 0 cutback, `u_max` = 4.664172049663e-01 mm,
`sigma_1_max` = 1.599963399299e-01 MPa, `total_Fz` = 2.480275894546e+01 N. `reference/` fige journaux,
CSV, métriques haute précision et statistiques de champs pour la série et `np = 2` ; `reference/README.md` en détaille la génération, la normalisation et l'essai SNES.

```bash
python tools/capture_metrics.py studies/O-ring_compression.yaml --out results/series_metrics.json --fields-out results/series_fields.json
python tools/compare_reference.py --run results/series_metrics.json --reference reference/O-ring_compression_series_metrics.json --fields-run results/series_fields.json --fields-reference reference/O-ring_compression_series_fields.json
```

Pour `np = 2`, préfixer **uniquement** `capture_metrics.py` par `mpiexec -n 2` et remplacer `series` par
`np2` ; `compare_reference.py` reste série. Tolérances : écart relatif $\le 10^{-12}$ en série (attendu
0), $\le 10^{-8}$ pour `np = 2`, compteurs en égalité stricte ; la reproductibilité série exige les threads BLAS à 1 fixés par `tools/wsl_run.sh`.

`tests/` contient 118 cas de test (durée dépendante de l'environnement, le test golden exécute un solveur complet) : `test_config.py` (schéma strict, bornes), `test_contact.py`
(jauge, normale, pénalité, frottement), `test_exodus.py` (permutations `HEX8`/`HEX20`, sidesets, erreurs) et `test_oring_golden.py` (run complet comparé à `reference/`).

## Limites connues

- **Un seul corps** : contact unilatéral contre un obstacle analytique, pas de corps-à-corps.
- **Pénalité sans multiplicateurs** : non-pénétration approchée, pénétration $\propto 1/\varepsilon_n$.
- **Frottement holonome** : chargement monotone seulement ; un cycle exigerait une variable d'état.
- **Indicateur** : `sigma_1_max` est une traction maximale ; la compression minimale serait plus parlante.
- **Axe du cylindre et centre de la sphère** : la normale divise par la distance radiale (`d_perp/r` au
  cylindre, `d/r` à la sphère) ; en `r = 0` elle est indéfinie, `contact.py` ne protège pas cette
  division et ce cas n'est pas traité.
- **Solveur et chargement** : LU MUMPS (tailles modérées) ; $\lambda$ ne multiplie que gravité et
  Dirichlet, donc un cas porté par le seul contact est indépendant de $\lambda$.
