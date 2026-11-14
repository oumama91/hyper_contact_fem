# hyper_contact_fem

Solveur éléments finis 3D pour un **corps hyperélastique quasi-incompressible** en **contact
unilatéral pénalisé avec frottement de Coulomb** contre des **obstacles rigides analytiques**.
Le maillage est lu au format **Exodus II** ; les études sont décrites par un fichier **YAML**.

Le projet est volontairement minimal : une seule loi de comportement (néo-hookéenne), un seul
corps déformable, un seul lecteur de maillage, un solveur direct. Il est conçu pour des joints
élastomères écrasés entre une pièce rigide et un plan de symétrie.

## Table des matières

- [Présentation](#présentation)
- [Modèle mécanique](#modèle-mécanique)
- [Prérequis et environnement](#prérequis-et-environnement)
- [Utilisation](#utilisation)
- [Fichier YAML](#fichier-yaml)
- [Maillage](#maillage)
- [Sorties](#sorties)
- [Architecture](#architecture)
- [Validation](#validation)
- [Limites connues et pistes](#limites-connues-et-pistes)

## Présentation

### Ce que fait le projet

- **Un corps hyperélastique** néo-hookéen, grandes déformations, quasi-incompressible, en
  **formulation mixte $(\mathbf u, p)$ Taylor-Hood $P_k$-$P_{k-1}$** ($k \ge 2$).
- **Contact unilatéral** avec une pénalité normale, et **frottement de Coulomb régularisé**
  (formulation holonome), contre un **plan**, un **cylindre infini** ou une **sphère**
  analytiques. L'obstacle est rigide : il n'est pas maillé et n'a pas de champ mécanique.
- **Pilotage de charge** proportionnel $\lambda \in [0,1]$ avec **réductions de pas
  automatiques** (*cutback*) et recherche linéaire par backtracking.
- **Maillage Exodus II** (`HEX8` ou `HEX20`), un seul bloc volumique, avec des *sidesets*
  nommés qui deviennent les frontières du YAML.
- **Post-traitement** : réactions d'appui en CSV, export ParaView (`VTX`/`BP4`), résumé
  console.

### Ce que le projet ne fait pas

- Pas de contact **corps-à-corps** : un seul solide déformable, l'autre est un obstacle
  analytique rigide.
- Pas de chargement de **Neumann** (pression ou effort surfacique imposé) ; seule la gravité
  est une charge extérieure pilotée.
- Pas d'autre loi que le **néo-hookéen mixte** (pas d'élasticité linéaire, pas de loi
  viscoélastique, pas de plasticité).
- Pas de **génération de maillage** : le fichier Exodus est fourni par l'utilisateur.
- Pas de dynamique, pas de contact cyclique avec variable d'état, pas de multiplicateurs de
  Lagrange.

## Modèle mécanique

### Unités

Le système est cohérent : **mm, t, s, N, MPa**. Une masse de 1 t accélérée à 1 mm/s² produit
1 N, et 1 N/mm² vaut 1 MPa.

| Grandeur | Unité |
|---|---|
| longueur, déplacement, ouverture | mm |
| temps | s |
| masse | t |
| force | N |
| contrainte, pression moyenne $p$ | MPa = N/mm² |
| densité $\rho$ | t/mm³ |
| gravité $\mathbf g$ | mm/s² |
| pénalité de contact $\varepsilon_n$, $\varepsilon_t$ | MPa/mm = N/mm³ |

### Cinématique et énergie potentielle

Avec $\mathbf F = \mathbf I + \nabla \mathbf u$ (gradient par rapport à la configuration de
référence), $J = \det \mathbf F$ et $\mathbf w = (\mathbf v, q)$ un champ test mixte :

$$
\Pi(\mathbf u,p) = \int_{\Omega_0}
\left[\psi_{\text{iso}}(\mathbf F) + p\,(J-1) - \frac{p^2}{2K}\right] \mathrm d\Omega_0
- W_{\text{ext}}(\mathbf v).
$$

Le terme $p\,(J-1) - p^2/(2K)$ remplace la partie volumique de la loi ; la stationnarité en
$p$ donne $p = K(J-1)$.

### Loi néo-hookéenne

$$
\psi_{\text{iso}} = \frac{\mu}{2}\left(\bar I_1 - 3\right), \qquad
\bar I_1 = J^{-2/3}\,\mathrm{tr}\left(\mathbf F^T\mathbf F\right),
$$

$$
\mu = \frac{E}{2(1+\nu)}, \qquad K = \frac{E}{3(1-2\nu)}.
$$

$E$ est le module d'Young [MPa], $\nu$ le coefficient de Poisson [-] et $\mu$ le module de
cisaillement [MPa]. Pour $\nu = 0{,}5$, $1/K = 0$ : incompressibilité stricte. Le cas de
référence utilise $\nu = 0{,}4999$.

### Contraintes et déformations

Premier tenseur de Piola-Kirchhoff (PK1) :

$$
\mathbf P = \frac{\partial \psi}{\partial \mathbf F}
= \mathbf P_{\text{iso}} + p\,J\,\mathbf F^{-T}.
$$

Contrainte de Cauchy (post-traitement) :

$$
\boldsymbol\sigma = \frac{1}{J}\mathbf P\mathbf F^T
= \frac{1}{J}\mathbf P_{\text{iso}}\mathbf F^T + p\,\mathbf I.
$$

Déformation de Green-Lagrange et allongements principaux :

$$
\mathbf E = \frac{1}{2}\left(\mathbf F^T\mathbf F - \mathbf I\right), \qquad
\lambda_i = \sqrt{1 + 2 E_i}.
$$

La contrainte moyenne exportée est $\mathrm{tr}(\boldsymbol\sigma)/3$ : la partie isochore de
$\boldsymbol\sigma$ est de trace nulle, donc cette moyenne vaut $p$. $p$ est une contrainte
moyenne (positive en traction), pas une pression au sens usuel.

### Contact unilatéral et frottement

La position courante est $\mathbf x = \mathbf X + \mathbf u$. Chaque obstacle fournit une
jauge signée $g_N(\mathbf x)$ et une normale $\mathbf n(\mathbf x)$ qui pointe du côté
admissible (hors de la matière rigide) : $g_N > 0$ hors contact, $g_N < 0$ en pénétration.

Pour un plan de point $\mathbf x_0$ et de normale $\mathbf n_0$ :
$g_N = \mathbf n_0\cdot(\mathbf x-\mathbf x_0)$, $\mathbf n = \mathbf n_0$.
Pour un cylindre d'axe $(\mathbf x_0,\mathbf a)$ et de rayon $R$ :
$\mathbf d_\perp = \mathbf d - (\mathbf d\cdot\mathbf a)\mathbf a$ avec $\mathbf d=\mathbf x-\mathbf x_0$,
$r=\lVert\mathbf d_\perp\rVert$, $g_N = s\,(r-R)$, $\mathbf n = s\,\mathbf d_\perp/r$.
Pour une sphère de centre $\mathbf c$ et de rayon $R$ :
$r=\lVert\mathbf x-\mathbf c\rVert$, $g_N = s\,(r-R)$, $\mathbf n = s\,(\mathbf x-\mathbf c)/r$.
Les vecteurs $\mathbf n_0$ et $\mathbf a$ sont normalisés en interne ; seul leur sens compte.
Le signe $s=+1$ correspond à `outside` (matière rigide à l'intérieur) et $s=-1$ à `inside`
(cavité rigide à l'extérieur).

Pression normale par pénalité (crochet de Macaulay) :

$$
p_N = \varepsilon_n\,\langle -g_N\rangle, \qquad \langle a\rangle = \max(a,0).
$$

Frottement de Coulomb holonome et régularisé. Le glissement est le déplacement total projeté
sur le plan tangent, $\mathbf u_t = (\mathbf I-\mathbf n\otimes\mathbf n)\,\mathbf u$, et la
traction d'essai est $\mathbf t_{\text{trial}} = -\varepsilon_t\,\mathbf u_t$. Le retour radial
sur le cône de Coulomb s'écrit

$$
\mathbf t_T = \mathbf t_{\text{trial}}\,
\min\left(1, \frac{\mu_f\,p_N}{\lVert\mathbf t_{\text{trial}}\rVert_\eta}\right),
\qquad
\lVert\mathbf t\rVert_\eta = \sqrt{\mathbf t\cdot\mathbf t + \eta^2},
$$

avec $\mu_f$ le coefficient de frottement et $\eta = 10^{-9}\varepsilon_t$ une régularisation
de la norme. La traction de contact exercée par l'obstacle sur le solide est

$$
\mathbf t_c = p_N\,\mathbf n + \mathbf t_T,
\qquad
W_{\text{ext}}(\mathbf v) = \lambda\int_{\Omega_0}\rho\,\mathbf g\cdot\mathbf v\,\mathrm d\Omega_0
+ \int_{\Gamma_c}\mathbf t_c\cdot\mathbf v\,\mathrm ds.
$$

La traction de contact n'est **pas** multipliée par $\lambda$ : c'est une réaction de
contrainte, active dès que $g_N < 0$. Les intégrales de contact sont prises sur la **facette
de référence** (`ds` sur le maillage de référence) : la surface de contact n'est pas
réactualisée en configuration courante. Le frottement holonome est valable pour un chargement
monotone, sans inversion du sens de glissement.

### Pilotage de charge, Newton et cutback

Le résidu et sa tangente sont obtenus par dérivation automatique :

$$
R(\mathbf u,p;\mathbf w) = D\Pi(\mathbf u,p)[\mathbf w] - W_{\text{ext}}(\mathbf v),
\qquad
\mathbf J = \frac{\partial R}{\partial(\mathbf u,p)}.
$$

Le chargement est découpé en incréments nominaux ($1/$`increments`) et $\lambda$ va de 0 à 1.
Les conditions de Dirichlet et la gravité sont multipliées par $\lambda$ ; le contact ne l'est
pas.

À chaque incrément, les conditions de Dirichlet sont imposées par **relèvement**
(`apply_lifting`) : la première correction répartit le saut imposé dans tout le domaine, ce qui
évite de concentrer la déformation dans la première couche d'éléments. La convergence porte sur
la norme du **résidu libre** (degrés de liberté de Dirichlet exclus) :

$$
r_k \le \max\left(r_{\text{tol}}\,r_{\text{ref}},\ a_{\text{tol}}\right),
$$

où $r_{\text{ref}}$ est la norme du résidu libre de la première itération de l'incrément.
Chaque correction $\delta\mathbf u$ est essayée avec un pas $\alpha = 1, 1/2, 1/4, \ldots$
(jusqu'à $2^{-6}$) et n'est acceptée que si la norme du résidu libre décroît strictement.
Si aucun pas ne convient, l'incrément échoue : l'état est restauré, le pas de charge est
multiplié par $0{,}5$, et l'incrément est retenté (au plus `max_cutbacks` fois). Après deux
incréments « faciles » consécutifs (au plus 4 itérations), le pas est agrandi d'un facteur 1,5
jusqu'au pas nominal.

Le solveur linéaire est un **LU direct MUMPS** (`ksp_type preonly`, `pc_type lu`,
`pc_factor_mat_solver_type mumps`) : la tangente mixte $(\mathbf u,p)$ est de type point-selle
indéfinie, un solveur direct est donc nécessaire.

## Prérequis et environnement

Le projet s'exécute sous **Ubuntu (WSL)** dans un environnement **conda** nommé
`fenicsx-env`. Il n'y a ni `setup.py` ni `pyproject.toml` : les modules du projet ne sont pas
installés, les commandes se lancent donc depuis la racine du dépôt (le dossier courant est
alors dans `sys.path`). Seuls les scripts de `tools/` ajoutent eux-mêmes la racine à
`sys.path`, ce qui permet de les lancer par leur chemin depuis n'importe quel dossier.

Versions de l'environnement de référence :

| Paquet | Version |
|---|---|
| Python | 3.12.13 |
| `fenics-dolfinx` | 0.11.0 |
| PETSc | 3.25.6 (avec MUMPS) |
| `petsc4py` | 3.25.6 |
| `mpi4py` | 4.1.2 |
| `numpy` | 2.5.3 |
| `netCDF4` | 1.7.4 |
| `pydantic` | 2.14.1 |
| `pyyaml` | 6.0.3 |
| `pytest` | 9.1.1 |

MUMPS doit être disponible dans PETSc, car la tangente mixte est résolue par factorisation
directe. Sur une nouvelle machine, créer un environnement conda depuis `conda-forge`
contenant ces paquets (mêmes versions, PETSc compilé avec MUMPS), puis l'activer :

```bash
conda activate fenicsx-env
```

Sous Windows, l'environnement `fenicsx-env` et le `mpiexec` associé à PETSc ne sont pas
disponibles : les commandes de calcul passent par le wrapper `tools/wsl_run.sh`, qui source
conda, active `fenicsx-env` et fixe par défaut
`OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = 1` (une valeur déjà définie par
l'appelant reste prioritaire).

## Utilisation

Depuis Ubuntu/WSL, environnement activé, à la racine du dépôt :

```bash
conda activate fenicsx-env
cd /mnt/d/Recherche/hyper_contact_fem
python main.py studies/O-ring_compression.yaml
```

En parallèle MPI :

```bash
mpiexec -n 2 python main.py studies/O-ring_compression.yaml
```

Tests :

```bash
python -m pytest tests
```

Depuis Windows, avec le wrapper (une seule ligne par commande) :

```bash
wsl.exe --cd /mnt/d/Recherche/hyper_contact_fem -e bash tools/wsl_run.sh python main.py studies/O-ring_compression.yaml
wsl.exe --cd /mnt/d/Recherche/hyper_contact_fem -e bash tools/wsl_run.sh mpiexec -n 2 python main.py studies/O-ring_compression.yaml
wsl.exe --cd /mnt/d/Recherche/hyper_contact_fem -e bash tools/wsl_run.sh python -m pytest tests
```

Les sorties sont écrites dans `output.outdir`, clé **obligatoire** (le cas de référence
utilise `"../results"`, résolu en `results/` à la racine du dépôt, dossier ignoré par git).
Les chemins `mesh.file` et `output.outdir` sont relatifs au dossier du fichier YAML.

## Fichier YAML

### Règles générales

- Validation **Pydantic v2 stricte** : `extra="forbid"` partout, une clé inconnue est une
  erreur, jamais ignorée.
- Les entiers sont stricts (`StrictInt`) : ni booléen, ni chaîne `"20"`.
- `boundaries` est une **liste non vide**, avec au moins une condition de Dirichlet et des
  noms uniques. Chaque nom doit être un *sideset* du maillage Exodus.
- Les frontières sont une union discriminée sur `type` ; les obstacles une union discriminée
  sur `kind`.
- `solver` et `output` sont obligatoires ; `physics` est optionnel.

### Schéma complet

Racine :

| Clé | Type | Défaut | Contrainte |
|---|---|---|---|
| `study_name` | chaîne | — | requis ; sert de nom aux sorties |
| `mesh` | objet | — | requis |
| `material` | objet | — | requis |
| `physics` | objet | `{gravity: [0,0,0]}` | optionnel |
| `boundaries` | liste | — | requis, non vide, au moins un Dirichlet, noms uniques |
| `solver` | objet | — | requis |
| `output` | objet | — | requis |

`mesh` :

| Clé | Type | Défaut | Contrainte |
|---|---|---|---|
| `file` | chaîne | — | requis ; chemin du `.exo` |
| `order` | entier | 2 | $\ge 2$ (Taylor-Hood $P_k$-$P_{k-1}$) |
| `quadrature_degree` | entier | 4 | $\ge 1$ ; degré de quadrature du volume |

`material` :

| Clé | Type | Défaut | Contrainte |
|---|---|---|---|
| `E` | flottant | — | $> 0$ [MPa] |
| `nu` | flottant | — | $-1 < \nu \le 0{,}5$ |
| `rho` | flottant | 0.0 | $\ge 0$ [t/mm³] |

`physics` :

| Clé | Type | Défaut | Contrainte |
|---|---|---|---|
| `gravity` | liste de 3 flottants | `[0.0, 0.0, 0.0]` | exactement 3 composantes [mm/s²] |

`boundaries` — clés communes :

| Clé | Type | Défaut | Contrainte |
|---|---|---|---|
| `name` | chaîne | — | requis, unique, *sideset* existant |
| `type` | `"dirichlet"` ou `"contact"` | — | requis (discriminant) |
| `probe_points` | liste de points `[x,y,z]` | `[]` | optionnel ; contrôle de sideset [mm] |
| `probe_tol` | flottant | 0.1 | $> 0$ [mm] |

`boundaries` de type `dirichlet` :

| Clé | Type | Défaut | Contrainte |
|---|---|---|---|
| `components` | liste d'entiers | — | non vide, valeurs distinctes dans `{0,1,2}` |
| `value` | liste de flottants | — | même longueur que `components` |

`boundaries` de type `contact` :

| Clé | Type | Défaut | Contrainte |
|---|---|---|---|
| `obstacle` | objet | — | requis (voir ci-dessous) |
| `epsilon_n` | flottant | — | $> 0$ [MPa/mm] |
| `epsilon_t` | flottant ou `null` | `null` | si absent, vaut `epsilon_n` ; $> 0$ [MPa/mm] |
| `friction` | flottant | 0.0 | $\ge 0$ ; coefficient de Coulomb |

Obstacles (`obstacle.kind` discrimine) :

| `kind` | Clés | Défaut | Contrainte |
|---|---|---|---|
| `"plane"` | `point` `[x,y,z]`, `normal` `[x,y,z]` | — | requis ; `normal` non nulle [mm] |
| `"cylinder"` | `axis_point` `[x,y,z]`, `axis_dir` `[x,y,z]`, `radius`, `orientation` | `orientation: "outside"` | requis ; `axis_dir` non nulle, `radius > 0` [mm] |
| `"sphere"` | `center` `[x,y,z]`, `radius`, `orientation` | `orientation: "outside"` | requis ; `radius > 0` [mm] |

`kind` et `type` sont les discriminants des unions : ils sont **obligatoires** dans le YAML,
même lorsque leur valeur est la valeur par défaut du modèle. `orientation` vaut `"outside"`
(matière rigide à l'intérieur de la forme) ou `"inside"` (cavité rigide).

`solver.newton` :

| Clé | Type | Défaut | Contrainte |
|---|---|---|---|
| `increments` | entier | 1 | $\ge 1$ ; nombre d'incréments nominaux |
| `max_iter` | entier | 25 | $\ge 1$ ; itérations de Newton par incrément |
| `rtol` | flottant | `1.0e-6` | $> 0$ ; résidu libre relatif |
| `atol` | flottant | `1.0e-10` | $\ge 0$ ; résidu libre absolu [N] |
| `max_cutbacks` | entier | 8 | $\ge 0$ ; réductions de pas successives |

La section `newton` peut être omise entièrement (tous ses champs ont une valeur par défaut).

`output` :

| Clé | Type | Défaut | Contrainte |
|---|---|---|---|
| `fields` | `"all"` ou liste de chaînes | `"all"` | sous-ensemble de `displacement`, `cauchy_stress`, `principal_stress`, `mean_stress`, `strain`, `principal_stretch`, `J` |
| `outdir` | chaîne | — | requis ; dossier de sortie |

`displacement` est toujours exporté, même s'il n'est pas demandé. Si une liste est fournie,
les champs sont exportés dans l'ordre du catalogue ci-dessus.

### Exemple commenté

Fichier complet du cas de référence (`studies/O-ring_compression.yaml`) : quart de joint
torique, trois plans de symétrie et un plan de contact.

```yaml
# Système d'unités cohérent : mm, t (tonne), s, N, MPa
#   -> E [MPa], rho [t/mm^3], gravité [mm/s^2], forces [N], déplacements [mm]

study_name: "O-ring_compression"

mesh:
  file: "../mesh/O-ring.exo"
  order: 2
  quadrature_degree: 4

material:
  E: 6.0
  nu: 0.4999               # formulation mixte u-p (Taylor-Hood)
  rho: 1.2e-9

physics:
  gravity: [0.0, 0.0, 0.0]

boundaries:
  # Symétrie plan y = 0 : face de section de départ
  - name: "symmetry_y"
    type: "dirichlet"
    components: [1]
    value: [0.0]
    probe_points: [[15.0, 0.0, 0.9]]

  # Symétrie plan x = 0 : face de section d'arrivée (quart de tore)
  - name: "symmetry_x"
    type: "dirichlet"
    components: [0]
    value: [0.0]
    probe_points: [[0.0, 15.0, 0.9]]

  # Plan médian z = 0 : demi-section (joint statique)
  - name: "symmetry_z"
    type: "dirichlet"
    components: [2]
    value: [0.0]
    probe_points: [[10.607, 10.607, 0.0]]

  # Peau torique (sommet initial à z = 2.0 mm) écrasée par le plan supérieur
  - name: "compression"
    type: "contact"
    obstacle:
      kind: "plane"
      point: [0.0, 0.0, 1.8]      # cote du plan (1.8 mm -> 0.2 mm d'écrasement)
      normal: [0.0, 0.0, -1.0]    # normale sortante de l'obstacle (vers le joint : -ez)
    epsilon_n: 1.0e+3             # [MPa/mm] pénalité normale
    epsilon_t: 1.0e+3             # [MPa/mm] pénalité tangentielle
    friction: 0.0                 # 0.0 = sans frottement
    probe_points: [[10.607, 10.607, 2.0]]
    probe_tol: 0.2

solver:
  newton:
    increments: 1
    max_iter: 30
    rtol: 1.0e-6

output:
  fields: all
  outdir: "../results"
```

Une variante plus sévère du même cas est fournie dans
`studies/O-ring_compression_severe.yaml` (plan de contact à `z = 0.6 mm`, soit 1.4 mm
d'écrasement) ; elle sert de test de robustesse et ne fait pas partie des références figées.

## Maillage

Le maillage est un fichier **Exodus II** fourni par l'utilisateur. Contraintes du lecteur :

- **3D** (`num_dim = 3`), coordonnées en **mm** ;
- **un seul bloc volumique** (`num_el_blk = 1`), type `HEX8` ou `HEX20` ;
- `HEX8` : géométrie linéaire (Lagrange $P_1$) ; `HEX20` : géométrie quadratique
  (serendipity $P_2$, 20 nœuds) ;
- numérotation d'éléments identitaire (`elem_num_map` absent ou trivial) ;
- *sidesets* nommés, qui doivent référencer des **faces de bord** (une face interne est
  refusée) ; deux sidesets ne peuvent pas partager une facette ;
- maillage conforme et jacobien strictement positif aux coins (contrôlé au chargement).

Les noms de `boundaries` du YAML sont les noms des sidesets. Un sideset du fichier non utilisé
par l'étude est simplement ignoré (signalé en mode `--verbose`). L'option `probe_points`
permet de vérifier qu'un sideset est bien celui attendu : chaque point sonde doit être à moins
de `probe_tol` de la face taguée.

Le motif MPI est important : le fichier n'est lu et validé que sur le rang 0, puis
`dolfinx.mesh.create_mesh` reçoit la connectivité d'un seul rang (les autres passent des
tableaux vides). Ce contournement évite une corruption mémoire de dolfinx 0.11.0 en parallèle.

## Sorties

### Réactions CSV

`results/<study_name>_reactions.csv`, une ligne par incrément accepté. Colonnes : `lambda`,
puis `<frontière>_Fx`, `<frontière>_Fy`, `<frontière>_Fz` pour chaque condition de Dirichlet,
puis `total_Fx`, `total_Fy`, `total_Fz` et `imbalance`. Les forces sont en N et représentent
la force exercée par l'appui sur la structure. `imbalance` est le bilan global relatif
(somme des réactions + forces extérieures).

### Export ParaView

`results/<study_name>.bp` est un fichier VTX (moteur `BP4`) lisible directement dans
ParaView. Champs exportés par défaut, dans cet ordre :

| Champ | Composantes | Contenu |
|---|---|---|
| `displacement` | 3 | déplacement $\mathbf u$ [mm] |
| `cauchy_stress` | 9 | tenseur de Cauchy 3×3 [MPa] |
| `principal_stress` | 3 | contraintes principales $\sigma_1 \ge \sigma_2 \ge \sigma_3$ [MPa] |
| `mean_stress` | 1 | $\mathrm{tr}(\boldsymbol\sigma)/3 = p$ [MPa] |
| `strain` | 9 | Green-Lagrange $\mathbf E$ [-] |
| `principal_stretch` | 3 | allongements principaux $\lambda_1 \ge \lambda_2 \ge \lambda_3$ [-] |
| `J` | 1 | $J = \det\mathbf F$ [-] |

Tous les champs sont projetés aux nœuds par une projection $L^2$ à masse concentrée $P_1$,
puis interpolés à l'ordre du maillage. Les tenseurs sont écrits en 9 composantes (3×3) et les
valeurs principales en vecteurs de 3 composantes, sans « Magnitude ».

### Résumé console

Le résumé (rang 0) affiche : versions dolfinx/PETSc, nom de l'étude, loi et formulation, nombre
de processus MPI, caractéristiques du maillage (type, cellules, DDLs, ordre, nœuds, volume en
mm³), compteurs de Newton (itérations, incréments, cutbacks), solveur linéaire, temps
d'assemblage et de résolution, résidu relatif final, évaluations de recherche linéaire,
réactions par frontière et totales, bilan global, chemin du CSV, déplacement maximal
$|\mathbf u|_{\max}$ [mm], `sigma_1_max` [MPa] et chemin de l'export ParaView.

`sigma_1_max` est la plus grande contrainte principale de Cauchy, maximale sur les nœuds de la
projection $P_1$ lissée. C'est un indicateur de **traction** ; pour un joint écrasé, un
indicateur de compression serait plus parlant (voir les limites).

## Architecture

### Fichiers

| Fichier | Rôle |
|---|---|
| `main.py` | CLI (`study`, `--verbose`), enchaînement du calcul, résumé console |
| `config.py` | modèles Pydantic v2 du YAML, obstacles, catalogue des champs exportables |
| `exodus_reader.py` | lecture bas niveau du fichier Exodus II avec `netCDF4` (points, connectivité, sidesets) |
| `exodus_mesh.py` | permutations Exodus → basix, création du maillage et des `MeshTags` dolfinx, motif MPI rang 0, contrôles |
| `material.py` | cinématique ($\mathbf F$, $\mathbf E$, $J$), loi néo-hookéenne mixte, PK1 et Cauchy |
| `contact.py` | jauge et normale des obstacles, pénalité normale, Coulomb régularisé |
| `forms.py` | espace Taylor-Hood, résidu et tangente UFL, assemblage des frontières |
| `dirichlet.py` | localisation des DDL de Dirichlet, `DirichletSet`, valeurs multipliées par $\lambda$ |
| `solver.py` | pilotage de charge, Newton avec backtracking, `LinearSolver` LU MUMPS, métriques |
| `reactions.py` | calcul des réactions par frontière et historique CSV |
| `postproc.py` | champs aux points de quadrature, valeurs principales, métriques, export VTX |
| `projection.py` | projection $L^2$ à masse concentrée $P_1$ et espaces d'export autonomes |
| `utils.py` | journalisation, vecteur PETSc partagé, intégrales, tailles de DDL, assemblage |
| `studies/O-ring_compression.yaml` | cas de référence |
| `studies/O-ring_compression_severe.yaml` | variante sévère (test de robustesse) |
| `mesh/O-ring.exo` | maillage Exodus II du cas de référence |
| `reference/` | références figées (journaux, CSV, métriques, statistiques de champs) et leur README |
| `tools/capture_metrics.py` | capture des métriques en précision complète et des statistiques de champs |
| `tools/compare_reference.py` | comparaison d'un run aux références, avec tolérances |
| `tools/wsl_run.sh` | wrapper Windows/WSL : conda, environnement, threads BLAS à 1 |
| `tests/` | suite pytest (configuration, contact, Exodus, cas O-ring) |

### Flux d'exécution de `main.py`

1. `argparse` : chemin du YAML, option `--verbose` ; configuration de la journalisation
   (rang 0 uniquement).
2. `config.load_config` : lecture, validation Pydantic, résolution des chemins relatifs au
   YAML.
3. `exodus_mesh.build_mesh` : lecture/validation rang 0, distribution MPI, tags de facettes.
4. `forms.build_forms` : loi, espace mixte, mesures, résidu et tangente ; puis
   `dirichlet.apply_bcs` : localisation unique des DDL de Dirichlet.
5. `reactions.ReactionHistory` puis `solver.solve_newton` : pilotage de charge, Newton et
   enregistrement des réactions à chaque incrément accepté.
6. `postproc.postprocess` : projection nodale, métriques `u_max` et `sigma_1_max`, export
   VTX.
7. `ReactionHistory.write_csv` puis `main.print_summary`.

Les erreurs attendues (YAML invalide, maillage introuvable, non-convergence, dépendance
manquante) sont affichées sans traceback ; en MPI, un échec termine tous les rangs avec un
code de retour non nul.

## Validation

### Cas de référence

`studies/O-ring_compression.yaml` : quart de joint torique, `HEX20`, $E = 6$ MPa,
$\nu = 0{,}4999$, $P_2$-$P_1$, `quadrature_degree: 4`, contact plan pénalisé sans frottement.
Valeurs figées (série) :

| Grandeur | Valeur |
|---|---|
| incréments | 1 |
| itérations de Newton | 10 |
| cutbacks | 0 |
| `u_max` | 4.664172049663e-01 mm |
| `sigma_1_max` | 1.599963399299e-01 MPa |
| `total_Fz` | 2.480275894546e+01 N |

### Références figées

`reference/` contient, pour la série et pour `np = 2` : journaux console (`*_series.log`,
`*_np2.log`), historique des réactions (`*_reactions.csv`), métriques haute précision
(`*_metrics.json`) et statistiques des champs (`*_fields.json`). `reference/README.md` détaille
leur génération et la normalisation des écarts.

### Outils

```bash
python tools/capture_metrics.py studies/O-ring_compression.yaml --out results/O-ring_compression_series_metrics.json --fields-out results/O-ring_compression_series_fields.json
python tools/compare_reference.py --run results/O-ring_compression_series_metrics.json --reference reference/O-ring_compression_series_metrics.json --fields-run results/O-ring_compression_series_fields.json --fields-reference reference/O-ring_compression_series_fields.json
```

Pour `np = 2`, préfixer **uniquement** `capture_metrics.py` par `mpiexec -n 2` et remplacer
`series` par `np2` dans les deux commandes ; `compare_reference.py` reste un script série.
`capture_metrics.py` rejoue exactement la séquence de `main._run` et écrit les valeurs en
précision complète ; `compare_reference.py` affiche les écarts et renvoie `VALIDATION RÉUSSIE`
ou `VALIDATION ÉCHOUÉE`.

### Tolérances et reproductibilité

- Série : écart relatif $\le 10^{-12}$ (attendu : bit à bit, écart 0) ;
- `np = 2` : écart relatif $\le 10^{-8}$ (l'ordre des réductions MPI n'est pas garanti) ;
- compteurs (incréments, itérations, cutbacks, évaluations de recherche linéaire) : égalité
  stricte.

Les réactions ont des composantes nulles par symétrie et d'autres de l'ordre de $10^{-9}$ N :
`compare_reference.py` normalise chaque composante par
`max(|référence|, 1e-8 × max|réaction|)`, et applique la même règle aux statistiques de champs.

La reproductibilité série dépend des threads BLAS à 1 : `tools/wsl_run.sh` fixe par défaut
`OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = 1`. Sans ce réglage, MUMPS via un
BLAS multi-thread n'est pas déterministe au niveau $10^{-15}$.

### Tests

`tests/` contient 42 tests (environ 90 s) :

| Fichier | Contenu |
|---|---|
| `test_config.py` | schéma strict, clés inconnues et supprimées, bornes des paramètres |
| `test_contact.py` | jauge, normale, pénalité, frottement sur des cas analytiques |
| `test_exodus.py` | permutations `HEX8`/`HEX20`, sidesets, maillage de référence, erreurs |
| `test_oring_golden.py` | run complet du cas O-ring et comparaison à `reference/` |

## Limites connues et pistes

- **Un seul corps** : le contact est unilatéral contre un obstacle analytique. Le contact
  corps-à-corps (deux maillages, deux champs) n'est pas traité.
- **Pénalité sans multiplicateurs** : la non-pénétration est approchée, avec une pénétration
  proportionnelle à $1/\varepsilon_n$. Un lagrangien augmenté ou des multiplicateurs de
  Lagrange supprimeraient ce réglage.
- **Frottement holonome** : valable pour un chargement monotone ; un chargement cyclique
  demanderait une variable d'état persistante aux points de quadrature et un retour radial
  incrémental.
- **Indicateur de contrainte** : `sigma_1_max` est une traction maximale. Pour un joint
  écrasé, la contrainte principale minimale (compression) serait plus pertinente ; c'est un
  point ouvert.
- **Pas de Neumann** : seules la gravité (pilotée) et le contact chargent le solide.
- **Une seule loi** : néo-hookéen mixte $(\mathbf u,p)$ Taylor-Hood ; pas d'autre modèle
  hyperélastique.
- **Maillage imposé** : Exodus II, un bloc, `HEX8`/`HEX20`, numérotation identitaire ; pas de
  génération ni d'adaptation de maillage.
- **Solveur direct** : MUMPS convient aux cas de taille modérée ; un solveur itératif
  parallèle serait nécessaire pour de très grands modèles.
- **Pilotage de charge** : $\lambda$ multiplie la gravité et les Dirichlet, mais pas le
  contact. Dans le cas de référence (gravité et Dirichlet nuls), la réponse est donc
  indépendante de $\lambda$, et un cutback rejoue le même problème ; une étude où le
  chargement doit réellement être incrémenté demande une charge extérieure non nulle.
