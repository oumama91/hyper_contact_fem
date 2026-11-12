# Plan de migration : `hyper_contact_fem`

> Document destiné à un agent de code (type OpenCode). À lire en entier avant toute action.
> Chemins : `<OLD>` = `D:\Recherche\FEM` (WSL : `/mnt/d/Recherche/FEM`), **lecture seule** ; `<NEW>` = `D:\Recherche\hyper_contact_fem` (WSL : `/mnt/d/Recherche/hyper_contact_fem`), répertoire de travail.
> Version du plan : mise à jour après validation de l'Étape 1.

## 0. Objectif

Extraire de `<OLD>` un projet minimal, maintenable, qui ne traite que :

- **un unique corps hyperélastique quasi-incompressible** (néo-hookéen, grandes déformations, formulation mixte $(\mathbf{u}, p)$ Taylor-Hood $P_k$-$P_{k-1}$, $k \ge 2$) ;
- **en contact unilatéral pénalisé avec frottement de Coulomb régularisé** contre des **obstacles rigides analytiques** (plan, cylindre, sphère) ;
- maillage **Exodus II uniquement** (`HEX8` / `HEX20`) ;
- gravité conservée, **Neumann supprimé**.

Cas de référence unique : `studies/O-ring_compression.yaml` (quart de joint torique, $E = 6$ MPa, $\nu = 0{,}4999$, $P_2$-$P_1$, `quadrature_degree: 4`). Unités : mm, t, s, N, MPa.

Principe directeur : **réduire la complexité, pas la déplacer.** Supprimer plutôt que réécrire. Aucune abstraction, option ou généralisation que le cas cible ne demande pas. Un nombre de fichiers est un indicateur, pas un but : ne pas fusionner deux modules de rôles différents pour atteindre un chiffre.

## 1. Décisions figées (validées par l'utilisateur)

| # | Décision |
|---|----------|
| D1 | Contact : on garde la formulation actuelle (un corps déformable contre obstacle rigide analytique). Pas de contact corps-à-corps. |
| D2 | Gravité conservée (`PhysicsConfig.gravity`). **Neumann supprimé** (déjà retiré à l'Étape 1). |
| D3 | Export restreint aux champs pertinents pour un élastomère : `displacement`, `cauchy_stress`, `principal_stress`, `mean_stress` ($= p$), `strain` ($\mathbf{E}$ Green-Lagrange), `principal_stretch`, `J`. **Supprimés** : `von_mises`, `tresca`, `equivalent_strain`, `principal_strain`. |
| D4 | Pas de rétrocompatibilité YAML : les clés `mesh.source`, `material.model`, `solver.method` (et toute clé liée à Gmsh, Neumann, GAMG, ou sans effet) sont supprimées. Schéma strict (`extra="forbid"`), YAML de référence mis à jour en conséquence. |
| D5 | Gmsh supprimé définitivement : aucune occurrence du mot `gmsh` dans le dépôt, **hors `PLAN_MIGRATION.md`** (ce fichier le cite pour décrire le travail et est exclu du `grep`). |
| D6 | Lecteur Exodus : on **conserve `netCDF4`** (pas de `meshio`, qui gère mal les sidesets). |

## 2. Règles de travail pour l'agent

1. `<OLD>` est **en lecture seule**. Tout travail se fait dans `<NEW>`.
2. Lire le code avant d'affirmer quoi que ce soit ; citer fichier et fonction dans les comptes rendus.
3. N'inventer aucune API, fonction ou bibliothèque. En cas de doute sur une API `dolfinx` / `petsc4py` / `basix` / `pydantic`, vérifier dans la version installée (`help()`, source, docs officielles) ou écrire « à vérifier ».
4. **Une sous-étape = un commit + une validation numérique passée avant de continuer.** Si la validation échoue : s'arrêter, rapporter, ne pas enchaîner, ne pas « corriger » en modifiant les références ou les tolérances.
5. Étapes 1 et 2 : **zéro changement d'algorithme numérique.** Seule exception explicite : la réduction des champs exportés (sous-étape 2.5, D3), qui change le contenu des sorties mais pas les calculs.
6. Code : noms en anglais, commentaires et docstrings en français. Formules en LaTeX dans les docstrings quand utile, avec la mesure de contrainte nommée (Cauchy, PK1, PK2).
7. Choix de conception qui dépend d'une information absente de ce document : s'arrêter et poser la question, ne pas trancher seul.
8. **Environnement** : aucun `python`, `mpiexec`, `pip` ni `git` côté Windows. Tout passe par `tools/wsl_run.sh` (charge conda, active `fenicsx-env`, fixe `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`). Ces threads à 1 sont **indispensables** à la reproductibilité de la tolérance série ($10^{-12}$) : ne pas les modifier. Fichiers écrits en fins de ligne LF.
9. Fin de chaque sous-étape : compte rendu court (fichiers modifiés, résultats de validation, écarts, points ouverts).

## 3. Invariants à préserver (ne pas toucher sans validation)

- **Énergie potentielle mixte** :
  $\Pi(\mathbf{u}, p) = \int_{\Omega_0} \left[ \psi_{\text{iso}}(\mathbf{F}) + p(J-1) - \tfrac{1}{2K} p^2 \right] \mathrm{d}\Omega_0 - W_{\text{ext}}$,
  avec $K = \frac{E}{3(1-2\nu)}$, $\mu = \frac{E}{2(1+\nu)}$, $\psi_{\text{iso}} = \frac{\mu}{2}(\bar I_1 - 3)$, $\bar I_1 = J^{-2/3}\,\mathrm{tr}(\mathbf{F}^T\mathbf{F})$.
  Travail virtuel interne et jacobienne obtenus par `ufl.derivative` (PK1 : $\mathbf{P} = \mathbf{P}_{\text{iso}} + pJ\mathbf{F}^{-T}$).
- **Cauchy** (post-traitement) : $\boldsymbol\sigma = \frac{1}{J}\mathbf{P}_{\text{iso}}\mathbf{F}^T + p\,\mathbf{I}$. **Green-Lagrange** : $\mathbf{E} = \frac12(\mathbf{F}^T\mathbf{F} - \mathbf{I})$.
- **Contact** (`contact.py`) : $p_N = \varepsilon_n \langle -g_N \rangle$ ; frottement de Coulomb holonome régularisé avec $\eta = 10^{-9}$ ; traction de contact $\mathbf{t}_c = p_N\mathbf{n} + \mathbf{t}_T$ intégrée sur la facette de référence, **non multipliée par le facteur de charge $\lambda$**. Garder les `ufl.conditional` tels quels.
- **Lecteur Exodus** : un seul bloc volumique, `elem_type` $\in$ {`HEX8`, `HEX20`}, `elem_num_map` identitaire, permutations `EXODUS_ORDER` (calculées par `_compute_order`), vérification jacobien $> 0$ aux 8 coins.
- **Création du maillage en MPI** : `dolfinx.mesh.create_mesh` reçoit `cells` et `points` **uniquement sur le rang 0** (tableaux vides sur les autres rangs), contournement documenté d'une corruption mémoire (`SIGABRT`) de `dolfinx` 0.11.0. Conserver ce motif et les levées d'exception collectives (`_raise_collective`) pour éviter tout deadlock MPI.
- **Solveur linéaire** : LU direct MUMPS obligatoire (matrice tangente mixte point-selle indéfinie : pas de GAMG).
- **Contact non lissé** : ne pas régulariser davantage $\langle\cdot\rangle$ ni le retour radial, et ne pas modifier la stratégie de pilotage de charge / cutback.

## 4. État du dépôt `<NEW>` (après Étape 1)

Commits : `f797744` (1.0), `6715131` (1.1), `9c9cabe` (1.0 suite, threads BLAS = 1), `00e07f0` (1.2, HEAD).

```text
hyper_contact_fem/
├── main.py  utils.py  .gitignore  PLAN_MIGRATION.md
├── mesh/        O-ring.exo  O-ring.jou
├── studies/     O-ring_compression.yaml
├── preproc/     config.py  config_types.py  contact.py  dirichlet.py  exodus.py  fields.py
│                forms.py  hyperelasticity.py  materials.py  mesher_exodus.py  neo_hookean.py  obstacles.py
├── solver/      linear_solver.py  solver.py
├── postproc/    postproc.py  projection.py  reactions.py  tensors.py  vtx_export.py
├── reference/   *_series.{log,_metrics.json,_reactions.csv}  *_np2.{...}  README.md   (références figées)
├── results/     sorties de run (O-ring_compression_reactions.csv, O-ring_compression.bp/)
└── tools/       wsl_run.sh  capture_metrics.py  compare_reference.py
```

Écarts connus de l'Étape 1 par rapport au plan initial :

- `preproc/config_types.py` conservé (dépendance de `config.py` et `obstacles.py`) : sa suppression est prévue en 2.3.
- `solver.rtol` conservé dans le schéma YAML mais **sans effet** (LU direct) : à supprimer en 2.3 après vérification par `grep` qu'aucun code ne le lit.
- Les fonctions de sondage de points (`_check_probe_points`) sont repliées dans `mesher_exodus.py` ; `MeshResult` aussi.
- Pas de `__init__.py` : paquets en namespace packages. Sans objet après aplatissement (2.1).
- Journaux de `reference/` : bannière d'environnement réduite à `dolfinx 0.11.0 / PETSc (3, 25, 6)` (documenté dans `reference/README.md`).
- Aucun dossier `tests/` n'existe encore.

## 5. Étape 1 : TERMINÉE et validée

Résultats (commit `00e07f0`) : série **bit à bit identique** aux références (écart 0 sur `u_max`, `vm_max`, `rel_res`, `imbalance`, réactions, `total_Fz`) ; np=2 dans la tolérance MPI (écart max $2{,}2\times10^{-11}$ sur `imbalance`) ; 1 incrément / 10 itérations de Newton / 0 cutback, identiques.

Note sur les réactions : les composantes de réaction « nulles » valent ~$4\times10^{-9}$ N face à $F_z \approx 24{,}8$ N ; leur écart n'a de sens qu'**en absolu rapporté à la force totale**, pas en relatif composante par composante. La sous-étape 2.0 vérifie et documente comment `tools/compare_reference.py` normalise cet écart.

## 6. Étape 2 : Aplatissement, Pydantic natif, réduction des champs

**Objectif** : structure plate, suppression des validateurs maison, application de D3 et D4. Toutes les sous-étapes sont **itération-identiques** (mêmes sorties numériques) sauf 2.5.

### Architecture cible (13 fichiers `.py` hors `tools/` et `tests/`)

```text
hyper_contact_fem/
├── main.py            # CLI + résumé console
├── config.py          # modèles Pydantic v2 : YAML, matériau, obstacles, solveur, physique/gravité
├── exodus_reader.py   # ex preproc/exodus.py : lecture NetCDF4 bas niveau
├── exodus_mesh.py     # ex preproc/mesher_exodus.py : permutations, Mesh/MeshTags dolfinx, MPI, MeshResult
├── material.py        # ex materials + hyperelasticity + neo_hookean : cinématique + néo-hookéen mixte (u, p)
├── contact.py
├── forms.py
├── dirichlet.py
├── solver.py          # ex linear_solver + solver ; logique de Newton inchangée
├── reactions.py
├── postproc.py        # ex postproc + tensors + vtx_export + fields
├── projection.py      # inchangé jusqu'à l'Étape 3
├── utils.py
├── mesh/  studies/  results/  reference/  tools/  tests/
```

`obstacles.py` est absorbé par `config.py` (ce sont des modèles Pydantic) ; si le fichier fusionné dépasse environ 300 lignes, garder `obstacles.py` séparé et le signaler. `exodus_reader.py` et `exodus_mesh.py` restent **deux fichiers** (I/O NetCDF brut contre logique dolfinx/MPI).

### Sous-étapes (un commit chacune, validation avant la suivante)

**2.0 Préparation (aucun changement de code de production)**
- Poser le tag git `step1-validated` sur le HEAD actuel.
- Vérifier `.gitignore` : `results/` (sorties de run) ignoré, `reference/` suivi. Si `results/` est actuellement suivi par git, le retirer de l'index (`git rm --cached`) **après** avoir copié `results/O-ring_compression.bp` dans `reference/` si ce .bp est la référence de champs voulue ; sinon demander.
- Étendre `tools/capture_metrics.py` pour enregistrer, pour chaque champ exporté, des statistiques globales (min, max, norme $\ell^2$ des valeurs nodales, rang 0 après réduction) dans `reference/*_fields.json` (série et np=2). Pour lire les champs, d'abord tester si le module Python `adios2` est disponible ; sinon capturer en cours d'exécution. Cette instrumentation ne modifie pas les résultats.
- Régénérer ces références de champs depuis le tag `step1-validated` (série bit-identique à `<OLD>`).
- Étendre `tools/compare_reference.py` pour comparer ces statistiques ; documenter dans `reference/README.md` la normalisation des écarts de réactions (cf. §5).
- Contrôle de l'environnement et de la validité : run série et np=2 de l'état actuel → `VALIDATION RÉUSSIE`.

**2.1 Aplatissement des dossiers**
- `git mv` des fichiers vers la racine ; renommer `exodus.py` en `exodus_reader.py` et `mesher_exodus.py` en `exodus_mesh.py`.
- Corriger uniquement les imports (aussi dans `tools/`). Aucun autre changement de contenu.

**2.2 Fusions de modules (sans modifier le contenu des fonctions)**
- `material.py` ← `materials.py` + `hyperelasticity.py` + `neo_hookean.py` : une seule classe `NeoHookean` en formulation mixte, sans ABC ni `Protocol` ; conserver les noms de méthodes utiles (`psi_iso`, `piola_iso`, `potential`, `internal_work`, `cauchy`, `strain`).
- `solver.py` ← `linear_solver.py` + `solver.py` : logique de Newton et de pilotage de charge inchangée.
- `postproc.py` ← `postproc.py` + `tensors.py` + `vtx_export.py` + `fields.py` : **tout** le contenu est conservé à ce stade (y compris `von_mises`, `tresca`…).
- `projection.py` et `reactions.py` inchangés.

**2.3 Pydantic natif et YAML (D4)**
- Fusionner `obstacles.py` dans `config.py` ; supprimer `config_types.py`.
- Types Pydantic v2 natifs (`PositiveFloat`, `NonNegativeFloat`, `Field(gt=0, ge=0)`, `StrictInt`…), `model_config = ConfigDict(extra="forbid")`.
- `MaterialConfig(E: PositiveFloat, nu: float, rho: NonNegativeFloat = 0.0)` avec la contrainte $-1 < \nu \le 0{,}5$ (reproduire exactement les bornes actuelles) ; `quadrature_degree: int = 4` (jamais `None`) ; `order >= 2` imposé.
- Supprimer `mesh.source`, `material.model`, `solver.method`, `solver.rtol` (après `grep` confirmant qu'aucun code ne le lit), et toute autre clé sans effet repérée ; mettre à jour `studies/O-ring_compression.yaml`.
- Créer `tests/test_config.py` : rejet des clés inconnues, de $E \le 0$, de $\nu \notin\,]-1; 0{,}5]$, de `order < 2`, des anciennes clés supprimées.

**2.4 Suppression des branches `mixed`**
- Supprimer `Forms.mixed` et toutes les branches `if not self.mixed`, ainsi que le paramètre `hyperelastic` de `available_fields` / `resolve_fields`. Aucune autre modification.

**2.5 Réduction des champs exportés (D3) : seule sous-étape qui change les sorties**
- Réduire la liste des champs à celle de D3 ; supprimer `von_mises`, `tresca`, `equivalent_strain`, `principal_strain` du calcul, de l'export et du code mort associé.
- `print_summary` ne doit plus afficher `vm_max`. Le remplacer par un indicateur pertinent en hyperélasticité (par exemple la plus grande contrainte principale de Cauchy, ou $\max\|\boldsymbol\sigma\|$) : choisir, documenter, et **régénérer** les références concernées (`reference/*_metrics.json`, `*_fields.json`, `.log`) en le consignant dans `reference/README.md`.
- Poser le tag `step2-validated` à la fin.

**2.6 Tests de non-régression** (si le temps le permet ; sinon reporter, sans bloquer l'étape)
- Regarder si `<OLD>/tests/` existe (`test_mesher_exodus.py`, `test_contact_geometry.py`, cités dans l'ancien code). Si oui, reprendre ces tests (adaptés aux nouveaux noms), sinon écrire `tests/test_contact.py` (cas 0D analytique : jauge $g_N$, normale, signe du frottement) et `tests/test_exodus.py` (permutations, sidesets).
- `tests/test_oring_golden.py` : appelle la comparaison avec `reference/`.

### Validation (bloquante, après chaque sous-étape)

| Sous-étape | Run série | Run np = 2 |
|---|---|---|
| 2.0 à 2.4 | tous les champs, réactions, `u_max`, `vm_max`, `rel_res`, `imbalance`, incréments/itérations : écart $\le 10^{-12}$ (attendu : 0) | écart $\le 10^{-8}$ |
| 2.5 | mêmes grandeurs sauf `vm_max` (remplacé) ; champs restants identiques ; nouveaux indicateurs régénérés | idem |

Contrôles supplémentaires : `grep -ri gmsh . --exclude-dir=.git --exclude=PLAN_MIGRATION.md` vide ; `python -m pyflakes` (ou équivalent déjà installé) sans import manquant ; bilan de fichiers `.py` et de lignes de code avant/après dans le compte rendu (objectif indicatif $-35\,\%$ de lignes).

**Risque** : les messages d'erreur Pydantic remplacent les anciens ; adapter les tests qui vérifient des sous-chaînes. Autre risque : imports circulaires après aplatissement (résoudre par déplacement de fonctions, pas par imports locaux).

## 7. Étape 3 : Plomberie numérique maison vs primitives `dolfinx` / PETSc (avec point de décision)

**Objectif** : alléger solveur et post-traitement, mettre à jour les appels dépréciés. Chaque sous-tâche est indépendante et **réversible** (un commit par sous-tâche). Référence de départ : tag `step2-validated` et `reference/` régénéré en 2.5.

**3.1 Appels dépréciés** : remplacer l'accès à `f.x.petsc_vec` dans `utils.petsc_vec` par l'API actuelle de la version installée (`dolfinx.la.petsc.create_vector_wrap` à vérifier). Objectif : zéro `DeprecationWarning` sur le cas O-ring.

**3.2 Solveur non linéaire (point de décision)**
- Prototyper dans une branche : `PETSc.SNES` (via `dolfinx.fem.petsc.NonlinearProblem`, signature exacte à vérifier dans la version installée) + MUMPS, en conservant la boucle externe de pilotage de charge / cutback autour de `SNES.solve()`.
- Tester les recherches linéaires `bt`, `basic`, `l2` sur O-ring. Attention : la dérivée de `ufl.conditional` (contact, retour radial) est discontinue ; le backtracking maison (décroissance stricte de $\|R\|$) tolère ces sauts et `SNES` peut stagner.
- **Règle de décision** : adopter `SNES` seulement si le cas O-ring converge avec le même nombre d'incréments (±10 %) et sans cutback supplémentaire. Sinon, **conserver `_newton_increment`** (le nettoyer, ne pas le remplacer) et le consigner dans le compte rendu.

**3.3 Projection $L^2$ (optionnel)** : remplacer `projection.py` par `dolfinx.fem.Expression` + `Function.interpolate()` **seulement si** la perte du lissage $L^2$ condensé $P_1$ sur les champs de contrainte est acceptable (l'interpolation directe aux nœuds $P_2$ n'est pas lissée entre éléments). Sinon, conserver `projection.py`.

**Validation (bloquante)**
- O-ring converge sans avertissement de dépréciation.
- Réactions finales et $\|\mathbf{u}\|_\infty$ : écart $\le 10^{-6}$ par rapport à `reference/` (si `SNES` est adopté, le chemin de convergence change : resserrer à $10^{-8}$ si la tolérance Newton le permet). Si 3.3 est appliqué, les champs de contrainte peuvent légitimement différer : documenter l'écart.
- Run série et np=2.

**Risque principal** : comportement de la recherche linéaire native face à $J \to 0$ et à la non-régularité du contact.

## 8. Dépendances cibles

- `python >= 3.10`, `fenics-dolfinx == 0.11.0`, PETSc 3.25.6 (versions confirmées par les journaux de référence), `basix`, `ufl`, `ffcx`, `petsc4py` avec **MUMPS**, `mpi4py`
- `numpy`, `netCDF4`, `pydantic >= 2`, `pyyaml`
- Supprimées : `gmsh`. `scipy` : à ne garder que si une ligne du code l'utilise réellement (vérifier avec `grep`).

## 9. Points à vérifier

1. Signature exacte de `NonlinearProblem` et de `create_vector_wrap` dans `dolfinx` 0.11.0 (Étape 3).
2. Présence de `adios2` côté Python dans `fenicsx-env` (outil de lecture des `.bp`, sous-étape 2.0).
3. Existence de `<OLD>/tests/` et de `DESIGN_CONTACT.md` (sous-étape 2.6).
4. Normalisation exacte des écarts de réactions dans `tools/compare_reference.py` (sous-étape 2.0).

## 10. Définition de « terminé »

- `grep -ri gmsh` vide (hors `PLAN_MIGRATION.md`) ; plus aucune référence à Neumann, élasticité linéaire, GAMG, registres.
- Cas O-ring OK en série et np=2, dans les tolérances de chaque sous-étape.
- Structure plate de 13 fichiers `.py` au plus (hors `tools/` et `tests/`), tests verts, aucun `DeprecationWarning`.
- Compte rendu final : bilan avant/après (fichiers, lignes, dépendances), décision SNES documentée, points ouverts.
