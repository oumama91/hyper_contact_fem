# Plan de migration : `hyper_contact_fem`

> Document destiné à un agent de code (type OpenCode). À lire en entier avant toute action.
> Chemins : `<OLD>` = `D:\Recherche\FEM` (WSL : `/mnt/d/Recherche/FEM`), **lecture seule** ; `<NEW>` = `D:\Recherche\hyper_contact_fem` (WSL : `/mnt/d/Recherche/hyper_contact_fem`), répertoire de travail.
> État : **Étapes 1 et 2 terminées et validées** (tags `step1-validated`, `step2-validated`). Reste l'Étape 3.

## 0. Objectif

Extraire de `<OLD>` un projet minimal, maintenable, qui ne traite que :

- **un unique corps hyperélastique quasi-incompressible** (néo-hookéen, grandes déformations, formulation mixte $(\mathbf{u}, p)$ Taylor-Hood $P_k$-$P_{k-1}$, $k \ge 2$) ;
- **en contact unilatéral pénalisé avec frottement de Coulomb régularisé** contre des **obstacles rigides analytiques** (plan, cylindre, sphère) ;
- maillage **Exodus II uniquement** (`HEX8` / `HEX20`) ;
- gravité conservée, **Neumann supprimé**.

Cas de référence unique : `studies/O-ring_compression.yaml` (quart de joint torique, $E = 6$ MPa, $\nu = 0{,}4999$, $P_2$-$P_1$, `quadrature_degree: 4`). Unités : mm, t, s, N, MPa.

Principe directeur : **réduire la complexité, pas la déplacer.** Supprimer plutôt que réécrire. Aucune abstraction, option ou généralisation que le cas cible ne demande pas. Un nombre de fichiers est un indicateur, pas un but.

## 1. Décisions figées (validées par l'utilisateur)

| # | Décision |
|---|----------|
| D1 | Contact : on garde la formulation actuelle (un corps déformable contre obstacle rigide analytique). Pas de contact corps-à-corps. |
| D2 | Gravité conservée (`PhysicsConfig.gravity`). Neumann supprimé. |
| D3 | Export restreint aux champs pertinents pour un élastomère : `displacement`, `cauchy_stress`, `principal_stress`, `mean_stress` ($= p$), `strain` ($\mathbf{E}$ Green-Lagrange), `principal_stretch`, `J`. Supprimés : `von_mises`, `tresca`, `equivalent_strain`, `principal_strain`. |
| D4 | Pas de rétrocompatibilité YAML ; schéma Pydantic strict (`extra="forbid"`). |
| D5 | Gmsh supprimé définitivement : aucune occurrence du mot `gmsh` dans le dépôt, **hors `PLAN_MIGRATION.md`** (exclu du `grep`). |
| D6 | Lecteur Exodus : on conserve `netCDF4` (pas de `meshio`). |
| D7 | **`projection.py` est conservé tel quel** (projection $L^2$ condensée $P_1$). La remplacer par une interpolation directe changerait les champs de contrainte et l'indicateur `sigma_1_max`, donc invaliderait les références, pour un gain d'une centaine de lignes. |

## 2. Règles de travail pour l'agent

1. `<OLD>` est **en lecture seule**. Tout travail se fait dans `<NEW>`.
2. Lire le code avant d'affirmer quoi que ce soit ; citer fichier et fonction dans les comptes rendus.
3. N'inventer aucune API, fonction ou bibliothèque. En cas de doute sur une API `dolfinx` / `petsc4py` / `basix` / `pydantic`, vérifier dans la version installée (`help()`, source, docs officielles) ou écrire « à vérifier ».
4. **Une sous-étape = un commit + une validation numérique passée avant de continuer.** Si la validation échoue : s'arrêter, rapporter, ne pas enchaîner, ne pas « corriger » en modifiant les références, les tolérances ou `tools/compare_reference.py`.
5. Zéro changement d'algorithme numérique, sauf décision explicite et validée (Étape 3.2, règle de décision).
6. Code : noms en anglais, commentaires et docstrings en français. Formules en LaTeX dans les docstrings quand utile, avec la mesure de contrainte nommée (Cauchy, PK1, PK2).
7. Choix de conception qui dépend d'une information absente de ce document : choisir l'option la plus simple, la documenter dans `reference/README.md` et la signaler dans le compte rendu ; s'arrêter et demander seulement si le choix est irréversible ou touche un invariant.
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
- **Création du maillage en MPI** : `dolfinx.mesh.create_mesh` reçoit `cells` et `points` **uniquement sur le rang 0** (tableaux vides sur les autres rangs), contournement documenté d'une corruption mémoire (`SIGABRT`) de `dolfinx` 0.11.0. Conserver ce motif et les levées d'exception collectives (`_raise_collective`).
- **Solveur linéaire** : LU direct MUMPS obligatoire (matrice tangente mixte point-selle indéfinie : pas de GAMG).
- **Contact non lissé** : ne pas régulariser davantage $\langle\cdot\rangle$ ni le retour radial, et ne pas modifier la stratégie de pilotage de charge / cutback.

## 4. État actuel du dépôt `<NEW>` (après Étape 2)

Tags : `step1-validated` (sur `e31bd08`, code identique à `00e07f0`), `step2-validated` (sur `db4c12e`).

```text
hyper_contact_fem/
├── main.py          # CLI + résumé console
├── config.py        # modèles Pydantic v2 (YAML, matériau, obstacles, solveur, physique/gravité) + catalogue des champs
├── exodus_reader.py # lecture NetCDF4 bas niveau
├── exodus_mesh.py   # permutations Exodus→basix, Mesh/MeshTags dolfinx, MPI, MeshResult
├── material.py      # cinématique + NeoHookean mixte (u, p)
├── contact.py       # jauge, normale, pénalité normale, Coulomb régularisé
├── forms.py         # espace Taylor-Hood, résidu et tangente UFL
├── dirichlet.py
├── solver.py        # pilotage de charge, Newton, LinearSolver (LU MUMPS), NewtonDiverged
├── reactions.py
├── postproc.py      # champs, tenseurs, export VTX
├── projection.py    # projection L2 condensée P1 (conservée, D7)
├── utils.py
├── .gitignore  PLAN_MIGRATION.md
├── mesh/            # O-ring.exo, O-ring.jou
├── studies/         # O-ring_compression.yaml
├── reference/       # références figées (*_series et *_np2 : .log, _metrics.json, _fields.json, _reactions.csv) + README.md
├── results/         # sorties de run, ignorées par git
├── tests/           # test_config.py, test_contact.py, test_exodus.py, test_oring_golden.py (42 tests, ~85 s)
└── tools/           # wsl_run.sh, capture_metrics.py, compare_reference.py
```

Résumé des Étapes 1 et 2 (terminées, ne pas refaire) :

| Étape | Résultat |
|---|---|
| 1 | Gmsh, élasticité linéaire, GAMG, Neumann, registres supprimés ; références capturées ; série bit à bit identique à `<OLD>`, np=2 dans la tolérance MPI. |
| 2 | Aplatissement (21 → 13 fichiers `.py`), `material.py`, `solver.py`, `postproc.py` fusionnés, Pydantic natif, clés YAML sans effet retirées, champs D3, 3 fichiers de tests ajoutés. Série bit à bit identique, np=2 ≤ $3\times10^{-9}$ ; −7 % de lignes seulement (code déplacé, pas réécrit). |

Choix pris à l'Étape 2, à connaître :

- `sigma_1_max` (max nodal $P_1$ lissé de la plus grande contrainte principale de Cauchy) remplace `vm_max` dans le résumé console et les références. Valeur : $0{,}16$ MPa (l'ancien `vm_max` valait $0{,}754$ MPa). Question ouverte : un indicateur compressif (contrainte principale minimale) serait plus parlant pour un joint écrasé. Ne pas modifier sans accord explicite.
- Le catalogue de champs (`COMPONENTS`, `COMMON_FIELDS`, `available_fields`, `resolve_fields`) est dans `config.py` (évite un import circulaire `config → postproc → forms → config`).
- La sous-étape 2.4 (branches `mixed`) était déjà vide : commit documenté.
- `LinearSolver(comm, forms.V)` : paramètre `config` et `rtol` supprimés.
- Les statistiques de champs de `reference/*_fields.json` sont capturées à l'exécution en enrobant `postproc.write_vtx` (pas de lecteur `adios2` côté Python) ; le `.bp` n'est pas archivé.
- Les tests de contact utilisent un maillage tétraédrique unique et une perturbation `1e-30` pour lier les expressions constantes au maillage (contournement UFL, acceptable dans les tests).

## 5. Outils et validation (valables pour toute sous-étape)

- Run série : `tools/wsl_run.sh python main.py studies/O-ring_compression.yaml`. Run np=2 : `... mpiexec -n 2 python main.py studies/O-ring_compression.yaml`.
- Comparaison : `tools/capture_metrics.py` puis `tools/compare_reference.py` contre `reference/` ; verdict attendu `VALIDATION RÉUSSIE`.
- Tests : `tools/wsl_run.sh python -m pytest tests` (42 tests, environ 85 s).
- Réactions : les composantes « nulles » valent ~$4\times10^{-9}$ N face à $F_z \approx 24{,}8$ N ; leur écart n'a de sens qu'en absolu rapporté à la force totale (voir `reference/README.md` pour la normalisation exacte).
- Contrôle Gmsh : `grep -ri gmsh . --exclude-dir=.git --exclude=PLAN_MIGRATION.md` doit être vide.

Tolérances : série $\le 10^{-12}$ (attendu : 0, bit à bit), np=2 $\le 10^{-8}$ ; incréments, itérations de Newton et cutbacks identiques.

## 6. Étape 3 : plomberie numérique maison vs primitives `dolfinx` / PETSc

**Objectif** : alléger le solveur, mettre à jour les appels dépréciés, nettoyer les références orphelines. Référence de départ : tag `step2-validated`. Un commit par sous-étape.

**3.0 Synchronisation du plan** : vérifier que §4 de ce fichier correspond à l'arborescence réelle ; corriger l'écart éventuel sans toucher aux décisions, invariants et §6.

**3.1 Appels dépréciés** : remplacer l'accès à `f.x.petsc_vec` dans `utils.petsc_vec` par l'API actuelle de la version installée (`dolfinx.la.petsc.create_vector_wrap` à vérifier). Objectif : zéro `DeprecationWarning` sur le cas O-ring. Même code numérique, mêmes sorties.

**3.2 Solveur non linéaire (point de décision)**, dans une branche `step3-snes` :
- Prototype `PETSc.SNES` (via `dolfinx.fem.petsc.NonlinearProblem` ou équivalent ; signature exacte à vérifier) avec MUMPS, en conservant la boucle externe de pilotage de charge / cutback autour de `SNES.solve()`.
- Tester les recherches linéaires `basic`, `bt` et `l2`. Attention : la dérivée de `ufl.conditional` (contact, retour radial) est discontinue ; le backtracking maison (décroissance stricte de $\|R\|$) tolère ces sauts, `SNES` peut stagner.
- Le cas O-ring de référence (1 incrément, 10 itérations de Newton, 0 cutback) est trop doux pour juger : construire une **variante sévère** sans toucher au cas de référence (copie du YAML, par exemple `studies/O-ring_compression_severe.yaml`, avec seule l'amplitude de compression augmentée pour provoquer plusieurs incréments et au moins un cutback avec le solveur actuel ; augmenter progressivement, redescendre si le solveur actuel ne converge plus ; documenter la valeur retenue).
- **Règle de décision** : adopter `SNES` seulement si (i) O-ring converge avec le même nombre d'incréments ($\pm 10\,\%$) et sans cutback supplémentaire, **et** (ii) la variante sévère converge au moins aussi bien qu'avec le solveur actuel (cutbacks identiques ou moindres, résultat final cohérent à $10^{-6}$ près). Sinon : abandonner, supprimer la branche, conserver le solveur actuel (nettoyage minimal sans réécriture), consigner le tableau d'essais dans `reference/README.md`.
- Si adopté : validation série et np=2 (réactions et $\|\mathbf{u}\|_\infty \le 10^{-6}$ face à `reference/`, resserrer à $10^{-8}$ si la tolérance de Newton le permet), tests passés, merge.

**3.3 Projection $L^2$** : **non réalisée** (D7). `projection.py` reste inchangé.

**3.4 Nettoyage final** : `contact.py` cite `DESIGN_CONTACT.md`, absent du dépôt. Si `<OLD>/DESIGN_CONTACT.md` existe, le signaler à l'utilisateur sans le copier ; sinon retirer la référence et reformuler le commentaire pour qu'il soit autonome. Aucune autre modification.

Fin d'étape : poser le tag `step3-validated`.

**Risque principal** : comportement de la recherche linéaire native de `SNES` face à $J \to 0$ et à la non-régularité du contact.

## 7. Dépendances cibles

- `python >= 3.10`, `fenics-dolfinx == 0.11.0`, PETSc 3.25.6 (versions confirmées par les journaux de référence), `basix`, `ufl`, `ffcx`, `petsc4py` avec **MUMPS**, `mpi4py`
- `numpy`, `netCDF4`, `pydantic >= 2`, `pyyaml` ; `pytest` pour les tests
- Absentes : `gmsh`, `scipy`.

## 8. Points ouverts

1. Indicateur de contrainte du résumé console : `sigma_1_max` (traction) ou ajout d'un équivalent compressif ; décision de l'utilisateur en attente.
2. Réduction de lignes réelle (−7 % seulement) : une revue de code ciblée à l'intérieur des fonctions est possible après l'Étape 3, hors plan actuel.
3. Signature exacte de `NonlinearProblem` et de `create_vector_wrap` dans `dolfinx` 0.11.0 (Étape 3, à vérifier dans la version installée).
4. Existence de `<OLD>/DESIGN_CONTACT.md` (3.4).

## 9. Définition de « terminé »

- `grep -ri gmsh` vide (hors `PLAN_MIGRATION.md`) ; plus aucune référence à Neumann, élasticité linéaire, GAMG, registres.
- Cas O-ring OK en série et np=2, dans les tolérances ; 42 tests verts ; aucun `DeprecationWarning`.
- Décision `SNES` documentée avec son tableau d'essais.
- Tags `step1-validated`, `step2-validated`, `step3-validated` posés.
- Compte rendu final : bilan avant/après (fichiers, lignes, dépendances), points ouverts.
