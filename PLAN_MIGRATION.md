# Plan de migration : `hyper_contact_fem`

> Document destiné à un agent de code (type OpenCode). À lire en entier avant toute action.
> Placeholders à remplacer avant lancement : `<OLD>` = chemin du projet existant (lecture seule), `<NEW>` = chemin du nouveau projet.

## 0. Objectif

Extraire de `<OLD>` un nouveau projet minimal, maintenable, qui ne traite que :

- **un unique corps hyperélastique quasi-incompressible** (néo-hookéen, grandes déformations, formulation mixte $(\mathbf{u}, p)$ Taylor-Hood $P_k$-$P_{k-1}$, $k \ge 2$) ;
- **en contact unilatéral pénalisé avec frottement de Coulomb régularisé** contre des **obstacles rigides analytiques** (plan, cylindre, sphère) ;
- maillage **Exodus II uniquement** (`HEX8` / `HEX20`) ;
- gravité conservée, **Neumann supprimé**.

Cas de référence unique : `studies/O-ring_compression.yaml` (quart de joint torique, $E = 6$ MPa, $\nu = 0{,}4999$, $P_2$-$P_1$, `quadrature_degree: 4`). Unités : mm, t, s, N, MPa.

Principe directeur : **réduire la complexité, pas la déplacer.** Supprimer plutôt que réécrire. Aucune abstraction, option ou généralisation que le cas cible ne demande pas.

## 1. Décisions figées (validées par l'utilisateur)

| # | Décision |
|---|----------|
| D1 | Contact : on garde la formulation actuelle (un corps déformable contre obstacle rigide analytique). Pas de contact corps-à-corps. |
| D2 | Gravité conservée (`PhysicsConfig.gravity`). **Neumann supprimé** (`neumann.py`, `NeumannBoundary`, terme associé dans `forms.py`). |
| D3 | Export restreint aux champs pertinents pour un élastomère : `displacement`, `cauchy_stress`, `principal_stress`, `mean_stress` ($= p$), `strain` ($\mathbf{E}$ Green-Lagrange), `principal_stretch`, `J`. **Supprimés** : `von_mises`, `tresca`, `equivalent_strain`, `principal_strain`. |
| D4 | Pas de rétrocompatibilité YAML : les clés `mesh.source`, `material.model`, `solver.method` (et toute clé liée à Gmsh, Neumann, GAMG) sont supprimées. Le schéma est strict (`extra="forbid"`) et le YAML de référence est mis à jour en conséquence. |
| D5 | Gmsh supprimé définitivement : aucune occurrence du mot `gmsh` dans le nouveau dépôt. |
| D6 | Lecteur Exodus : on **conserve `netCDF4`** (pas de `meshio`, qui gère mal les sidesets). |

## 2. Règles de travail pour l'agent

1. `<OLD>` est **en lecture seule**. Tout travail se fait dans `<NEW>`.
2. Lire le code avant d'affirmer quoi que ce soit ; citer fichier et fonction dans les comptes rendus.
3. N'inventer aucune API, fonction ou bibliothèque. En cas de doute sur une API `dolfinx` / `petsc4py` / `basix`, vérifier dans la version installée (`help()`, source, docs officielles) ou écrire « à vérifier ».
4. **Une étape = un commit (ou plus) + une validation numérique passée avant de continuer.** Ne jamais enchaîner sur l'étape suivante si la validation échoue : s'arrêter et rapporter.
5. Étapes 1 et 2 : **zéro changement d'algorithme numérique.** Seul le périmètre et la structure changent.
6. Code : noms en anglais, commentaires et docstrings en français. Formules en LaTeX dans les docstrings quand utile, avec la mesure de contrainte nommée (Cauchy, PK1, PK2).
7. En cas de choix de conception qui dépend d'une information absente de ce document : s'arrêter et poser la question, ne pas trancher seul.
8. Fin de chaque étape : produire un compte rendu court (fichiers créés/supprimés, résultats de validation, écarts, points ouverts).

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

## 4. Disposition des fichiers

### 4.1 À conserver (chemin critique)

`main.py`, `utils.py`, `preproc/config.py`, `preproc/obstacles.py`, `preproc/exodus.py`, `preproc/mesher_exodus.py` (+ dataclass `MeshResult`), `preproc/materials.py`, `preproc/hyperelasticity.py`, `preproc/neo_hookean.py`, `preproc/contact.py`, `preproc/dirichlet.py`, `preproc/forms.py`, `preproc/fields.py`, `solver/linear_solver.py` (branche `direct`/MUMPS seulement), `solver/solver.py`, `postproc/reactions.py`, `postproc/projection.py`, `postproc/tensors.py`, `postproc/vtx_export.py`, `postproc/postproc.py`, `mesh/O-ring.exo`, `mesh/O-ring.jou`, `studies/O-ring_compression.yaml`, `results/O-ring_compression_reactions.csv`, `results/O-ring_compression.bp/`.

### 4.2 À ne pas copier / à supprimer

| Élément | Motif |
|---|---|
| `preproc/mesher_gmsh.py`, `preproc/mesh_refinement.py`, `preproc/probing.py`, `cad/` | Gmsh / OpenCASCADE (D5) |
| `preproc/mesher.py::build_mesh`, `MESH_BACKENDS`, `MESH_REQUIREMENTS` | Aiguillage dynamique inutile (Exodus seul) |
| Dans `main.py` : bloc `try: import gmsh`, `GMSH_LABEL` | D5 |
| Dans `preproc/config.py` : `GmshMeshConfig`, `GeometryConfig`, `BoundaryRefinement`, `SphereRefinement`, `Refinement`, `GMSH_ONLY_MESH_KEYS`, `DEFAULT_SOURCE`, `mesh_source()`, branches `if source == "gmsh"` | D5 |
| `preproc/linear_elastic.py` | Élasticité linéaire hors périmètre |
| `MATERIAL_MODELS`, `register_material()`, `DEFAULT_MODEL`, `Material(ABC)`, `MeshSetup`, `SolverSetup`, `preproc/__init__.py` | Registre / plugins : sur-ingénierie |
| `preproc/validators.py` | Code mort ou redondant avec Pydantic |
| `build_nullspace`, `check_nullspace`, `GAMG_ESTEIG_STEPS`, branche `gamg`, `SOLVER_METHODS`, `KRYLOV_TYPES`, clés `method` / `krylov` / `check_nullspace` | GAMG interdit en mixte |
| `preproc/neumann.py`, `NeumannBoundary`, terme Neumann de `build_forms` | D2 |
| `Forms.mixed` et branches `if not self.mixed`, paramètre `hyperelastic` de `available_fields` / `resolve_fields` | Formulation toujours mixte |
| `studies/beam_bending.yaml`, `studies/tubular_T-joint_compression.yaml`, `results/beam_bending*`, `results/tubular_T-joint*` | Hors périmètre |

## 5. Plan en 3 étapes

### Étape 1 : Extraction iso-fonctionnelle du chemin critique

**Objectif** : obtenir un nouveau projet qui exécute le cas O-ring, sans Gmsh, sans élasticité linéaire, sans GAMG, sans Neumann, avec les mêmes algorithmes.

**1.0 Références de non-régression (à faire en premier, dans `<OLD>`, sans le modifier)**
- Lancer `python main.py studies/O-ring_compression.yaml` en série, puis `mpiexec -n 2 python main.py studies/O-ring_compression.yaml`.
- Sauvegarder dans `<NEW>/reference/` : sortie console complète, `O-ring_compression_reactions.csv`, et les métriques finales ($\|\mathbf{u}\|_\infty$, résidu relatif final, nombre d'incréments / itérations de Newton, `vm_max`, `imbalance`) pour série et np=2.
- Écrire `tools/compare_reference.py` : compare un run à ces références et affiche les écarts relatifs.

**1.1 Squelette et copie**
- Créer `<NEW>` avec les fichiers de §4.1 uniquement (arborescence provisoire identique à l'ancienne).
- Intégrer `MeshResult` dans `mesher_exodus.py` ; `main.py` appelle directement `mesher_exodus.build_mesh`.

**1.2 Suppressions (§4.2)**
- Retirer tout ce qui concerne Gmsh, élasticité linéaire, GAMG / nullspace, Neumann (config, forms, main).
- `solver/linear_solver.py` : ne garder que LU MUMPS.
- Adapter le YAML de référence **uniquement si le schéma l'exige** à ce stade.

**Validation (bloquante)**
- `grep -ri gmsh <NEW>` : aucun résultat ; exécution OK dans un environnement sans le paquet `gmsh`.
- Run série vs référence : écart relatif $\le 10^{-12}$ sur réactions, $\|\mathbf{u}\|_\infty$, `vm_max`, et mêmes nombres d'incréments / itérations.
- Run np=2 vs référence np=2 : écart relatif $\le 10^{-8}$ (l'ordre des réductions MPI n'est pas garanti bit à bit).

**Risque** : import croisé oublié vers `preproc.mesher` ou `GmshMeshConfig`.

### Étape 2 : Aplatissement de l'architecture, Pydantic natif, réduction des champs

**Objectif** : structure plate (cible $\le 12$ fichiers `.py`), suppression des validateurs maison, application de D3 et D4.

**2.1 Structure plate** (cible ci-dessous)
```text
hyper_contact_fem/
├── main.py          # CLI + résumé console
├── config.py        # modèles Pydantic v2 (YAML, matériau, obstacles, solveur, physique/gravité)
├── exodus.py        # lecture NetCDF4 + conversion Mesh/MeshTags dolfinx (ex exodus.py + mesher_exodus.py)
├── material.py      # cinématique + néo-hookéen mixte (u, p) (ex materials + hyperelasticity + neo_hookean)
├── contact.py
├── forms.py
├── dirichlet.py
├── solver.py        # (étape 2 : linear_solver + solver fusionnés, logique de Newton inchangée)
├── reactions.py
├── postproc.py      # champs + export VTX (ex postproc + tensors + vtx_export ; projection.py conservé tel quel jusqu'à l'étape 3)
├── utils.py
├── mesh/  studies/  results/  reference/  tools/
└── tests/
```
(`projection.py` peut rester un fichier séparé jusqu'à l'étape 3 : le plafond de 12 fichiers l'autorise.)

**2.2 `material.py`** : fusionner les trois modules en un seul, classe unique `NeoHookean` en formulation mixte, sans registre, sans `Protocol`, sans ABC. Conserver les noms des méthodes utiles (`psi_iso`, `piola_iso`, `potential`, `internal_work`, `cauchy`, `strain`).

**2.3 `config.py`** : supprimer `config_types.py` et `validators.py`. Types Pydantic v2 natifs (`PositiveFloat`, `NonNegativeFloat`, `Field(gt=0, ge=0)`, `StrictInt`…), `model_config = ConfigDict(extra="forbid")`.
- `MaterialConfig(E: PositiveFloat, nu: float, rho: NonNegativeFloat = 0.0)` avec contrainte $-1 < \nu \le 0{,}5$ (reproduire exactement les bornes actuelles).
- `quadrature_degree: int = 4` par défaut (jamais `None`, pour éviter l'explosion de quadrature UFL avec $J^{-2/3}$ et la racine du frottement).
- `order >= 2` imposé (Taylor-Hood).
- Supprimer `mesh.source`, `material.model`, `solver.method` et mettre à jour `studies/O-ring_compression.yaml` (D4).

**2.4 `forms.py` / `fields.py`** : supprimer `Forms.mixed` et toutes les branches associées ; réduire `COMMON_FIELDS` / `HYPERELASTIC_FIELDS` à la liste de D3 ; supprimer `von_mises` de `tensors` et du post-traitement.
- Le résumé console (`print_summary`) ne doit plus afficher `vm_max`. Le remplacer par un indicateur pertinent, par exemple $\max \|\boldsymbol\sigma\|$ ou la plus grande contrainte principale de Cauchy, et **régénérer** les références concernées en le documentant (voir validation).

**Validation (bloquante)**
- Réactions finales, $\|\mathbf{u}\|_\infty$, nombre d'incréments / itérations : mêmes tolérances qu'à l'étape 1 (série $10^{-12}$, np=2 $10^{-8}$). Ces grandeurs ne dépendent pas de D3.
- Les champs exportés (`.bp`) restants coïncident avec la référence (`displacement`, `cauchy_stress`, `mean_stress`, `J`…) aux mêmes tolérances.
- Tests de config : rejet des clés inconnues, de $E \le 0$, de $\nu \notin\,]-1; 0{,}5]$, de `order < 2`, des anciennes clés supprimées (D4).
- Réduction des fichiers `.py` de 25 à $\le 12$ ; bilan LOC avant/après dans le compte rendu (objectif indicatif : $-35\,\%$).

**Risque** : les messages d'erreur Pydantic remplacent les anciens messages en français ; adapter les tests qui vérifient des sous-chaînes.

### Étape 3 : Plomberie numérique maison vs primitives `dolfinx` / PETSc (avec point de décision)

**Objectif** : alléger solveur et post-traitement, mettre à jour les appels dépréciés. Chaque sous-tâche est indépendante et **réversible** (un commit par sous-tâche).

**3.1 Appels dépréciés** : remplacer l'accès à `f.x.petsc_vec` dans `utils.petsc_vec` par l'API actuelle de la version installée (`dolfinx.la.petsc.create_vector_wrap` à vérifier). Objectif : zéro `DeprecationWarning` sur le cas O-ring.

**3.2 Solveur non linéaire (point de décision)**
- Prototyper dans une branche : `PETSc.SNES` (via `dolfinx.fem.petsc.NonlinearProblem`, signature exacte à vérifier dans la version installée) + MUMPS, en conservant la boucle externe de pilotage de charge / cutback autour de `SNES.solve()`.
- Tester les recherches linéaires `bt`, `basic`, `l2` sur O-ring. Attention : la dérivée de `ufl.conditional` (contact, retour radial) est discontinue ; le backtracking maison (décroissance stricte de $\|R\|$) tolère ces sauts et `SNES` peut stagner.
- **Règle de décision** : adopter `SNES` seulement si le cas O-ring converge avec le même nombre d'incréments (±10 %) et sans cutback supplémentaire. Sinon, **conserver `_newton_increment`** (le nettoyer, ne pas le remplacer) et le consigner dans le compte rendu.
- Dans les deux cas, fusionner `linear_solver.py` et `solver.py` si ce n'est pas déjà fait.

**3.3 Projection $L^2$ (optionnel)** : remplacer `projection.py` par `dolfinx.fem.Expression` + `Function.interpolate()` **seulement si** la perte du lissage $L^2$ condensé $P_1$ sur les champs de contrainte est acceptable (l'interpolation directe aux nœuds $P_2$ n'est pas lissée entre éléments). Si le lissage est jugé utile, conserver `projection.py`.

**Validation (bloquante)**
- O-ring converge sans avertissement de dépréciation.
- Réactions finales et $\|\mathbf{u}\|_\infty$ : écart relatif $\le 10^{-6}$ par rapport à `reference/` (si `SNES` est adopté, le chemin de convergence change : l'écart dépend de la tolérance Newton ; resserrer à $10^{-8}$ si cette tolérance le permet). Si 3.3 est appliqué, les champs de contrainte exportés peuvent légitimement différer : documenter l'écart au lieu de le comparer à $10^{-8}$.
- Run série et np=2.

**Risque principal** : comportement de la recherche linéaire native face à $J \to 0$ et à la non-régularité du contact.

## 6. Dépendances cibles

- `python >= 3.10`, `fenics-dolfinx == 0.11.0` (avec `basix`, `ufl`, `ffcx`), `petsc4py` compilé avec **MUMPS**, `mpi4py`
- `numpy`, `netCDF4`, `pydantic >= 2`, `pyyaml`
- Supprimées : `gmsh`. `scipy` : à ne garder que si une ligne du code l'utilise réellement (vérifier avec `grep`).

## 7. Tests cibles

- `tests/test_exodus.py` : lecture `HEX8` / `HEX20`, permutations, détection d'éléments inversés, sidesets.
- `tests/test_contact.py` : jauge $g_N$, normale $\mathbf{n}$, signe du frottement (cas 0D analytique).
- `tests/test_config.py` : rejets listés en 2.3.
- `tests/test_oring_golden.py` : non-régression bout en bout contre `reference/`.

Les fichiers `tests/test_mesher_exodus.py` et `tests/test_contact_geometry.py` sont cités dans les docstrings de l'ancien code mais leur présence n'a pas été confirmée : vérifier dans `<OLD>` et les reprendre (renommés) s'ils existent.

## 8. Points à vérifier avant ou pendant le travail

1. Contenu réel de `tests/` et de `DESIGN_CONTACT.md` dans `<OLD>` (cités dans les docstrings, non vus).
2. Signature exacte de `NonlinearProblem` et de `create_vector_wrap` dans la version `dolfinx` installée.
3. Si le golden de la sortie console (cité dans `utils.configure_logging`) existe : il changera à l'étape 2 (suppression de `vm_max`), le régénérer et le consigner.

## 9. Définition de « terminé »

- `grep -ri gmsh` vide ; plus aucune référence à Neumann, élasticité linéaire, GAMG, registres.
- Cas O-ring OK en série et np=2, dans les tolérances de chaque étape.
- Structure plate $\le 12$ fichiers `.py`, tests verts, aucun `DeprecationWarning`.
- Compte rendu final : bilan avant/après (fichiers, LOC, dépendances), décision SNES documentée, points ouverts.
