# Audit statique et plan de refactoring

**Dépôt :** `D:\Recherche\hyper_contact_fem`  
**Révision :** `07d12ae0f529f84e3c77cd3cef59a6b4d80b3494` (`v1.0`)  
**Méthode :** lecture seule ; ni solveur, ni pytest, ni outils, ni analyseurs statiques exécutés.

## 1. TL;DR

Le code est compact et déjà découpé en responsabilités globalement saines : il ne justifie **ni réécriture ni refactor numérique large**. Trois défauts sont toutefois prioritaires :

1. **F-001 — réactions : confusion de numérotations.** Des indices d'un sous-espace sont utilisés directement dans le vecteur de l'espace mixte (`forms.py:42-45`, `dirichlet.py:84-89`, `reactions.py:53-55`). Déplacements et champs peuvent rester corrects tandis que réactions et bilan sont faux. Une vérification indépendante est requise : le golden ne fait que reproduire les références produites par cette implémentation.
2. **F-002 — contrôle du jacobien moins complet qu'annoncé.** Un seul déterminant par élément est vérifié (`exodus_mesh.py:183-193`), contre un contrôle « aux coins » annoncé (`README.md:166-170`). Les nœuds milieux de HEX20 ne sont pas contrôlés.
3. **F-003 — risque de blocage MPI sur erreur de sonde.** Une exception locale de GJK peut survenir avant la synchronisation collective prévue (`exodus_mesh.py:341-351`, `exodus_mesh.py:402-407`).

Suivent les singularités cylindre/sphère (F-004), la division par zéro du comparateur (F-005), un maximum de contrainte borné par zéro (F-006), les entrées netCDF vides/non finies (F-007) et les options PETSc globales non restaurées (F-008).

**Recommandation :** ajouter d'abord les tests ciblés ; corriger ensuite les défauts isolés, un par un ; seulement après, faire les nettoyages mécaniques et les deux découpages de complexité. Ne pas regrouper une correction de réactions avec une régénération des références ni avec un refactor de Newton.

## 2. Périmètre et méthode

### Lecture effectuée

- intégralité de `README.md`, des 13 modules Python racine, de `tests/`, `tools/`, `reference/README.md`, des deux études YAML et des références JSON ;
- historique et état Git par commandes de lecture ; référence figée à la révision indiquée ci-dessus ;
- reconstruction manuelle des dépendances et de la couverture : **26 fonctions `test_*`, 42 cas après paramétrisation** (27 configuration, 8 contact, 6 Exodus, 1 golden), en accord avec `README.md:238-239`.

### Contraintes de l'audit

Aucun diagnostic n'a été rejoué. Les positions ruff/vulture/radon/mypy fournies sont traitées comme des **indices**, vérifiés dans le code. Les messages mypy complets et les versions des stubs n'étant pas disponibles, les corrections qui dépendent d'une surcharge dolfinx/PETSc restent conditionnelles à une vérification documentaire.

Aucun `ruff.toml`, `.ruff.toml`, `pyproject.toml` ou équivalent n'est présent dans le dépôt ni dans son historique ; les emplacements de configuration utilisateur usuels inspectés n'en ont pas révélé. La provenance exacte des règles TRY/PYI/PLW/RUF100 n'est donc pas établie : utiliser à l'avenir une configuration de dépôt explicite, sans changer le code à cette seule fin.

### Gravité

- **Critique :** résultat potentiellement faux ou blocage sur un chemin normal, à traiter avant refactor ;
- **Élevée :** défaut conditionnel important ou garde-fou de non-régression défaillant ;
- **Moyenne :** robustesse, couverture ou maintenabilité dégradée sans preuve d'impact sur le cas de référence ;
- **Faible :** nettoyage local, documentation ou typage sans effet attendu à l'exécution.

Les complexités radon sont reprises comme indices de repérage, pas comme preuves de défaut.

## 3. Cartographie du projet

### Dépendances internes

```text
config ──> contact ──> forms ──> material ──> utils
   ▲                   │  ▲         ▲          ▲
   │                   │  └─────────┘          │
   │              dirichlet                    │
   │                   │                       │
exodus_reader ──> exodus_mesh                 │
                     │                         │
main ────────────────┼──> solver ──────────────┤
  │                  ├──> reactions ───────────┤
  │                  └──> postproc ─> projection ┘
  └──> tests et tools/capture_metrics réutilisent ce pipeline
tools/compare_reference : JSON seulement, indépendant de dolfinx
```

`config` est sans dolfinx ; `exodus_reader` aussi. `utils` centralise le scalaire dolfinx, les expressions UFL, le JIT, les tailles de DDL et l'assemblage PETSc. Aucun cycle d'import fonctionnel n'est apparu ; la localisation du catalogue de champs dans `config.py:42-75` évite explicitement un cycle avec `postproc`.

### Flux et zones sensibles

1. `main._run` (`main.py:92-114`) orchestre configuration → maillage → formes/Dirichlet → Newton/réactions → post-traitement/CSV/résumé.
2. `build_mesh` lit sur le rang 0, passe des tableaux vides aux autres rangs et construit les tags ; le contournement `create_mesh` pour dolfinx 0.11.0 (`exodus_mesh.py:374-388`) est à conserver.
3. `build_forms` construit Taylor-Hood, résidu, tangente et charges (`forms.py:82-137`).
4. `solve_newton` et `_newton_increment` assurent pilotage, relèvement, LU MUMPS, backtracking et restauration (`solver.py:151-340`).
5. `compute_reactions`, `postprocess` et `write_vtx` produisent les métriques et fichiers.

Sont particulièrement sensibles : les collectifs MPI, l'ordre des opérations flottantes, le partage et la remise à zéro des vecteurs PETSc, `ghostUpdate`, la durée de vie de la matrice et des cinq vecteurs, ainsi que `PETSc.Options()` dans `LinearSolver`. Le format VTX impose aussi des espaces d'export de même numérotation (`postproc.py:4-9`, `projection.py:28-42`, `projection.py:72-93`).

## 4. Constats par module

### F-001 — Réactions calculées avec des indices du mauvais espace
**Catégorie :** exactitude / typage de contrat — **Gravité : critique**

**Preuves.** `Forms.component_dofs` annonce des indices de `V`, mais appelle `locate_dofs_topological` avec le seul sous-espace (`forms.py:42-45`). Ces indices conviennent à `dirichletbc` (`dirichlet.py:91-92`), pas directement au vecteur assemblé de `V`. Or `DirichletSet` remplit `owner` et `comp_of`, dimensionnés pour `V`, avec ces mêmes indices (`dirichlet.py:77-89`) ; `compute_reactions` sélectionne ensuite `r[constrained]` dans le vecteur mixte (`reactions.py:41-55`). Le filtre `bc.dofs < n_flat` ne convertit ni n'exclut correctement les fantômes. La conversion via `collapse()` existe déjà pour les DDL possédés (`forms.py:47-52`), ce qui confirme la distinction.

**Impact.** Réactions par frontière, total, `imbalance`, CSV/console/JSON potentiellement faux. Le check de conflits (`dirichlet.py:39-56`) n'est pas remis en cause : il compare des indices de sous-espaces de même composante. La solution, Newton et les champs ne devraient pas changer lors de la correction.

**Limite.** La numérotation exacte doit être démontrée par un test sur dolfinx 0.11.0 ; ne pas se contenter d'une apparente coïncidence sur un petit espace. Avant toute nouvelle référence, comparer chaque réaction à `sum(r[parent_dofs])` calculé indépendamment, pour deux frontières sécantes, en série et `np=2`.

### F-002 — Jacobien vérifié en un point, pas « aux coins »
**Catégorie :** validation géométrique / documentation — **Gravité : critique**

`_check_mesh` calcule le déterminant des trois arêtes issues du coin 0 (`exodus_mesh.py:183-193`). Cela ne prouve pas la positivité aux autres coins d'un HEX8 gauchi, ni dans un élément HEX20. La docstring parle pourtant des coins (`exodus_mesh.py:176-181`) et le README promet ce contrôle (`README.md:166-170`). Le nombre de cellules par facette (`exodus_mesh.py:194-201`) détecte certaines non-conformités, mais pas tous les nœuds pendants.

**Action.** Ajouter des cas HEX8 tordus et HEX20 avec nœud milieu replié, puis évaluer la géométrie effective aux huit sommets avec Basix/dolfinx 0.11.0. Documenter séparément ce que ce contrôle ne garantit pas pour les points intérieurs. Ne pas prétendre à une validation globale de tout HEX20 sans échantillonnage/contrôle supplémentaire explicitement choisi.

### F-003 — Synchronisation MPI trop tardive autour des sondes
**Catégorie :** robustesse MPI — **Gravité : critique**

Dans `_check_probe_points`, le préfiltrage et GJK précèdent `allreduce` (`exodus_mesh.py:341-351`). Le `try` extérieur (`exodus_mesh.py:402-407`) ne rejoint `_raise_collective` qu'après sortie de toute la fonction. Si un rang échoue dans GJK et les autres atteignent un autre `allreduce`, les collectifs ne correspondent plus : blocage possible.

**Action.** Capturer l'erreur autour de **chaque calcul local précédant un collectif**, diffuser l'état avant le `allreduce`, puis utiliser une valeur neutre sur le rang fautif et lever une erreur homogène sur tous les rangs. Faire de même pour les exceptions des tests de tolérance. Tests `np=2` avec exception GJK injectée sur un seul rang, plusieurs points et sideset vide ; vérifier terminaison bornée.

### F-004 — Normales indéfinies sur axe de cylindre / centre de sphère
**Catégorie :** robustesse numérique — **Gravité : élevée**

Les divisions par `r` sont non protégées (`contact.py:68-85`). La normale géométrique est réellement indéfinie en `r=0`. Pour `orientation="inside"`, un point singulier peut être admissible avec pression nulle, mais l'expression `p_n * normal` peut tout de même contenir `0/0` (`contact.py:117-126`). Le golden plan n'exerce pas ces branches.

**Action.** Décision utilisateur préalable : convention locale documentée ou rejet explicite des géométries contenant la singularité. Ne pas ajouter arbitrairement un epsilon global à `r`, qui changerait jauge, normale et tangente au voisinage. Tests analytiques au centre/sur l'axe et à proximité, pour les deux orientations.

### F-005 — Comparateur fragile sur zéros et structure JSON
**Catégorie :** filet de sécurité / robustesse — **Gravité : élevée**

`_rel_error` divise par `max(|ref|, floor)` (`tools/compare_reference.py:45-50`), mais les scalaires n'ont aucun plancher (`tools/compare_reference.py:130-135`) : référence `0`, run non nul ⇒ `ZeroDivisionError`. Le comparateur ignore les réactions supplémentaires du run, une réaction manquante lève `KeyError`, et `mpi_size` du run n'est pas comparé à celui de la référence alors que la tolérance dépend de ce dernier (`tools/compare_reference.py:110-160`). Les champs, eux, vérifient bien la liste (`tools/compare_reference.py:70-77`).

**Action.** Retourner un échec de validation, pas une exception ; comparer les ensembles et tailles ; décider explicitement si les tailles MPI doivent correspondre. Conserver les planche­rs de réaction/champs et la règle NaN/NaN. Tests unitaires purs Python : zéro/zéro, zéro/non-zéro, NaN, infini, réaction manquante/supplémentaire et croisement série/np2.

### F-006 — `sigma_1_max` faux si toutes les valeurs sont négatives
**Catégorie :** post-traitement / MPI — **Gravité : élevée**

Le neutre d'un rang vide est `0.0` dans `_owned_max` (`postproc.py:58-60`), réutilisé pour la contrainte (`postproc.py:206-210`). Un état entièrement compressif produit donc `0`, pas le maximum négatif. Le neutre correct d'un maximum est `-np.inf`.

**Action.** Tester `[-5, -2] → -2`, rang vide + rang négatif, puis déplacement nul/non nul. Cette correction ne doit rien changer au cas O-ring dont la contrainte maximale est positive.

### F-007 — Entrées netCDF vides, masquées ou non finies mal rejetées
**Catégorie :** validation d'entrée — **Gravité : moyenne**

Les coordonnées ne subissent aucun contrôle `isfinite` (`exodus_reader.py:122-130`) ; les variables masquées sont converties en `NaN`, puis potentiellement en `int64` (`exodus_reader.py:46-49`, `exodus_reader.py:132-140`, `exodus_reader.py:156-157`). `cells.min()/max()` est appelé sans cas vide (`exodus_reader.py:132-136`) ; `_check_mesh` utilise également `carried_by.max()` (`exodus_mesh.py:194-201`). Enfin, un jacobien `NaN` n'est pas rejeté par `jacobian <= 0` (`exodus_mesh.py:185-193`).

**Action.** Rejeter explicitement maillage sans nœud/cellule, formes invalides, données masquées et valeurs non finies avant conversion entière. Tester zéro élément/nœud, NaN/Inf, connectivité et sidesets partiellement masqués. Conserver un sideset vide s'il est actuellement accepté ; son rejet éventuel par les sondes est un autre contrat.

### F-008 — `PETSc.Options()` global : valeurs préexistantes effacées et `_ksp` non gardé
**Catégorie :** ressource / état global / typage — **Gravité : moyenne**

`LinearSolver` enregistre seulement les clés posées et les supprime à la sortie (`solver.py:68-71`, `solver.py:90-97`) : il ne restaure pas une éventuelle ancienne valeur `fea_*`. Le flux actuel ne crée pas ces clés ailleurs, donc aucun changement du golden n'est attendu. Par ailleurs, `_ksp` est optionnel (`solver.py:65`) mais déréférencé sans garde à `solver.py:102-106` : mypy signale à juste titre ce contrat, même si `__enter__` l'initialise dans l'usage prévu.

**Action.** Sauvegarder existence et valeur des trois clés, puis restaurer en `close`; tester emboîtement et exception. Récupérer une variable KSP non optionnelle après garde dans `solve`, et définir le comportement d'un `convergedReason` nul/None avec un faux KSP. Ne pas changer le KSP choisi, son préfixe ou le critère de convergence PETSc sans décision séparée.

### F-009 — Capture de déplacement différente du champ écrit
**Catégorie :** non-régression / instrumentation — **Gravité : moyenne**

`write_vtx` interpole d'abord `u_h` dans `u_out`, puis écrit ce dernier (`postproc.py:150-178`). L'instrumentation capture pourtant `u_h` et calcule ses statistiques (`tools/capture_metrics.py:65-77`, `tools/capture_metrics.py:98-109`). Elle ne garantit donc pas sa promesse de figer les valeurs réellement écrites (`tools/capture_metrics.py:10-20`, `reference/README.md:78-92`). Sur le cas actuel les valeurs peuvent coïncider ; ce n'est pas un invariant vérifié.

**Action.** Capturer le tableau de `u_out` après interpolation, sans modifier l'API VTX ni les valeurs. Test avec un double de `write_vtx` où entrée et sortie diffèrent volontairement, et rangs fantômes.

### F-010 — Typage : erreurs projet réelles et limites de stubs
**Catégorie :** typage / maintenabilité — **Gravité : faible à moyenne**

Constats confirmés dans le code :

- `material.py:39-56` : `inv_K` est déclaré optionnel alors que `__init__` le crée toujours ; déclarer l'attribut non optionnel initialisé dans le constructeur, sans toucher au potentiel ;
- `solver.py:68-77` : seules des chaînes sont réellement passées à `_set`; resserrer le paramètre à `str` si le stub l'exige ;
- `solver.py:73` : `Self` est une amélioration de signature seulement ;
- `solver.py:176`, `contact.py:117`, `material.py:41` : paramètres publics/internes non annotés ;
- `tests/test_contact.py:124,132,140` : `p_n=100.0` ne respecte pas `Expr`; convertir en expression UFL dans les tests ;
- `exodus_mesh.py:105-110` : la normalisation `hasattr(..., "items")` sert une compatibilité Basix, pas du code mort à supprimer ; documenter un `cast` local si nécessaire ;
- `exodus_mesh.py:380-388` : types des tableaux des deux branches à déclarer explicitement si les stubs NumPy infèrent mal, sans changer allocation, dtype ni ordre.

Les alertes `solver.py:79` (communicateur), `solver.py:188` (norme polymorphe), `solver.py:239` (`assemble_matrix`), `utils.py:110-113` (`assemble_vector`), `reactions.py:55` (`weights`) et `forms.py:51-52` dépendent des signatures des stubs et/ou du narrowing. **Ne pas modifier les appels numériques à l'aveugle.** Vérifier dolfinx/petsc4py 0.11.0/3.25.6, puis employer garde locale, `cast` ou `type: ignore[code]` très ciblé avec justification. `forms.load.value = lam_try` (`solver.py:289`) peut devenir une affectation in-place uniquement après vérification de la forme/dtype du buffer.

### F-011 — Nettoyages ruff sûrs mais sans urgence numérique
**Catégorie :** style / dette mécanique — **Gravité : faible**

- `config.py:285` (`TRY004`) : `TypeError` est plus approprié que `ValueError` pour une racine YAML non-dictionnaire. **Changement du type public d'erreur**, donc à regrouper avec un test, pas à faire aveuglément ;
- imports I001 (`forms.py:1`, `main.py:1`, `reactions.py:1`, `solver.py:1`, `tools/capture_metrics.py:44`) : ordre mécanique seulement ;
- `solver.py:68,73` (PYI) : signatures uniquement, voir F-010 ;
- `tests/test_contact.py:121` (F841) : `x` inutilisé dans `test_friction_opposes_sliding_below_cone` ; le supprimer sans retirer le `ufl.SpatialCoordinate` des tests qui l'utilisent ;
- `tests/test_oring_golden.py:25,41` (PLW1510) : le golden appelle `subprocess.run` sans `check`. Ajouter `check=True` ou, mieux pour garder le message du comparateur, tester explicitement les codes retour et afficher stdout/stderr ; ne pas masquer un échec de run ;
- neuf RUF100 (`tools/capture_metrics.py:44-52`) : suppression des `# noqa: E402` seulement si la règle E402 est réellement désactivée dans la future configuration ; sinon les directives sont nécessaires après les modifications de `sys.path` (`tools/capture_metrics.py:40-42`), pas du code mort.

### F-012 — Complexité localisée : deux fonctions à découper, pas une refonte générale
**Catégorie :** SRP / complexité — **Gravité : moyenne**

`read_exodus` (radon D29, `exodus_reader.py:88-169`) mélange en-tête, bloc volumique et sidesets. `compare_reference.main` (D23, `tools/compare_reference.py:97-173`) mélange CLI, normalisation, comparaison et affichage. `resolve_fields` (C12, `config.py:78-95`) reste courte : ne pas la découper en micro-fonctions.

**Découpages proposés.** `_read_cell_block` et `_read_sidesets` pour Exodus ; `_parse_args`, `_reaction_scale`, comparaisons scalaires/compteurs et un helper de vecteurs pour le comparateur. Les deux boucles quasi identiques réactions/totaux (`tools/compare_reference.py:145-160`) justifient ce helper. Ne factoriser `main._run` et `capture_metrics.main` (`main.py:92-114`, `tools/capture_metrics.py:140-177`) qu'après tests de sorties : la duplication est réelle mais un pipeline public partagé risque de déplacer le problème.

### F-013 — Docstrings et commentaires à réaligner
**Catégorie :** documentation — **Gravité : faible**

- `exodus_mesh.py:209-210` promet une erreur « à l'identique », alors que `_ERROR_TYPES` ne conserve que trois types (`exodus_mesh.py:91-93`, `exodus_mesh.py:223-225`) ; préciser message préservé et type limité aux erreurs prévues ;
- `reference/README.md:35-38` annonce quatre chiffres pour `sigma_1_max`; `main.py:86` avec `:.4e` en affiche cinq significatifs ;
- `projection.py:3-8` généralise trop « P1 est le seul ordre » ; restreindre la justification au problème de masse concentrée visé et aux éléments supportés ;
- `forms.py:42-48` et `dirichlet.py:18-22,65` annoncent à tort la numérotation de `V`, à corriger avec F-001 ;
- `README.md:166-181` et `exodus_mesh.py:176-181` surestiment les contrôles géométriques/conformité, à aligner avec F-002.

## 5. Constats transversaux

### F-014 — Couverture insuffisante des validateurs et du comportement adaptatif
**Catégorie :** tests — **Gravité : élevée**

Les 42 cas existent, mais ne démontrent pas la couverture fonctionnelle entière :

- `tests/test_exodus.py:24-52` ne teste que permutations, bijectivité des faces, fichier valide et fichier absent ; aucun rejet de mauvaise dimension, multibloc, type, coordonnées, connectivité, `elem_num_map`, sideset, face interne, jacobien ou sonde ;
- aucun test unitaire de `compute_reactions`, `DirichletSet`, `ReactionHistory.write_csv`, `print_summary`, `postprocess`, `_owned_max` ou `check_same_numbering` ;
- le golden utilise un seul incrément, 10 Newton, aucun cutback (`reference/O-ring_compression_series_metrics.json:6-9`) : il n'exerce ni réduction/agrandissement de pas, ni backtracking effectif (10 évaluations pour 10 Newton), ni restauration après échec ;
- le contact testé en mécanique complète est plan et sans frottement ; cylindre/sphère et frottement ne sont évalués qu'en cas 0D (`tests/test_contact.py`) ;
- `tests/test_oring_golden.py:21-59` lance uniquement la série ; les références np2 ne constituent pas un test automatisé np2 ;
- aucun test des schémas/ordres/ précisions CSV, VTX, JSON ou console, ni de `resolve_fields` au-delà de sa valeur par défaut (`tests/test_config.py:43-49`).

**Action.** Ajouter des tests ciblés et petits avant tout déplacement de code ; enrichir seulement les comportements existants, sans nouvelle fonctionnalité ni nouvelle sortie.

### F-015 — Le golden fige l'implémentation, pas toutes les vérités physiques
**Catégorie :** stratégie de validation — **Gravité : élevée**

Le test rejoue la séquence et la compare aux fichiers de référence (`tests/test_oring_golden.py:21-59`). C'est un excellent verrou de régression pour ce cas, mais F-001, F-006 ou une erreur systématique commune au run et à la référence restent invisibles. Les sorties physiques exigent des oracles indépendants : sommes directes du résidu pour les réactions, maximum analytique pour `_owned_max`, petits maillages reproduisant une géométrie invalide.

### F-016 — Configuration de qualité non reproductible dans le dépôt
**Catégorie :** outillage — **Gravité : faible**

Sans configuration versionnée, 28 erreurs mypy dans huit fichiers et les règles ruff fournies ne sont pas reproductibles à partir du seul dépôt. Les stubs dolfinx/petsc4py suspects doivent être distingués des erreurs projet. Ajouter éventuellement une configuration minimale et documenter les versions, sans transformer les faux positifs de bibliothèques en modifications runtime. Ce point n'exige aucun changement du schéma YAML ni des formats de sortie.

### F-017 — Résultat vulture : faux positif à conserver
**Catégorie :** tri statique — **Gravité : nulle (aucun constat de défaut)**

Le paramètre `record` de `_RankZeroFilter.filter` (`utils.py:31`) est imposé par la signature de `logging.Filter.filter`. Vulture ne voit pas cet appel polymorphe du framework : **ne pas le renommer ni le supprimer**. Les autres attributs repérés (`_MeshData.n_elems/n_nodes`, `source`, `geom_degree`, `description`, `LinearSolver.V`) ont soit un usage effectif, soit seulement un nettoyage non prioritaire possible ; ils ne justifient pas une chasse au code mort générale.

### Invariants numériques à préserver explicitement

- `assemble_vec` : remise à zéro de la forme locale propre + fantômes, assemblage, lifting avant accumulation, puis `ghostUpdate(ADD, REVERSE)` (`utils.py:90-114`) ;
- entrée `create_mesh` détenue par un seul rang, contournement documenté pour dolfinx 0.11.0 (`exodus_mesh.py:374-388`) ;
- ordre du Newton : résidu initial et `r_ref`, relèvement `alpha=-1`, `set_bc` de première itération, assemblage matrice, résolution `J du=R`, `u <- u_backup-step*du` (`solver.py:151-249`) ;
- signe du résidu et de `axpy`, référence de résidu figée, au moins une résolution, décroissance stricte et pas minimal ;
- restauration `uh.x.array` + `scatter_forward` après échec (`solver.py:288-307`) ;
- destruction différée des quatre vecteurs et de la matrice dans `finally` (`solver.py:121-149`, `solver.py:274-328`) ;
- contact non multiplié par `lambda`, contrairement à gravité/Dirichlet (`forms.py:106-125`) ;
- potentiel mixte, condition `nu=0.5`, régularisation tangentielle et projection P1 (`material.py:39-60`, `contact.py:96-114`, `projection.py:45-69`) ;
- espaces VTX de même numérotation et exclusion des fantômes dans les statistiques.

Toute modification de ces points est hors « refactor invariant » et doit être isolée en dernier, sauf correction de défaut avéré. Aucun changement de surface de référence, frottement holonome, pénalité ou limite du README n'est proposé.

## 6. Filet de sécurité à construire

### Contrats de non-régression obligatoires

Pour chaque étape marquée « invariant », exécuter le golden **inchangé**, sans régénérer `reference/` :

- série : écart relatif `<= 1e-12` sur les grandeurs comparées ;
- `np=2` : écart relatif `<= 1e-8` ;
- `increments`, `newton_iterations`, `cutbacks`, `line_search_evaluations` : **strictement égaux** ;
- threads BLAS à 1 via le wrapper documenté, mêmes versions de référence ;
- schéma YAML, noms et ordre des champs, CSV, JSON, VTX et libellés console inchangés sauf décision explicite.

Les corrections F-001, F-002 à F-007 et F-009 peuvent révéler un changement légitime de sortie dans des cas non couverts. Elles ne doivent jamais être « validées » en régénérant d'abord les références : prouver d'abord le résultat avec un oracle indépendant, documenter les grandeurs affectées, puis demander une décision pour les références.

### Tests prioritaires avant refactor

1. **Mapping Dirichlet/réactions (F-001).** Sur petit maillage mixte `P2-P1`, obtenir séparément indices sous-espace et mapping `collapse()`; vérifier `owner`, fantômes, deux frontières sécantes, dernière déclarée, somme directe du résidu. Série et np2.
2. **Maximum générique (F-006).** Tableau entièrement négatif, rang vide, rangs négatifs ; conserver le test de norme de déplacement.
3. **Comparateur (F-005).** Tester fonctions pures sur zéros, NaN, infinis, clés et longueurs. Valider que l'échec produit code 1 et non traceback.
4. **Exodus (F-002/F-007).** Générer de minuscules datasets : HEX8 gauchi avec seul coin 0 valide, chaque coin inversé, HEX20 replié, vide, NaN/Inf, masques, blocs et sidesets invalides.
5. **Sondes MPI (F-003).** Injection d'erreur GJK sur un seul rang avec terminaison bornée ; sideset vide, plusieurs sondes, série/np2.
6. **Assemblage réutilisé (invariant).** Deux assemblages dans un même vecteur comparés à un vecteur frais, callback appelé une fois, contributions fantômes non nulles en np2.
7. **Newton (invariants).** Avec petits problèmes/doubles : pas plein accepté, backtracking réel, cutback/restauration, retour au pas nominal, épuisement `max_cutbacks`, destruction des ressources après exception. Compteurs attendus explicitement.
8. **Capture/export (F-009).** Champ d'entrée distinct du champ effectivement exporté ; vérifier données et DDL possédés.
9. **Sorties publiques.** Golden byte/structure pour en-têtes CSV, clés et ordre JSON, champs VTX, messages essentiels console. Ne pas figer temps, chemins temporaires ou détails sans contrat.
10. **Géométrie de contact (F-004).** Tests au centre/sur l'axe seulement après choix de la convention ; ils sont volontairement rouges tant que le comportement n'est pas défini.

## 7. Plan ordonné et exécutable

Les efforts sont des ordres de grandeur pour une personne, hors temps du golden (~90 s série) et des diagnostics d'environnement. **Une étape = un changement vérifiable ; pas de commit automatique.**

| Étape | Objectif et IDs | Fichiers et modifications précises | Effort | Risque | Validation |
|---|---|---|---:|---|---|
| **0** | Figer le point de départ | Vérifier état propre, versions, commandes golden série/np2 et hashes des références. Ne modifier aucun code. | 0,25 j | Faible | Golden inchangé : `1e-12`/`1e-8`, compteurs égaux. |
| **1** | Tests de cartographie et correction F-001, **isolés** | `tests/` : oracle indépendant ; `forms.py`, `dirichlet.py` : distinguer indices sous-espace et parents `V`; `reactions.py` inchangé si possible. | 1 j | **Élevé pour réactions**, faible pour solution | Tests mapping série/np2 ; déplacements/champs/Newton strictement inchangés ; analyse documentée de tout delta de réaction **avant** décision sur références. |
| **2** | Robustesse du comparateur F-005 | `tools/compare_reference.py` : plancher pour zéros, ensembles de clés, tailles et MPI ; nouveaux tests purs JSON. Ne pas encore déplacer les fonctions. | 0,5 j | Faible | Zéro/non-zéro = échec propre ; références existantes acceptées ; compteurs exacts. |
| **3** | Tests et correction F-006 | `postproc.py:58-60` : neutre du maximum ; tests NumPy/MPI. Aucun autre post-traitement modifié. | 0,25 j | Faible | Maximum négatif correct ; golden inchangé. |
| **4** | Corrections mécaniques groupées F-010/F-011/F-017 | Annotations, `Self`, `inv_K`, imports, variable inutilisée, statut des sous-processus, directives ruff justifiées. Aucun `ignore` global. `TypeError` de `config.py:285` dans un petit changement testé séparé. | 0,5-1 j | Faible | Suite + golden ; même CSV/VTX/console ; revue de diff sans instruction numérique modifiée. |
| **5** | Validation netCDF F-007, sans nouveau contrat | `exodus_reader.py`, `_check_mesh` : rejets explicites déjà implicites, guards vide/fini avant conversions ; tests fixtures minimales. | 0,75 j | Moyen sur messages uniquement | Tous cas invalides rejettent proprement ; O-ring et golden inchangés. |
| **6** | Jacobien F-002, **isolé** | `exodus_mesh.py` : évaluation aux coins via API Basix 0.11 vérifiée ; documentation du périmètre HEX20 ; tests géométriques ciblés. | 1 j | **Élevé** : peut rejeter maillages | HEX8 tordu et HEX20 ; aucun changement de degré/connectivité ; O-ring accepté ; golden inchangé. |
| **7** | Synchronisation MPI F-003, **isolée** | `exodus_mesh.py:_check_probe_points` : erreurs diffusées avant collectifs, valeur neutre, levée commune. Pas de refactor de lecture mêlé. | 0,75-1 j | **Élevé** : collectifs | Injection une-rang et séries d'échecs ; terminaison np2 ; golden série/np2. |
| **8** | Capture et ressources F-008/F-009 | `tools/capture_metrics.py`, `postproc.py` : capture de `u_out`; `solver.py` : restauration options et garde KSP. Tests ciblés. | 0,5-0,75 j | Moyen | Statistiques réellement exportées ; golden ; options restaurées après succès/exception ; mêmes solveurs/compteurs. |
| **9** | Découpage `compare_reference.main` F-012 | Extraire parsing/échelle/comparaisons/helper vecteurs, puis dédupliquer les deux boucles. Conserver texte, ordre, tolérances et codes retour. | 0,5 j | Faible | Tests JSON et golden ; sortie comparée ligne à ligne sur fixtures. |
| **10** | Découpage `read_exodus` F-012 | Extraire bloc volumique et sidesets ; préserver dataclass, tableaux, dtype, ordre et messages. Laisser `resolve_fields` tel quel. | 0,5-0,75 j | Moyen | Tous tests Exodus + golden ; diff limité à la structure. |
| **11** | Documentation F-013 et outillage F-016 | Réaligner commentaires/README ; si souhaité, configuration ruff/mypy versionnée reflétant les règles déjà utilisées. | 0,25-0,5 j | Faible | Relecture ; golden si commentaires seuls non requis, obligatoire si code/configuration influence l'exécution. |
| **12** | Décision puis singularités F-004, **en dernier** | `contact.py`, tests analytiques ; aucune régularisation sans convention approuvée. | 0,75-1,5 j | **Élevé** : tangente/contact | Centre/axe et voisinage ; golden plan inchangé ; références cylindre/sphère établies par oracle indépendant. |

### Regroupement recommandé

- **Lot A (petit, sûr) :** étape 4, hors changement d'API d'erreur ;
- **Lot B (garde-fous) :** étapes 2, 3, 8 avec tests indépendants, sans déplacements structurels ;
- **Lots C isolés :** F-001, F-002, F-003, F-004 chacun séparé, car comportement ou MPI/numérique ;
- **Lot D (restructure seulement) :** étapes 9 et 10 une fois C stabilisé.

Ne pas profiter de ces travaux pour introduire nouveaux obstacles, multi-corps, autre solveur, historique de frottement ou nouveau format : ce sont des fonctionnalités hors périmètre, non des refactors.

## 8. Décisions utilisateur nécessaires

1. **F-001 / références :** après preuve indépendante, faut-il régénérer les réactions CSV/JSON/log si elles changent ? Recommandation : oui uniquement avec note de correction ; ne jamais modifier les valeurs pour préserver artificiellement le golden.
2. **F-004 / singularités :** choisir convention locale finie ou rejet explicite ; préciser le comportement `inside`. Recommandation : erreur explicite si aucune normale ne peut être définie sans hypothèse supplémentaire, plutôt qu'epsilon silencieux.
3. **F-002 / HEX20 :** le contrôle attendu est-il seulement aux sommets ou inclut-il un échantillonnage interne ? Recommandation : sommets obligatoires + limites documentées, sans promettre une preuve globale.
4. **F-005 / MPI :** faut-il refuser une comparaison si `run.mpi_size != ref.mpi_size` ? Recommandation : oui, sauf option explicite de comparaison croisée ; aucune nouvelle option nécessaire si ce besoin n'existe pas.
5. **F-011 / contrat d'erreur :** accepter `TypeError` à la place de `ValueError` pour une racine YAML non-dictionnaire ? `main.py:30` n'inclut actuellement pas `TypeError` dans les erreurs attendues ; décision nécessaire avant ruff TRY004.
6. **F-016 / outillage :** souhaite-t-on versionner les configurations ruff/mypy ? Aucun nouveau résultat public n'est nécessaire ; recommandation : oui, avec ignores locaux documentés plutôt que désactivation large.
7. **Sorties :** confirmer qu'aucun changement de schéma YAML, CSV, VTX, JSON ou console n'est autorisé par défaut. C'est l'hypothèse retenue dans tout le plan.

## 9. Points à vérifier lors de l'implémentation

- **API dolfinx 0.11.0 :** signatures et dualité des retours de `locate_dofs_topological`, `FunctionSpace.collapse`, `dirichletbc`, interpolation `u_h → u_out`, tableaux de `create_mesh`, évaluation du jacobien Basix. Vérifier dans la documentation/version réellement installée, pas par hypothèse.
- **PETSc/petsc4py 3.25.6 :** types `Comm`, `Options`, `Vec.norm`, raisons `getConvergedReason` et sémantique de `Constant.value`; conserver les appels éprouvés si seul le stub est incorrect.
- **MPI :** chaque rang doit rencontrer les mêmes collectifs dans le même ordre, erreurs comprises ; ne pas tester uniquement le rang 0 et ne pas « réparer » un deadlock en ajoutant une barrière aveugle.
- **Flottants :** ne pas réordonner sommes, réductions, projections, dérivations, options MUMPS ou recherche linéaire dans les étapes mécaniques. Des `cast`, annotations et renommages ne doivent générer aucune instruction runtime nouvelle.
- **Vecteurs PETSc :** conserver `localForm().zeroEntries()`, `ghostUpdate`, `scatter_forward`, `apply_lifting` et destruction `finally`; tester la réutilisation en np2 avant toute retouche.
- **Golden :** comparaison série `1e-12`, np2 `1e-8`, compteurs strictement égaux, threads BLAS à 1. Un golden rouge ne se résout pas par régénération.
- **F-001 :** après correction, contrôler explicitement que `u_max`, champs et compteurs sont inchangés ; seuls les objets dérivés du mauvais mapping devraient pouvoir varier.
- **Limites assumées :** ne pas transformer les rappels du README (pénalité, frottement monotone, facette de référence, LU direct, unique corps/bloc) en défauts ou fonctionnalités à ajouter.
- **Validation finale :** `git diff --check`, revue des fichiers modifiés et `git status --short`; aucun artefact `results/`, cache ou nouvelle référence mêlé au refactor.

---

**Conclusion.** Le meilleur rapport coût/risque consiste à corriger d'abord les contrats d'indices et les garde-fous, puis les deux points de complexité réellement concentrés. Le reste du solveur — notamment Newton, assemblages PETSc, formulation UFL et export commun — doit rester stable. Ce rapport n'a entraîné la création d'aucun autre fichier et aucune exécution du code.
