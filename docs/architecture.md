# Architecture du code

## Modules

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

## Flux d'exécution

1. `argparse` et journalisation (rang 0) ; `config.load_config` : validation Pydantic et résolution
   des chemins relatifs au YAML.
2. `exodus_mesh.build_mesh` : lecture et validation rang 0, distribution MPI, tags de facettes.
3. `forms.build_forms` puis `dirichlet.apply_bcs` : formulation mixte, résidu, tangente, DDL.
4. `solver.solve_newton` avec `reactions.ReactionHistory` : incréments, Newton, cutbacks.
5. `postproc.postprocess`, `write_csv`, `print_summary` : projection, métriques, VTX, CSV.

## Dépendances

- `config.py` ne dépend d'aucun module du projet ; il porte aussi le catalogue des champs pour
  éviter un import circulaire avec `postproc.py`.
- `utils.py` est la couche partagée (journalisation, PETSc, intégrales) et ne dépend que des
  bibliothèques externes.
- `contact.py` et `material.py` fournissent les lois physiques à `forms.py`.
- `forms.py` construit la formulation ; `dirichlet.py`, `solver.py`, `reactions.py` et
  `postproc.py` consomment les objets produits.
- `main.py` est le seul point d'entrée qui assemble toutes les couches.

## Règles de rangement

- **Code** : modules Python du solveur à la racine du dépôt pour l'instant ; ils sont destinés à
  être regroupés dans un paquet `hyper_contact_fem/` sans changer leurs rôles.
- **Entrées** : `mesh/` (maillage Exodus et journal Cubit) et `studies/` (cas YAML).
- **Références** : `reference/` contient uniquement la vérité de non-régression, jamais des sorties
  de run courantes.
- **Sorties** : `results/`, ignoré par Git et créé à l'exécution.
- **Outillage** : `tools/` pour les scripts de capture/comparaison et le wrapper WSL.
- **Tests** : `tests/`, plats ; les helpers partagés doivent aller dans `conftest.py` plutôt que
  d'importer un fichier de test depuis un autre.
- **Documentation** : `docs/` pour les sujets détaillés ; le `README.md` reste une porte d'entrée.
