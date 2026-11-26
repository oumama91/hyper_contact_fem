# Essai de PETSc.SNES : résultats et conclusion

Décision historique archivée : le solveur retenu reste la boucle de Newton avec recherche linéaire
par backtracking de `hyper_contact_fem/solver.py`. Aucun code SNES ne subsiste dans le dépôt.

Une variante du solveur a été évaluée : `dolfinx.fem.petsc.NonlinearProblem` (dolfinx 0.11.0,
PETSc 3.25.6, LU direct MUMPS) avec la boucle externe de pilotage de charge / cutback inchangée,
`snes_rtol = 1e-6`, `snes_atol = 1e-10`, `snes_stol = 0`, `snes_max_it = 30` et les recherches
linéaires PETSc `basic`, `bt` et `l2`.

La variante sévère `studies/O-ring_compression_severe.yaml` sert de test de robustesse : copie
du cas de référence dont seule la position du plan de contact change (z = 1.8 mm → 0.6 mm, soit
un écrasement de 1.4 mm = 70 % du rayon de la section torique) ; `study_name` est adapté pour ne
pas écraser les sorties de référence. L'amplitude a été retenue après des essais à 1.6, 1.4,
1.2, 1.0, 0.8 et 0.6 mm (convergents) puis 0.2 mm (échec du solveur).

| Étude | Solveur | Incréments | Itér. Newton | Cutbacks | `u_max` [mm] | `total_Fz` [N] | Temps [s] |
|---|---|---|---|---|---|---|---|
| O-ring | Newton (`hyper_contact_fem/solver.py`) | 1 | 10 | 0 | 4.664172049663e-01 | 2.480275894546e+01 | 82 |
| O-ring | SNES `basic` | 1 | 10 | 0 | 4.664172049663e-01 | 2.480275894546e+01 | 89 |
| O-ring | SNES `bt` | 1 | 10 | 0 | 4.664172049663e-01 | 2.480275894546e+01 | 86 |
| O-ring | SNES `l2` | 1 | 10 | 0 | 4.664172050253e-01 | 2.480275894804e+01 | 87 |
| sévère | Newton (`hyper_contact_fem/solver.py`) | 1 | 5 | 0 | 1.108784995715e+01 | 1.229537808709e+03 | 47 |
| sévère | SNES `basic` | 0 | — | 9 | échec (PETSc -4) | — | 164 |
| sévère | SNES `bt` | 0 | — | ≥ 6 | échec (PETSc -5) | — | > 1800 |
| sévère | SNES `l2` | 0 | — | ≥ 5 | échec (PETSc -5) | — | > 1800 |

Temps = durée totale du processus (maillage, résolution, post-traitement), série, threads BLAS
à 1. Les essais `bt` et `l2` sur la variante sévère ont été interrompus après 30 min : chaque
tentative échouait déjà (raison PETSc -5, norme de résidu non finie) et le cutback rejouait le
même problème.

Critère (i) satisfait sur l'O-ring : mêmes incréments (1), aucun cutback, et écarts au résultat
de référence nuls pour `basic` et `bt` (mêmes itérés) et ≤ 1.3e-10 pour `l2`. Critère (ii) non
satisfait : sur la variante sévère, le solveur Newton converge en 5 itérations sans cutback,
alors que les trois recherches linéaires SNES échouent. Conclusion : le solveur de Newton avec
recherche linéaire par backtracking est conservé.

Remarque sur les cutbacks de cette étude : les conditions de Dirichlet sont nulles, la gravité
est nulle et la traction de contact n'est pas multipliée par le facteur de charge λ. Le problème
résolu est donc indépendant de λ : après un échec, réduire le pas rejoue exactement le même
problème depuis le même état. Vérifié à z = 0.2 mm : les 8 cutbacks échouent avec le même résidu
libre (‖R‖ = 1.818e+02), puis l'incrément est abandonné. Une variante sévère obtenue par la seule
amplitude de compression ne peut donc pas produire de cutback récupérable ; elle reste un test
de la résolution d'un incrément unique.
