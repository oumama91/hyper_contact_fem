# Maillage O-ring

Ce dossier contient les entrées de maillage du cas de référence.

| Fichier | Rôle |
|---|---|
| `O-ring.exo` | maillage Exodus II `HEX20` utilisé par `studies/O-ring_compression.yaml` |
| `O-ring.jou` | journal Cubit/Trelis qui régénère `O-ring.exo` |
| `O-ring.step` | géométrie source importée par le journal ; fichier local, non versionné pour l'instant |

## Régénération

Le journal `O-ring.jou` importe `O-ring.step` depuis ce dossier, construit le maillage `HEX20`,
nomme le bloc et les sidesets, puis exporte `O-ring.exo`. Il doit être exécuté depuis `mesh/` pour
que l'import et l'export utilisent les chemins relatifs.

Les noms de sidesets produits par le journal doivent rester alignés sur les `boundaries` du YAML :
`symmetry_x`, `symmetry_y`, `symmetry_z` et `compression`. Les contraintes de lecture Exodus
(3D, un seul bloc volumique, faces de bord, déterminants positifs) sont détaillées dans
[`docs/usage.md`](../docs/usage.md#maillage-exodus).

`O-ring.step` est nécessaire pour rejouer le journal, mais n'est pas suivi par Git à ce stade ;
si sa conservation devient permanente, l'ajouter explicitement au dépôt et mettre à jour cette
note.
