#!/usr/bin/env bash
# Charge conda puis active l'environnement fenicsx-env avant d'exécuter la commande.
# Un shell WSL non interactif ne charge pas toujours l'initialisation de conda :
# on ne compte donc pas sur ~/.bashrc.
set -euo pipefail

CONDA_BASE="/home/oudan/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate fenicsx-env

# Reproductibilité bit à bit des références : MUMPS factorise via un BLAS multi-thread, dont
# les réductions ne sont pas déterministes (écarts observés ~1e-15 sur u_max). On force un
# thread par défaut ; une valeur déjà fixée par l'appelant reste prioritaire.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

exec "$@"
