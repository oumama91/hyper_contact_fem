#!/usr/bin/env bash
# Charge conda puis active l'environnement fenicsx-env avant d'exécuter la commande.
# Un shell WSL non interactif ne charge pas toujours l'initialisation de conda :
# on ne compte donc pas sur ~/.bashrc.
set -euo pipefail

CONDA_BASE="/home/oudan/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate fenicsx-env

exec "$@"
