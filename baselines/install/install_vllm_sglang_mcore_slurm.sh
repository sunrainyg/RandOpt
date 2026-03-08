#!/bin/bash
#SBATCH --job-name=
#SBATCH --output=
#SBATCH --error=
#SBATCH --partition=
#SBATCH --nodes=
#SBATCH --cpus-per-task=
#SBATCH --mem=
#SBATCH --time=

set -euo pipefail

bash scripts/install_vllm_sglang_mcore.sh
