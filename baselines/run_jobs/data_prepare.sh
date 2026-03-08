#!/bin/bash
#SBATCH --job-name=
#SBATCH --nodes=
#SBATCH --partition=
#SBATCH --account=
#SBATCH --ntasks-per-node=
#SBATCH --cpus-per-task=
#SBATCH --gres=gpu:
#SBATCH --time=
#SBATCH --output=
#SBATCH --error=
#SBATCH --environment=

srun --cpu-bind=none -ul --container-writable bash -c "
    export PYTHONNOUSERSITE=13
    python3 examples/data_preprocess/gsm8k.py --local_save_dir data/gsm8k/
"