#!/bin/bash
#SBATCH --job-name=glm_first_level
#SBATCH --time=00:20:00
#SBATCH --partition=normal,russpold
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --array=0-74

set -euo pipefail

source $SCRATCH/junkyard/.venv/bin/activate

python $SCRATCH/junkyard/run_first_level_fmri.py $SLURM_ARRAY_TASK_ID
