#!/bin/bash
#SBATCH -A akr
#SBATCH -n 9
#SBATCH --mem-per-cpu=6G
#SBATCH --time=00:00:00
#SBATCH --mail-type=END
set -euo pipefail

mkdir -p logs
source /ssd_scratch/akr/envs/sam3_train/bin/activate 2>/dev/null || \
  conda activate sam3_train 2>/dev/null || true

cd /home2/akr/mlns_project
python -u run_extra.py
