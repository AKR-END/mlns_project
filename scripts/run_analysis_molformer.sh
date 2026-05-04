#!/bin/bash
#SBATCH -A akr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=5G
#SBATCH --time=00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
set -euo pipefail

ENV="${MLNS_ENV:-/ssd_scratch/akr/envs/mlns}"
source "$ENV/bin/activate"

cd "$(dirname "$0")/.."
python -u src/run_analysis.py --ligand MolFormer
