#!/bin/bash
#SBATCH -A akr
#SBATCH -n 9
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=5G
#SBATCH --time=00:00:00
#SBATCH --mail-type=END
set -euo pipefail

ENV="${MLNS_ENV:-/ssd_scratch/akr/envs/mlns}"
source "$ENV/bin/activate"

mkdir -p /ssd_scratch/akr/data/pdbbind_v2016 && ln -sfn /home2/akr/datasets/pdbbind_v2016/v2016 /ssd_scratch/akr/data/pdbbind_v2016/v2016

cd "$(dirname "$0")/.."
python -u src/run_analysis.py --ligand Morgan
