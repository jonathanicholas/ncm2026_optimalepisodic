#!/bin/bash
#SBATCH --job-name=ec
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH -e ./results/slurm-%A_%a.err
#SBATCH -o ./results/slurm-%A_%a.out
#SBATCH --array=0-9

python -u simulate.py \
    --jobid=$SLURM_ARRAY_TASK_ID \
    --path=./results \
    --init_num_items=${1}
