#!/bin/bash
#SBATCH --job-name=ec
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=10G
#SBATCH -e ./results/slurm-%A_%a.err
#SBATCH -o ./results/slurm-%A_%a.out
#SBATCH --array=0-4

python -u transform.py \
    --jobid=$SLURM_ARRAY_TASK_ID \
    --path=./results \
    --init_num_items=${1}