#!/bin/bash
#SBATCH --job-name=test-LMA-q
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=2
#SBATCH --gres=gpu:tesla-smx2:2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-cpu=12G
#SBATCH --partition=LMA
#SBATCH --time=00:10:00

srun hostname
