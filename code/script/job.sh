#!/bin/bash
#SBATCH --job-name=SGD-Train
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:tesla-smx2:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-cpu=12G
#SBATCH --partition=LMA
#SBATCH --time=00:10:00

export HOROVOD_CUDA_HOME=$CUDA_HOME
export HOROVOD_GPU_OPERATIONS=NCCL

module load anaconda/3.7
module load cuda/11.3.0

unset CONDA_ENVS_PATH

eval "$(conda shell.bash hook)"
conda activate data-science

mpiexec -np ${SLURM_NTASKS} \
    -bind-to none -map-by slot \
    -genv NCCL_DEBUG WARN \
    -genvlist LD_LIBRARY_PATH,PATH \
    python multi-fashion-minst.py
