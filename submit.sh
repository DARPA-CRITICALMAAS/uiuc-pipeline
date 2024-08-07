#!/bin/bash
### job info
#SBATCH --job-name="CMAAS Pipeline"
#SBATCH --account=bbym-hydro
#SBATCH --time=04:00:00

### Node Details
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1

### set a file name for the stdout and stderr from the job
### the %j parameter will be replaced with the job ID.
### By default, stderr and stdout both go to the --output
### file, but you can optionally specify a --error file to
### keep them separate
#SBATCH --output=logs/slurm/job_%j.o
#SBATCH --error=logs/slurm/job_%j.e

mkdir -p "logs/slurm"

srun ./start_pipeline.sh
