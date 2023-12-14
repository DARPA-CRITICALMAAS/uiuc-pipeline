#!/bin/bash
### set the wallclock time
#SBATCH --time=04:00:00

### job info
#SBATCH --partition=a100
#SBATCH --gres=gpu:1

### set the job name
#SBATCH --job-name="pipeline"

### set a file name for the stdout and stderr from the job
### the %j parameter will be replaced with the job ID.
### By default, stderr and stdout both go to the --output
### file, but you can optionally specify a --error file to
### keep them separate
#SBATCH --output=pipeline.o%j
#SBATCH --error=pipeline.e%j

module load python/3.9.13 cuda/12.2.1
. ./venv/bin/activate
python ./pipeline.py
