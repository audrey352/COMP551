#!/bin/bash
#
#SBATCH -w gpu-teach-01
#SBATCH -c 4 # number of cores
#SBATCH --mem=4G
#SBATCH --gpus=1
#SBATCH -t 0-2:00 # time (D-HH:MM)
#SBATCH -o job_logs/scripts/slurm.%N.%j.out # STDOUT
#SBATCH -e job_logs/scripts/slurm.%N.%j.err # STDERR
#SBATCH --mail-user=<your_name>@mail.mcgill.ca
#SBATCH --mail-type=BEGIN,END,FAIL

module load miniconda/miniconda-fall2024

python /home/2024/aberni24/Courses/COMP551/Assignment3/gridsearch_L2.py 