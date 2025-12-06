#!/bin/bash
#
#SBATCH -w gpu-teach-01
#SBATCH -c 4 # number of cores
#SBATCH --mem=4G  # memory per core 
#SBATCH --gpus=1
#SBATCH -t 0-4:00 # time (D-HH:MM), max is 4 hours
#SBATCH -o job_logs/scripts/slurm.%N.%j.out   # STDOUT log
#SBATCH -e job_logs/scripts/slurm.%N.%j.err   # STDERR log
#SBATCH --mail-user=audreanne.bernier@mail.mcgill.ca
#SBATCH --mail-type=BEGIN,END,FAIL

module load miniconda/miniconda-fall2024

python /home/2024/aberni24/Courses/COMP551/Assignment4/scripts/bert_train.py \
       /home/2024/aberni24/Courses/COMP551/Assignment4/