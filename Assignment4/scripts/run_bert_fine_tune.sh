#!/bin/bash
#
#SBATCH -w gpu-teach-01
#SBATCH -c 4 # number of cores
#SBATCH --mem=4G  # memory per core 
#SBATCH --gpus=1
#SBATCH -t 0-4:00 # time (D-HH:MM)                          # MAX TIME IS 4 HOURS
#SBATCH -o job_logs/scripts/slurm.%N.%j.out # STDOUT        # LOCATION TO SAVE STDOUT LOGS **IN YOUR OWN MIMI ACCOUNT**
#SBATCH -e job_logs/scripts/slurm.%N.%j.err # STDERR        # LOCATION TO SAVE STDERR LOGS **IN YOUR OWN MIMI ACCOUNT**
#SBATCH --mail-user=audreanne.bernier@mail.mcgill.ca        # PUT YOUR EMAIL HERE
#SBATCH --mail-type=BEGIN,END,FAIL

module load miniconda/miniconda-fall2024  # LOAD THE MINICONDA MODULE (has all the packages we need alrrady installed)

python /home/2024/aberni24/Courses/COMP551/Assignment4/scripts/bert_fine_tune.py  /home/2024/aberni24/Courses/COMP551/Assignment4/