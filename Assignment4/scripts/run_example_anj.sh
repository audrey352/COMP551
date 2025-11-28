#!/bin/bash
#
#SBATCH -w gpu-teach-01
#SBATCH -c 4 # number of cores
#SBATCH --mem=4G  # memory per core 
#SBATCH --gpus=1
#SBATCH -t 0-4:00 # time (D-HH:MM)                          # MAX TIME IS 4 HOURS
#SBATCH -o /home/2024/atrach1/job_logs/scripts/slurm.%N.%j.out # STDOUT        # LOCATION TO SAVE STDOUT LOGS **IN YOUR OWN MIMI ACCOUNT**
#SBATCH -e /home/2024/atrach1/job_logs/scripts/slurm.%N.%j.err # STDERR        # LOCATION TO SAVE STDERR LOGS **IN YOUR OWN MIMI ACCOUNT**
#SBATCH --mail-user=anjara.trachsel-bourbeau@mail.mcgill.ca        # PUT YOUR EMAIL HERE
#SBATCH --mail-type=BEGIN,END,FAIL

module load miniconda/miniconda-fall2024  # LOAD THE MINICONDA MODULE (has all the packages we need alrrady installed)

python /home/2024/atrach1/Courses/COMP551/Assignment4/scripts/example.py   
# USE ABS PATH TO THE PYTHON SCRIPT YOU WANT TO RUN **IN YOUR MIMI ACCOUNT SINCE SCRIPTS RUN FROM YOUR HOME DIRECTORY**


# See gpu tutorial for how to set up jobs on mimi
# Then to run this script:
# 1. ssh into mimi in your terminal
# 2. module load slurm
# 3. sbatch abs path to this script Eg. sbatch /home/2024/atrach1/Courses/COMP551/Assignment4/scripts/run_example.sh