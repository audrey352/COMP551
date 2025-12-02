#!/bin/bash
#
#SBATCH -w gpu-teach-01
#SBATCH -c 4                     # number of CPU cores
#SBATCH --mem=2G                 # memory per CPU core
#SBATCH --gpus=1                 # request 1 GPU per job
#SBATCH -t 0-4:00                # max time 4 hours
#SBATCH --propagate=NONE         # IMPORTANT for long jobs
#SBATCH -o job_logs/scripts/slurm.%N.%j.out   # STDOUT log
#SBATCH -e job_logs/scripts/slurm.%N.%j.err   # STDERR log
#SBATCH --mail-user=audreanne.bernier@mail.mcgill.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --array=0-2              # <<< RUN 3 JOBS (batch sizes: 4, 8, 16)

module load miniconda/miniconda-fall2024

# Define batch sizes for array jobs
BATCH_SIZES=(4 8 16)

# Select batch size based on SLURM_ARRAY_TASK_ID
BS=${BATCH_SIZES[$SLURM_ARRAY_TASK_ID]}

echo "Running BERT fine-tuning with batch size: $BS"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "GPU Assigned: $CUDA_VISIBLE_DEVICES"

python /home/2024/aberni24/Courses/COMP551/Assignment4/scripts/bert_fine_tune.py \
       --batch_size $BS \
       /home/2024/aberni24/Courses/COMP551/Assignment4/
