#!/bin/bash
#SBATCH --job-name=tf_mirrored
#SBATCH --account=bemv-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=tf_mirror.out
#SBATCH --error=tf_mirror.err

echo "---------------------------------------"
echo "Job Started"
echo "User:        $USER"
echo "Job ID:      $SLURM_JOB_ID"
echo "Time:        $(date)"
echo "Compute IP:  $(hostname -I | awk '{print $1}')"
echo "Submit IP:   $SLURM_SUBMIT_HOST"
echo "---------------------------------------"

# setting up the environment
ml tensorflow-conda
source $CONDA_PREFIX/etc/profile.d/conda.sh && conda activate base
module load aws-ofi-nccl/1.14.2


# Run your script
python mirrored.py