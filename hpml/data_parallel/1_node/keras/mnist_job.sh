#!/bin/bash
#SBATCH --job-name=tf_multiworker
#SBATCH --partition=gpu2v100
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:20:00
#SBATCH --output=tf_out.out   # Saves standard output to a file
#SBATCH --error=tf_error.err    # Saves errors to a separate file

# 1. Load the Anaconda module
module load anaconda3

# 3. Environment Setup (setup is only required one time)
# conda create --name tf310 python=3.10 -y
# conda activate tf310_216

# pip install --upgrade pip
# pip install "tensorflow[and-cuda]==2.16.1"
# pip install tf_keras


conda activate tf310_216

# 4. Execute the Python Script
srun python mnist_multigpu.py
