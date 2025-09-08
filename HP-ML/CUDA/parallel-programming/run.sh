#!/bin/bash
#SBATCH -J hello_world_cudac
#SBATCH -o hello_world_out.txt
#SBATCH -p gpu1v100
#SBATCH -t 00:05:00
#SBATCH -N 1
#SBATCH -n 1
ml cuda/toolkit/11.2
./helloworld-2 #the executable obtained after compiling the program