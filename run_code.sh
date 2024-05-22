#!/bin/bash -l
#SBATCH --job-name=train-nn-gpu
#SBATCH -p gpu
#SBATCH -N 1
# Request the number of GPUs per node to be used (if more than 1 GPU per node is required, change 1 into Ngpu, where Ngpu=2,3,4)
#SBATCH --gres=gpu:2
# Request the number of CPU cores. (There are 24 CPU cores and 4 GPUs on each GPU node, so 6 cores for 1 GPU.)
#SBATCH -n 12

conda activate gamellm # Or whatever you called your environment.
python --version
python train.py