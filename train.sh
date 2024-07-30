#!/bin/bash

#SBATCH --job-name=palette
#SBATCH --ntasks=1
#SBATCH --partition=gpuq
#SBATCH --gpus-per-task=1

srun --ntasks=1 --nodes=1 python run.py -c /home/prrgpp000/Palette-Image-to-Image-Diffusion-Models/config/reconstructions/reconstruction2.json