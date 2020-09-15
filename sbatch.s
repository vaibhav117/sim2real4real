#!/bin/bash
#
#SBATCH --job-name=classification
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:p40:1
#SBATCH --mem=10GB

. ~/.bashrc
cd /scratch/$USER/rl/sim2real4real/
bash run_mujoco.sh
