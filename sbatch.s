#!/bin/bash
#
#SBATCH --job-name=classification
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=256GB

. ~/.bashrc
cd /scratch/$USER/rl/sim2real4real/
bash test_mujoco.sh
