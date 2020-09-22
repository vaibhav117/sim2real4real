#!/bin/bash
#
#SBATCH --job-name=classification
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:v100:4
#SBATCH --mem=30GB

. ~/.bashrc
#cd /scratch/$USER/rl/sim2real4real/
module load mpi/openmpi-4.0
cd /home/ksc487/sim2real4real
bash run.sh
