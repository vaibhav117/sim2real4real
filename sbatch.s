#!/bin/bash
#
#SBATCH --job-name=classification
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=30GB

. ~/.bashrc
#cd /scratch/$USER/rl/sim2real4real/
module load mpi/openmpi-4.0
source activate rl
pip install --upgrade --force-reinstall torch torchvision
cd /home/ksc487/sim2real4real
bash run.sh

