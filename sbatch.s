#!/bin/bash
#
#SBATCH --job-name=classification
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=50GB

. ~/.bashrc
#cd /scratch/$USER/rl/sim2real4real/
module load mpi/openmpi-4.0
source activate rl
pip install --upgrade --force-reinstall torch torchvision
cd /home/ksc487/sim2real4real
bash run.sh

