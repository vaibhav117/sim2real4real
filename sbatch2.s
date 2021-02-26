#!/bin/bash
#
#SBATCH --job-name=classification
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=40:00:00
#SBATCH --constraint=pascal|turing|volta
#SBATCH --gres=gpu:4
#SBATCH --mem=50GB

. ~/.bashrc
#cd /scratch/$USER/rl/sim2real4real/
module load mpi/openmpi-4.0
#module load cuda-9.0
nvidia-smi
source activate dmc
#pip install --upgrade --force-reinstall --no-cache-dir torch torchvision
cd /home/ksc487/sim2real4real
bash run2.sh

