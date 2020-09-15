#!/bin/bash

module load examl/openmpi/intel/3.0.22
singularity exec --nv \
        /beegfs/work/public/singularity/mujoco-200.sif \
        bash -c "
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/home/ksc487/.mujoco/mujoco200/bin
#export LD_LIBRARY_PATH=/share/apps/openmpi/2.0.1/intel/lib:/share/apps/intel/17.0.1/mkl/lib/intel64:/share/apps/intel/17.0.1/lib/intel64:/share/apps/centos/7/usr/lib64:/opt/slurm/lib64:/home/ksc487/.mujoco/mujoco200/bin:/share/apps/centos/7/usr/lib64:/opt/slurm/lib64:/home/ksc487/.mujoco/mujoco200/bin

export PATH=/share/apps/examl/3.0.22/openmpi/intel/bin:$PATH
#export PATH=/share/apps/examl/3.0.22/openmpi/intel/bin:/share/apps/openmpi/2.0.1/intel/bin:/share/apps/intel/17.0.1/bin:/scratch/ksc487/miniconda3/envs/mujoco_test/bin:/scratch/ksc487/miniconda3/condabin:/usr/lib64/qt-3.3/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/opt/ibutils/bin:/share/apps/local/bin:/share/apps/centos/7/bin:/share/apps/singularity/bin:/opt/slurm/bin:/opt/dell/srvadmin/bin:/home/ksc487/.local/bin:/home/ksc487/bin:/share/apps/local/bin:/share/apps/centos/7/bin:/share/apps/singularity/bin

source $HOME/.bashrc
module load examl/openmpi/intel/3.0.22
conda activate mujoco_test
nvidia-smi
mpirun -np 1 --use-hwthread-cpus --oversubscribe python -u test_scaling.py
"
