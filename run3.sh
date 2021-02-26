export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ksc487/.mujoco/mujoco200/bin
mpiexec --oversubscribe --use-hwthread-cpus -n 4 /misc/kcgscratch1/karan_exp/miniconda3/envs/dmc/bin/python -u distill.py --cuda
