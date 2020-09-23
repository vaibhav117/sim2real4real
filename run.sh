export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ksc487/.mujoco/mujoco200/bin
mpiexec --oversubscribe --use-hwthread-cpus -n 1 /misc/kcgscratch1/karan_exp/miniconda3/envs/rl/bin/python -u train.py --cuda --env-name='FetchPush-v1'| tee push.log 
