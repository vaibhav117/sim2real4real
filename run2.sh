export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ksc487/.mujoco/mujoco200/bin
#/misc/kcgscratch1/karan_exp/miniconda3/envs/dmc/bin/python -u train.py --cuda --plottrain --randomize --env-name='FetchReach-v1' --mode='reach' | tee fetch_pick_and_place.log
mpiexec --oversubscribe --use-hwthread-cpus -n 4 /misc/kcgscratch1/karan_exp/miniconda3/envs/dmc/bin/python -u train.py --cuda --depth --record --env-name='FetchPickAndPlace-v1' --mode='reach' | tee fetch_pick_and_place.log 
