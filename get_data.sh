# scp -r cassio:"/home/ksc487/sim2real4real/loss_plot.png /home/ksc487/sim2real4real/reward_plot.png" ./server_weights/
# scp -r cassio:/home/ksc487/sim2real4real/saved_models/ ./sym_server_weights/ 
scp -r cassio:/home/ksc487/sim2real4real/temp.mp4 ./
# scp -r cassio:/home/ksc487/sim2real4real/curr_bc_model_0.005648277043880046.pt ./dagger_rgb_model.pt

# scp -r cassio:/home/ksc487/sim2real4real/recording* ./recording_xarm/
# scp -r cassio:/home/ksc487/s im2real4real/BATCHES:* ./batches/ 

# scp -r ~/.mujoco/ greene:/home/ksc487/.mujoco/

# t.HYRx6K

# commands

# singularity exec --overlay $SCRATCH/overlay-50G-10M.ext3:rw /scratch/work/public/singularity/mujoco-200.sif /bin/bash
# singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:rw /scratch/work/public/singularity/mujoco-200.sif /bin/bash