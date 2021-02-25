# Image Based RL with Asym Actor Critic
This is a pytorch implementation of DDPG+[Hindsight Experience Replay](https://arxiv.org/abs/1707.01495) and [Asymmetric Actor Critic](https://arxiv.org/abs/1710.06542)


## Requirements
- python=3.6
- openai-gym (mujoco200 is supported)
- mujoco-py latest version
- pytorch latest version
- mpi4py latest version

## Instruction to run the code
The run2.sh, run.sh and run3.sh files consist of code that run the various models.

### Play Demo
```bash
python demo.py --env-name=<environment name>
```
### Download the Pre-trained Model
Please download them from the [Google Driver](https://drive.google.com/open?id=1dNzIpIcL4x1im8dJcUyNO30m_lhzO9K4), then put the `saved_models` under the current folder.

## Results
### Training Performance
It was plotted by using 5 different seeds, the solid line is the median value. 
![Training_Curve](figures/results.png)
### Demo:
**Tips**: when you watch the demo, you can press **TAB** to switch the camera in the mujoco.  

FetchReach-v1| FetchPush-v1
-----------------------|-----------------------|
![](figures/reach.gif)| ![](figures/push.gif)

FetchPickAndPlace-v1| FetchSlide-v1
-----------------------|-----------------------|
![](figures/pick.gif)| ![](figures/slide.gif)
