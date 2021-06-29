from mujoco_py import MjRenderContextOffscreen

def randomize_textures(modder, env):
    for name in env.sim.model.geom_names:
        print(name)
        if name != 'object0': 
            modder.rand_all(name)

def load_viewer_to_env(env):
    viewer = MjRenderContextOffscreen(env.sim, device_id=-1)
    viewer.cam.distance = 1.2 # this will be randomized baby: domain randomization FTW
    viewer.cam.azimuth = 180 # this will be randomized baby: domain Randomization FTW
    viewer.cam.elevation = -25 # this will be randomized baby: domain Randomization FTW
    viewer.cam.lookat[2] = 0.5 # IMPORTANT FOR ALIGNMENT IN SIM2REAL !!
    env.env._viewers["rgb_array"] = viewer
    return env