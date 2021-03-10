class Env:

    def __init__(self, target_distance, goal_boundary):
        """
        Initialises some properties of the env, take in an XML file, load the env,
        setup what is the maximum distance between the end effector and the goal to qualify as a successful "reach".

        Setup the 3D boundary limits of where the block can be sampled in the environment.
        
        """
        pass

    
    def reset():
        """
        Resets the environment to a state where the goal is sampled randomly,
        and the pose of the robot is also random.
        """
        pass


    def step(action: List[float]):
        """
        Steps the simulation so that end effector moves in the direction specified in the actions.
        The action will specify how much the end effector should move in meters in x,y,z directions

        Returns the new state of the env {joint positions, end effector position, goal position, goal pose, other env characteristics}
        Returns a reward (return 1 if the end effector is within some distance of the goal else return 0)
        """
        pass

    def get_state():
        """
        Get position of end effector, position of goal, velocity, torque information from the robot.
        Badically some physics stuff (joint angles) that would potentially be useful for the agent to know
        about the state of the simulation.

        In the future, with rendering, we can get depth/rgb observations as well.
        """
        pass

    def set_goal_state(position):
        """
        set position of goal (the red block) in the enviornment.

        """
        pass

    def set_robot_location(position):
        """
        move the robot in such a way that the end effector is at the given position.
        """
        pass

    def set_value(key, value):
        """

        THis function changes the environment by changing different properties.
        Help needed here
        This function can help us randomize certain aspects of the physics simulation of the robot and the world.
        Contact dynamics of the block, the table surface, the friction in the joints, the "rigidity" of the gripper. The size of ceratin things like the size of the block.
        It can also help us move certain objects in the env by specify the new position of the object.
        The key represents the type of property, and the value represents the randomized value.
        Instead of a key, value argument, possibly we can send a dictionary too of multiple keys/values.
        """
        pass

    def get_value(key):
        """
        Get position of a certain item (joint, end effector, goal block, robot base, table etc)
        """

    def close():
        """
        End everything in a graceful way (closing simulation etc)
        """