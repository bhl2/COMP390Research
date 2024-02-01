import numpy as np
import jac
import pybullet as p

class Goal(object):
    """
    A trivial goal that is always satisfied.
    """

    def __init__(self):
        pass

    def is_satisfied(self, state):
        """
        Determine if the query state satisfies this goal or not.
        """
        return True

class PackGoal1(Goal):
    """
    Try to push all boxes into the corner 

    Represented as a rectangular region
    """

    def __init__(self, x_g = [0.12, 0.28], y_g = [0.12, 0.28], n_boxes = 3):
        super(PackGoal1, self).__init__()
    
        self.x_g, self.y_g, self.n_boxes = x_g, y_g, n_boxes
    def is_satisfied(self, state):
        
        stateVec = state["stateVec"]

        for i in range(self.n_boxes):
            start_idx = -3*(i+1)
            end_idx = start_idx+2
            pos = stateVec[start_idx:end_idx]
            x_pos, y_pos = pos[0], pos[1]
            
            # does not account for cube size
            x_bound = x_pos > self.x_g[0] and x_pos < self.x_g[1]
            y_bound = y_pos > self.y_g[0] and y_pos < self.y_g[1]

            if not (x_bound and y_bound):
                return False
        
        return True



class RelocateGoal(Goal):
    """
    The goal for relocating tasks.
    (i.e., pushing the target object into a circular goal region.)
    """

    def __init__(self, x_g=0.2, y_g=-0.2, r_g=0.1):
        """
        args: x_g: The x-coordinate of the center of the goal region.
              y_g: The y-coordinate of the center of the goal region.
              r_g: The radius of the goal region.
        """
        super(RelocateGoal, self).__init__()
        self.x_g, self.y_g, self.r_g = x_g, y_g, r_g

    def is_satisfied(self, state):
        """
        Check if the state satisfies the RelocateGoal or not.
        args: state: The state to check.
                     Type: dict, {"stateID", int, "stateVec", numpy.ndarray}
        """
        stateVec = state["stateVec"]
        x_tgt, y_tgt = stateVec[7], stateVec[8] # position of the target object
        if np.linalg.norm([x_tgt - self.x_g, y_tgt - self.y_g]) < self.r_g:
            return True
        else:
            return False


class GraspGoal(Goal):
    """
    The goal for grasping tasks.
    (i.e., approaching the end-effector to a pose that can grasp the target object.)
    """

    def __init__(self):
        super(GraspGoal, self).__init__()
        self.jac_solver = jac.JacSolver() # the jacobian solver

    def is_satisfied(self, state):
        """
        Check if the state satisfies the GraspGoal or not.
        args: state: The state to check.
                     Type: dict, {"stateID", int, "stateVec", numpy.ndarray}
        returns: True or False.
        """
        ########## TODO ##########
        joint_angles = state["stateVec"][0:7]
        ee_pose, ee_quat = self.jac_solver.forward_kinematics(joint_angles)
        ee_euler = p.getEulerFromQuaternion(ee_quat)
        ee_theta = ee_euler[2] # only care about rotation about z axis
        ee_x, ee_y = ee_pose[0], ee_pose[1] # get x,y in world frame
        ee_x = ee_x - 0.4 # translation from base
        ee_y = ee_y - 0.2 # translation from base
        cube_pose = state["stateVec"][7:10] # x, y, theta in world frame
        c_x, c_y, c_theta = cube_pose[0], cube_pose[1], cube_pose[2]
        x_dist = (ee_x- c_x)**2
        y_dist = (ee_y - c_y)**2
        dist = np.sqrt((x_dist+y_dist))
        # if dist < np.sqrt(5)/100:
        #     return True
        gamma1 = ee_theta-c_theta # difference angle
        # all rotations of cube that dont actually rotate it, all are equivalent to ee_theta - c_theta - (amount)
        gamma2 = (gamma1 - np.pi/2) % 2*np.pi # rotate it to the right 90 deg
        gamma3 = (gamma1 + np.pi/2) % 2*np.pi # rotate to left 90 deg
        gamma4 = (gamma1 + np.pi) % 2*np.pi # rotate 180 
        gamma1 = gamma1 % 2*np.pi # just to make sure gamma can qualify < 0.2
        gammas = [gamma1, gamma2, gamma3, gamma4]


        
        for gamma in gammas:
            
            angle = ((np.pi/2) - gamma) % np.pi
            d1 = abs(dist*np.sin(angle))
            d2 = abs(dist*np.cos(angle))
            if d1 < 0.01 and d2 < 0.02 and gamma < 0.2:
                print("returned based on math")
                return True
        return False
        
        ##########################
        