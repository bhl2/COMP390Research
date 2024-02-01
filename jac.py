import copy
import numpy as np
import pybullet as p
import pybullet_data as pd
import pybullet_utils.bullet_client as bc
import scipy
import PyKDL
class JacSolver(object):
    """
    The Jacobian solver for the 7-DoF Franka Panda robot.
    """

    def __init__(self):
        self.bullet_client = bc.BulletClient(connection_mode=p.DIRECT)
        self.bullet_client.setAdditionalSearchPath(pd.getDataPath())
        self.panda = self.bullet_client.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

    def forward_kinematics(self, joint_values):
        """
        Calculate the Forward Kinematics of the robot given joint angle values.
        args: joint_values: The joint angle values of the query configuration.
                            Type: numpy.ndarray of shape (7,)
        returns:       pos: The position of the end-effector.
                            Type: numpy.ndarray [x, y, z]
                      quat: The orientation of the end-effector represented by quaternion.
                            Type: numpy.ndarray [x, y, z, w]
        """
        for j in range(7):
            self.bullet_client.resetJointState(self.panda, j, joint_values[j])
        ee_state = self.bullet_client.getLinkState(self.panda, linkIndex=11)
        pos, quat = np.array(ee_state[4]), np.array(ee_state[5])
        return pos, quat

    def get_jacobian_matrix(self, joint_values):
        """
        Numerically calculate the Jacobian matrix based on joint angles.
        args: joint_values: The joint angles of the query configuration.
                            Type: numpy.ndarray of shape (7,)
        returns:         J: The calculated Jacobian matrix.
                            Type: numpy.ndarray of shape (6, 7)
        """
        ########## TODO ##########
        J = np.zeros(shape=(6, 7))
        finite_diff = 0.01

        pos1, quat1 = self.forward_kinematics(joint_values)
        for col in range(7):
            angle_copy = copy.deepcopy(joint_values)
            angle_copy[col] = angle_copy[col] + finite_diff
            pos2, quat2 = self.forward_kinematics(angle_copy)

            angle_diff = p.getDifferenceQuaternion(quat1, quat2)
            axis, angle = p.getAxisAngleFromQuaternion(angle_diff)

            diff = np.subtract(pos2, pos1)
            column = np.zeros(shape=(6, ))
            for i in range(3):
                column[i+3] = (axis[i]*angle)/finite_diff
                column[i] = diff[i]/finite_diff
            for i in range(6):
                J[i, col] = column[i]
           
        
        ##########################
        return J