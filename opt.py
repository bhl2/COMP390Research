import numpy as np
import copy
import sim
import goal
import rrt
import utils
import jac
import pdef
########## TODO ##########
'''
Skeleton, does nothing.
'''
'''
A helper method to compute cost 

Returns total distance traveled of motion
'''
def Cost(motions):
    cost = 0
    for motion in motions:
        # control = [x_dot, y_dot, theta_dot, duration]
        ctrl = motion.get_control()
        x_dot, y_dot, d = ctrl[0], ctrl[1], ctrl[3]
        cost += (x_dot + y_dot) * d
    return cost
'''
Implementation of Physics Based Stochastic Traj Optimization

Returns: A new list of motions to execute that should be smoother
'''
def OptimizeTraj(pdef: pdef.ProblemDefinition, motions, max_iter):
    # params
    K = 5
    # set up the simulation
    pgui = utils.setup_bullet_client(p.GUI)
    panda_sim = sim.PandaSim(pgui)
    jac_solver = jac.JacSolver()
    init_state = pdef.get_start_state()
    states = []
    # a control is [x_dot, y_dot, theta_dot, duration]
    # get initial state sequence
    for motion in motions:
        ctrl = motion.get_control()
        panda_sim.execute(ctrl)
        states.append(panda_sim.save_state)
    c_thresh = Cost(motions)*0.9
    num_iterations = 0
    while num_iterations < max_iter and Cost(motions) > c_thresh:

        print()
    return None

##########################
