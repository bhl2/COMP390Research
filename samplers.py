import numpy as np
import control
import pybullet as bullet
import utils
import math
import scipy

def compute_centroid(list, facets):
    div = len(facets)
    avg = np.zeros(shape=(2, ))
    for fac in facets:
        avg += list[fac] / div
    
    return avg
class StateSampler(object):
    """
    The state sampler of Kinodynamic RRT.
    """

    def __init__(self, pdef):
        """
        args: pdef: The problem definition instance for the problem to be solved.
                    Type: pdef.ProblemDefinition
        """
        self.pdef = pdef
        self.dim = self.pdef.get_state_dimension() # dimensionality of state space
        self.low = self.pdef.bounds_state.low # the lower bounds of the state space
        self.high = self.pdef.bounds_state.high # the upper bounds of the state space

    def sample(self):
        """
        Uniformly random sample a state vector.
        returns: stateVec: The sampled state vector.
                           Type: numpy.ndarray of shape (self.dim,)
        """
        stateVec = np.random.uniform(self.low, self.high, self.dim)
        return stateVec


class ControlSampler(object):
    """
    The control sampler for Kinodynamic RRT.
    """

    def __init__(self, pdef):
        """
        args: pdef: The problem definition instance for the problem to be solved.
                    Type: pdef.ProblemDefinition
        """
        self.pdef = pdef
        self.dim = self.pdef.get_control_dimension() # dimensionality of the control space
        self.low = self.pdef.bounds_ctrl.low # the lower bounds of the control space
        self.high = self.pdef.bounds_ctrl.high # the upper bounds of the control space

    def sample_to(self, nnode, rstateVec, k):
        """
        Sample k candidates controls from nnode and return the control 
        whose outcome state is nearest to rstateVec.
        args:     nnode: The node from where the controls are sampled.
                         Type: rrt.Node
              rstateVec: The reference state vector towards which the controls are sampled.
                         Type: numpy.ndarray
                      k: The number of candidates controls.
                         Type: int
        returns:  bctrl: The best control which leads to a state nearest to rstateVec.
                         Type: numpy.ndarray of shape (self.dim,)
                               or None if all the k candidate controls lead to an invalid state
                 ostate: The outcome state of the best control.
                         Type: dict, {"stateID": int, "stateVec": numpy.ndarray}
                               or None if bctrl is None
        """
        nstate = nnode.state
        assert k >= 1
        controls = []
        pstates = []
        dists = []
        i = 0
        while (i < k):
            ctrl = np.random.uniform(self.low, self.high, self.dim)
            pstate, valid = self.pdef.propagate(nstate, ctrl)
            if valid and self.pdef.is_state_valid(pstate):
                dist = self.pdef.distance_func(pstate["stateVec"], rstateVec)
                controls.append(ctrl)
                pstates.append(pstate)
                dists.append(dist)
            i += 1

        bctrl, ostate = None, None 
        if len(dists) > 0:
            best_i = np.argmin(dists)
            bctrl, ostate = controls[best_i], pstates[best_i]
        return bctrl, ostate
    
    def set_max_duration(self, d):
        self.high[3] = d
    
    def set_min_duration(self, d):
        self.low[3] = d


class OOP_ControlSampler(object):
    def __init__(self, pdef, epsilon=0.2):
        self.pdef = pdef
        self.eps = epsilon
        self.dim = self.pdef.get_control_dimension() # dimensionality of the control space
        self.low = self.pdef.bounds_ctrl.low # the lower bounds of the control space
        self.high = self.pdef.bounds_ctrl.high # the upper bounds of the control space
        self.n_boxes = pdef.get_goal().get_n_boxes()
    
    def sample_to(self, nnode, rstateVec, k):
        """
        Sample k candidates controls from nnode and return the control 
        whose outcome state is nearest to rstateVec.
        args:     nnode: The node from where the controls are sampled.
                         Type: rrt.Node
              rstateVec: The reference state vector towards which the controls are sampled.
                         Type: numpy.ndarray
                      k: The number of candidates controls.
                         Type: int
        returns:  bctrl: The best control which leads to a state nearest to rstateVec.
                         Type: numpy.ndarray of shape (self.dim,)
                               or None if all the k candidate controls lead to an invalid state
                 ostate: The outcome state of the best control.
                         Type: dict, {"stateID": int, "stateVec": numpy.ndarray}
                               or None if bctrl is None
        """
        nstate = nnode.state
        radius = 0.05
        assert k >= 1
        controls = []
        pstates = []
        dists = []
        i = 0
        if np.random.rand() < (self.eps):
                # sample a lift movement
            state = nstate
            stateVec = state["stateVec"]
            box_pos_lst = np.zeros(shape=(self.n_boxes, 2))
            for i in range(self.n_boxes):
                start_idx = -3*(i+1)
                end_idx = start_idx+2
                pos = stateVec[start_idx:end_idx]
                x_pos, y_pos = pos[0], pos[1]
                box_pos_lst[i, 0] = x_pos
                box_pos_lst[i, 1] = y_pos
            
            box_idx = np.random.randint(0, self.n_boxes)
            box_loc = box_pos_lst[box_idx]
            angle = np.random.random() * np.pi * 2
            final_x, final_y = box_loc[0] + radius * np.cos(angle), box_loc[1] + radius * np.sin(angle)
            ctrl1 = control.LiftControl(final_x, final_y)
        else:
            ctrl1 = control.NormalControl(np.zeros(shape=(4,)))
        
        nstate2, valid = self.pdef.propagate(nstate, ctrl1)
        if not valid:
            return None, None
        while (i < k):
            ctrl = control.NormalControl(np.random.uniform(self.low, self.high, self.dim))
            pstate, valid = self.pdef.propagate(nstate2, ctrl)
            if valid and self.pdef.is_state_valid(pstate):
                dist = self.pdef.distance_func(pstate["stateVec"], rstateVec)
                controls.append(ctrl)
                pstates.append(pstate)
                dists.append(dist)
            i += 1

        bctrl, ostate = None, None 
        if len(dists) > 0:
            best_i = np.argmin(dists)
            bctrl, ostate = controls[best_i], pstates[best_i]
        bctrl = control.CompoundControl(ctrl1, bctrl)
        return bctrl, ostate

class OOP_ControlSampler(object):
    def __init__(self, pdef, epsilon=0.2):
        self.pdef = pdef
        self.eps = epsilon
        self.dim = self.pdef.get_control_dimension() # dimensionality of the control space
        self.low = self.pdef.bounds_ctrl.low # the lower bounds of the control space
        self.high = self.pdef.bounds_ctrl.high # the upper bounds of the control space
        self.n_boxes = pdef.get_goal().get_n_boxes()
    
    def sample_to(self, nnode, rstateVec, k):
        """
        Sample k candidates controls from nnode and return the control 
        whose outcome state is nearest to rstateVec.
        args:     nnode: The node from where the controls are sampled.
                         Type: rrt.Node
              rstateVec: The reference state vector towards which the controls are sampled.
                         Type: numpy.ndarray
                      k: The number of candidates controls.
                         Type: int
        returns:  bctrl: The best control which leads to a state nearest to rstateVec.
                         Type: numpy.ndarray of shape (self.dim,)
                               or None if all the k candidate controls lead to an invalid state
                 ostate: The outcome state of the best control.
                         Type: dict, {"stateID": int, "stateVec": numpy.ndarray}
                               or None if bctrl is None
        """
        nstate = nnode.state
        radius = 0.05
        assert k >= 1
        controls = []
        pstates = []
        dists = []
        i = 0
        # is this a lift control or no?
        if np.random.rand() < (self.eps):
                # sample a lift movement
            # print("-------BEFORE LIFT ----------")
            state = nstate
            # print(state)
            # print("----------------------")
            stateVec = state["stateVec"]
            box_pos_lst = np.zeros(shape=(self.n_boxes, 2))
            for i in range(self.n_boxes):
                start_idx = -3*(i+1)
                end_idx = start_idx+2
                pos = stateVec[start_idx:end_idx]
                x_pos, y_pos = pos[0], pos[1]
                box_pos_lst[i, 0] = x_pos
                box_pos_lst[i, 1] = y_pos
            
            box_idx = np.random.randint(0, self.n_boxes)
            box_loc = box_pos_lst[box_idx]
            angle = np.random.random() * np.pi * 2
            final_x, final_y = box_loc[0] + radius * np.cos(angle), box_loc[1] + radius * np.sin(angle)
            ctrl1 = control.LiftControl(final_x, final_y)
            return ctrl1
            # print("-------AFTER LIFT ----------")
            # print(pstate)
            # print("----------------------")
        else:
            ctrl1 = control.NormalControl(np.zeros(shape=(4,)))
            
        while (i < k):
            ctrl = control.CompoundControl(ctrl1, control.NormalControl(np.random.uniform(self.low, self.high, self.dim)))
            pstate, valid = self.pdef.propagate(nstate, ctrl)
            if valid and self.pdef.is_state_valid(pstate):
                dist = self.pdef.distance_func(pstate["stateVec"], rstateVec)
                controls.append(ctrl)
                pstates.append(pstate)
                dists.append(dist)
            i += 1

        bctrl, ostate = None, None 
        if len(dists) > 0:
            best_i = np.argmin(dists)
            bctrl, ostate = controls[best_i], pstates[best_i]
        return bctrl, ostate

class Heuristic_ControlSampler(object):
    def __init__(self, pdef, epsilon=0.2):
        self.pdef = pdef
        self.eps = epsilon
        self.dim = self.pdef.get_control_dimension() # dimensionality of the control space
        self.low = self.pdef.bounds_ctrl.low # the lower bounds of the control space
        self.high = self.pdef.bounds_ctrl.high # the upper bounds of the control space
        self.n_boxes = pdef.get_goal().get_n_boxes()
    
    def sample_to(self, nnode, rstateVec, k):
        """
        Sample k candidates controls from nnode and return the control 
        whose outcome state is nearest to rstateVec.
        args:     nnode: The node from where the controls are sampled.
                         Type: rrt.Node
              rstateVec: The reference state vector towards which the controls are sampled.
                         Type: numpy.ndarray
                      k: The number of candidates controls.
                         Type: int
        returns:  bctrl: The best control which leads to a state nearest to rstateVec.
                         Type: numpy.ndarray of shape (self.dim,)
                               or None if all the k candidate controls lead to an invalid state
                 ostate: The outcome state of the best control.
                         Type: dict, {"stateID": int, "stateVec": numpy.ndarray}
                               or None if bctrl is None
        """
        nstate = nnode.state
        radius = 0.05
        assert k >= 1
        controls = []
        pstates = []
        dists = []
        i = 0
        # is this a lift control or no?
        if np.random.rand() < (self.eps):
                # sample a lift movement
            # print("-------BEFORE LIFT ----------")
            state = nstate
            # print(state)
            # print("----------------------")
            stateVec = state["stateVec"]
            box_pos_lst = np.zeros(shape=(self.n_boxes, 2))
            for i in range(self.n_boxes):
                start_idx = -3*(i+1)
                end_idx = start_idx+2
                pos = stateVec[start_idx:end_idx]
                x_pos, y_pos = pos[0], pos[1]
                box_pos_lst[i, 0] = x_pos
                box_pos_lst[i, 1] = y_pos
            
            box_idx = np.random.randint(0, self.n_boxes)
            box_loc = box_pos_lst[box_idx]

            # use distance from box to optimal center
            ideal_pt = self.pdef.get_goal().optim_center
            dist_to_ideal = np.linalg.norm(box_loc-ideal_pt)
            if dist_to_ideal > 0.05:
                self.pdef.bounds_ctrl.set_bounds(0, -1.5*dist_to_ideal, 1.5*dist_to_ideal)
                self.pdef.bounds_ctrl.set_bounds(1, -1.5*dist_to_ideal, 1.5*dist_to_ideal)
            angle = np.random.random() * np.pi * 2
            final_x, final_y = box_loc[0] + radius * np.cos(angle), box_loc[1] + radius * np.sin(angle)
            ctrl1 = control.LiftControl(final_x, final_y)
            
            # print("-------AFTER LIFT ----------")
            # print(pstate)
            # print("----------------------")
        else:
            ctrl1 = control.NormalControl(np.zeros(shape=(4,)))

        nstate2, valid = self.pdef.propagate(nstate, ctrl1)
        if not (valid and self.pdef.is_state_valid(nstate2, state_curr=nstate)):
            return None, None
        while (i < k):

            ctrl = control.CompoundControl(ctrl1, control.NormalControl(np.random.uniform(self.low, self.high, self.dim)))
            pstate, valid = self.pdef.propagate(nstate2, ctrl)
            if valid and self.pdef.is_state_valid(pstate, state_curr = nstate2):
                dist = self.pdef.distance_func(pstate["stateVec"], rstateVec)
                controls.append(ctrl)
                pstates.append(pstate)
                dists.append(dist)
            i += 1

        bctrl, ostate = None, None 
        if len(dists) > 0:
            best_i = np.argmin(dists)
            bctrl, ostate = controls[best_i], pstates[best_i]
        return bctrl, ostate
    

"""
A control sampler where eps % of the time it samples the greediest push possible

"""
class Greedy_ControlSampler(object):
    def __init__(self, pdef, epsilon=0.2):
        self.pdef = pdef
        self.eps = epsilon
        self.dim = self.pdef.get_control_dimension() # dimensionality of the control space
        self.low = self.pdef.bounds_ctrl.low # the lower bounds of the control space
        self.high = self.pdef.bounds_ctrl.high # the upper bounds of the control space
        self.n_boxes = pdef.get_goal().get_n_boxes()
    '''
    Sample a lift
    '''
    def get_lift_action(self, nnode):
        nstate = nnode.state
        stateVec = nstate["stateVec"]
        radius = 0.08
        box_pos_lst = np.zeros(shape=(self.n_boxes, 2))
        for i in range(self.n_boxes):
            start_idx = -3*(i+1)
            end_idx = start_idx+2
            pos = stateVec[start_idx:end_idx]
            x_pos, y_pos = pos[0], pos[1]
            box_pos_lst[i, 0] = x_pos
            box_pos_lst[i, 1] = y_pos
        # pick a box
        box_idx = np.random.randint(0, self.n_boxes)
        box_loc = box_pos_lst[box_idx]
        # pick an angle
        angle = np.random.random() * np.pi * 2
        final_x, final_y = box_loc[0] + radius * np.cos(angle), box_loc[1] + radius * np.sin(angle)
        ctrl1 = control.LiftControl(final_x, final_y)

    """
    
    Use (f(x+h)-f(x)) / h 
    """
    def compute_local_gradient(self, box_idx, box_pos_lst):
        h = 1 ** -10
        ideal_pt = self.pdef.get_goal().optim_center

        grads = []
        # COMPUTE HULL
        hull = scipy.spatial.ConvexHull(box_pos_lst)
        
        # GET FACET IDXs
        facets = hull.vertices
        centroid = compute_centroid(box_pos_lst, facets)
        # FOR EACH FACET, COMPUTE GRADIENT

        # COMPUTE f(x) first tho
        centroid_dist = np.linalg.norm(centroid-ideal_pt)
        # GRADIENT = CHANGE IN DISTANCE FROM CENTROID TO CORNER
        for facetIdx in facets:
            coord = box_pos_lst[facetIdx]
            dist = coord - ideal_pt
            mag = np.linalg.norm(dist)
            norm_dist = dist / mag

            new_coord = coord -(norm_dist * h)
            box_pos_copy = np.copy(box_pos_lst)
            box_pos_copy[facetIdx] = new_coord
            new_hull = scipy.spatial.ConvexHull(box_pos_copy)
            new_facets = new_hull.vertices
            new_centroid = compute_centroid(box_pos_copy, new_facets)
            new_dist = np.linalg.norm(new_centroid - ideal_pt)

            grad = (new_dist - centroid_dist) / h
            grads.append(grad)
        # RETURN GRADIENTS
        return np.array(grads)
    def get_greedy_span_action(self, nnode):
        nstate = nnode.state
        stateVec = nstate["stateVec"]
        box_pos_lst = np.zeros(shape=(self.n_boxes, 2))
        box_weights = np.zeros(shape=(self.n_boxes))
        ideal_pt = self.pdef.get_goal().optim_center
        total_weight = 0
        for i in range(self.n_boxes):
            start_idx = -3*(i+1)
            end_idx = start_idx+2
            pos = stateVec[start_idx:end_idx]
            x_pos, y_pos = pos[0], pos[1]
            box_pos_lst[i, 0] = x_pos
            box_pos_lst[i, 1] = y_pos
            box_weights[i] = (np.linalg.norm(box_pos_lst[i] - ideal_pt))
            total_weight += box_weights[i]
        for j in range(self.n_boxes):
            box_weights[j] = box_weights[j] / total_weight
        
        print("Sum of weights", np.sum(box_weights))
        print("Weights :", box_weights)
       
        
        box_idx = np.random.choice(range(0, self.n_boxes), p=box_weights)
        box_loc = box_pos_lst[box_idx]
        

        # Find behind the box 
        # slope of line with box and corner
        y_delt = (ideal_pt[1] - box_loc[1])
        x_delt = (ideal_pt[0] - box_loc[0]) 

        pos, quat = self.pdef.panda_sim.get_ee_pose()
        euler = bullet.getEulerFromQuaternion(quat)
        yaw = euler[2]
        dot = np.dot(pos[0:2], [0.3, 0.3])
        norm1 = np.linalg.norm(pos[0:2])
        norm2 = np.linalg.norm([0.3, 0.3])
        val = dot / (norm1 * norm2)
        diff_angle = np.arccos(val) # use to turn EEF 
        land_x = box_loc[0] - ideal_pt[0]
        land_y = box_loc[1] - ideal_pt[1]
        
        radius = 0.08 # how far away to land
        land_vec = np.array([((land_x / (land_x + land_y)) * radius), 
                        ((land_y / (land_x + land_y)) * radius)])
        weight1 = np.random.rand()
        weight2 = 1 - weight1
        angle = ((np.pi/4) * weight1) + ((-np.pi/4) * weight2)

        rot_mat = utils.make_rot_mat(angle)
        # print("Rot mat :", rot_mat)

        final_land = np.matmul(rot_mat, land_vec)

        

        acc_land_x = box_loc[0] - final_land[0]
        acc_land_y = box_loc[1] - final_land[1]

        dir = np.array([x_delt, y_delt])
        final_dir = np.matmul(rot_mat, dir)


        c1 = control.LiftControl(acc_land_x, acc_land_y)
        c2 = control.NormalControl([0, 0, diff_angle, 1])
        c15 = control.CompoundControl(c1, c2)
        c3 = control.NormalControl([0.7 * final_dir[0], 0.7 * final_dir[1], 0, 0.75])
        return control.CompoundControl(c15, c3)
    def get_greedy_action(self, nnode):
        nstate = nnode.state
        stateVec = nstate["stateVec"]
        box_pos_lst = np.zeros(shape=(self.n_boxes, 2))
        for i in range(self.n_boxes):
            start_idx = -3*(i+1)
            end_idx = start_idx+2
            pos = stateVec[start_idx:end_idx]
            x_pos, y_pos = pos[0], pos[1]
            box_pos_lst[i, 0] = x_pos
            box_pos_lst[i, 1] = y_pos
        weights = np.zeros(shape=(self.n_boxes, ))
        
        box_idx = np.random.randint(0, self.n_boxes)
        box_loc = box_pos_lst[box_idx]
        ideal_pt = self.pdef.get_goal().optim_center

        # Find behind the box 
        # slope of line with box and corner
        y_delt = (ideal_pt[1] - box_loc[1])
        x_delt = (ideal_pt[0] - box_loc[0]) 

        pos, quat = self.pdef.panda_sim.get_ee_pose()
        euler = bullet.getEulerFromQuaternion(quat)
        yaw = euler[2]
        dot = np.dot(pos[0:2], [0.3, 0.3])
        norm1 = np.linalg.norm(pos[0:2])
        norm2 = np.linalg.norm([0.3, 0.3])
        val = dot / (norm1 * norm2)
        diff_angle = np.arccos(val) # use to turn EEF 
        land_x = box_loc[0] - ideal_pt[0]
        land_y = box_loc[1] - ideal_pt[1]
        acc_land_x = box_loc[0] - ((land_x / (land_x + land_y)) * 0.08)
        acc_land_y = box_loc[1] - ((land_y / (land_x + land_y)) * 0.08)

        c1 = control.LiftControl(acc_land_x, acc_land_y)
        c2 = control.NormalControl([0, 0, diff_angle, 1])
        c15 = control.CompoundControl(c1, c2)
        c3 = control.NormalControl([0.7 * x_delt, 0.7 * y_delt, 0, 1])
        return control.CompoundControl(c15, c3)
        pass
    
    def sample_to(self, nnode, rstateVec, k):
        """
        Sample k candidates controls from nnode and return the control 
        whose outcome state is nearest to rstateVec.
        args:     nnode: The node from where the controls are sampled.
                         Type: rrt.Node
              rstateVec: The reference state vector towards which the controls are sampled.
                         Type: numpy.ndarray
                      k: The number of candidates controls.
                         Type: int
        returns:  bctrl: The best control which leads to a state nearest to rstateVec.
                         Type: numpy.ndarray of shape (self.dim,)
                               or None if all the k candidate controls lead to an invalid state
                 ostate: The outcome state of the best control.
                         Type: dict, {"stateID": int, "stateVec": numpy.ndarray}
                               or None if bctrl is None
        """
        nstate = nnode.state
        radius = 0.05
        assert k >= 1
        controls = []
        pstates = []
        dists = []
        i = 0
        # is this a lift control or no?
        # if np.random.rand() < (self.eps):
        #     ctrl1 = self.get_greedy_action(nnode)
        #     ostate, valid = self.pdef.propagate(nstate, ctrl1)
        #     if valid:
        #         return ctrl1, ostate
        #     else: 
        #         ctrl1 = control.NormalControl(np.zeros(shape=(4,)))
        # else:
        #     ctrl1 = control.NormalControl(np.zeros(shape=(4,)))

        # nstate2, valid = self.pdef.propagate(nstate, ctrl1)
        # if not (valid and self.pdef.is_state_valid(nstate2, state_curr=nstate)):
        #     return None, None
        greedys = []
        greedy = False
        while (i < k):
            if np.random.rand() < (self.eps):
                ctrl = self.get_greedy_span_action(nnode)
                greedys.append(ctrl)
            else:
                ctrl = control.NormalControl(np.random.uniform(self.low, self.high, self.dim))
            pstate, valid = self.pdef.propagate(nstate, ctrl)
            if valid and self.pdef.is_state_valid(pstate):
                dist = self.pdef.distance_func(pstate["stateVec"], rstateVec)
                controls.append(ctrl)
                pstates.append(pstate)
                dists.append(dist)
            i += 1

        bctrl, ostate = None, None 
        if len(dists) > 0:
            best_i = np.argmin(dists)
            bctrl, ostate = controls[best_i], pstates[best_i]
        if bctrl in greedys:
            greedy = True
        return bctrl, ostate, greedy