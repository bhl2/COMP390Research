import numpy as np
import time
import samplers
import utils


class Tree(object):
    """
    The tree class for Kinodynamic RRT.
    """

    def __init__(self, pdef):
        """
        args: pdef: The problem definition instance.
                    Type: pdef.ProblemDefinition
        """
        self.pdef = pdef
        self.dim_state = self.pdef.get_state_dimension()
        self.nodes = []
        self.stateVecs = np.empty(shape=(0, self.dim_state))

    def add(self, node):
        """
        Add a new node into the tree.
        """
        self.nodes.append(node)
        self.stateVecs = np.vstack((self.stateVecs, node.state['stateVec']))
        assert len(self.nodes) == self.stateVecs.shape[0]

    def nearest(self, rstateVec):
        """
        Find the node in the tree whose state vector is nearest to "rstateVec".
        """
        dists = self.pdef.distance_func(rstateVec, self.stateVecs)
        nnode_id = np.argmin(dists)
        return self.nodes[nnode_id]

    def size(self):
        """
        Query the size (number of nodes) of the tree. 
        """
        return len(self.nodes)
    

class Node(object):
    """
    The node class for Tree.
    """

    def __init__(self, state):
        """
        args: state: The state associated with this node.
                     Type: dict, {"stateID": int, 
                                  "stateVec": numpy.ndarray}
        """
        self.state = state
        self.control = None # the control asscoiated with this node
        self.parent = None # the parent node of this node

    def get_control(self):
        return self.control

    def get_parent(self):
        return self.parent

    def set_control(self, control):
        self.control = control

    def set_parent(self, pnode):
        self.parent = pnode
    

class KinodynamicRRT(object):
    """
    The Kinodynamic Rapidly-exploring Random Tree (RRT) motion planner.
    """

    def __init__(self, pdef):
        """
        args: pdef: The problem definition instance for the problem to be solved.
                    Type: pdef.ProblemDefinition
        """
        self.pdef = pdef
        self.tree = Tree(pdef)
        self.state_sampler = samplers.StateSampler(self.pdef) # state sampler
        self.control_sampler = samplers.ControlSampler(self.pdef) # control sampler

    def solve(self, time_budget):
        """
        The main algorithm of Kinodynamic RRT.
        args:  time_budget: The planning time budget (in seconds).
        returns: is_solved: True or False.
                      plan: The motion plan found by the planner,
                            represented by a sequence of tree nodes.
                            Type: a list of rrt.Node
        """
        ########## TODO ##########
        if self.tree.size() == 0:
            # add the start
            print("Added start node")
            start_node = Node(self.pdef.start_state)
            start_node.set_parent(None)
            start_node.set_parent(None)
            self.tree.add(start_node)

        k = 10 # hyperparameter for algo
        offset = time.time()
        start = time.time()
        solved = False
        plan = None
        goal = self.pdef.get_goal()
        #print(start, time_budget)
        while time.time()-offset < time_budget:
            #print("running")
            x_samp = self.state_sampler.sample()
            x_nearest = self.tree.nearest(x_samp)
            # x_i_arr = np.zeros(shape=(k, ))
            u_istar, x_istar = self.control_sampler.sample_to(x_nearest, x_samp, k=k) # best control
            # add x_istar as a child of x_nearest with edge weight = u_istar
            if u_istar is None:
                continue
            new_node = Node(x_istar)
            new_node.set_control(u_istar)
            new_node.set_parent(x_nearest)
            self.tree.add(new_node)
            if self.pdef.goal.is_satisfied(x_istar):
                print("Found goal!")
                # find controls
                motions = [] # a list of parent nodes
                curr_node = new_node
                while curr_node != None:
                    motions.insert(0, curr_node)
                    curr_node = curr_node.get_parent()
                solved = True
                plan = motions
                break
        
                   


        ##########################

        return solved, plan

class dhRRT(object):
    """
    From Ren 2022 Rearrangment:
    
    Inputs: start_state, the state we start in
            g, the goal region (pdef.goal)
            h, the heuristic function for evaluating closeness to goal
            p, the progress threshold
            d_max, the tree limit  
    Output: Technically none, results in tau (sequence of controls) being executed
    """
    def __init__(self, pdef):
        self.pdef = pdef
        self.tree = Tree(pdef)
        self.state_sampler = samplers.StateSampler(self.pdef) # state sampler
        self.control_sampler = samplers.ControlSampler(self.pdef) # control sampler

    """
    Expands the current motion tree

    Input: tree, current motion tree

    Output: expanded tree
    """
    def expand_tree(self, tree):
        # Hyperparams
        m = 10

        # sample random state
        q_rand = self.state_sampler.sample()
        q_near = self.tree.nearest(q_rand)

        q_new, v_star = self.control_sampler.sample_to(q_near, q_rand, k=m)

        return
    
    """
    Input: tree, current motion tree

    Inputs from class: h, heuristic fn
                       p, progress threshold
                       d_max, tree size limit
    """
    def evaluate_progress(self, tree):

        return
    """
    The dhRRT algorithm 
    
    Inputs: time_budget, self-explanatory
    """
    def solve(self, time_budget):
        # Initialize Tree
        start_node = Node(self.pdef.start_state)
        start_node.set_parent(None)
        self.tree.add(start_node)

        tau = []
        t_s = time.time()
        while ((time.time() - t_s) > time_budget):
            self.tree = self.expand_tree(self.tree)
        return
    