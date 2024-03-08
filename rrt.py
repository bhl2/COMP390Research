import numpy as np
import time
import samplers
import utils
import control
import gc
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
        self.latest = None

    def add(self, node):
        """
        Add a new node into the tree.
        """
        self.nodes.append(node)
        self.stateVecs = np.vstack((self.stateVecs, node.state['stateVec']))
        assert len(self.nodes) == self.stateVecs.shape[0]
        self.latest = node

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
    
    def depth(self):
        
        max_depth = 0

        for node in self.nodes:
            curr_depth = node.compute_depth()
            if curr_depth > max_depth:
                max_depth = curr_depth
        #print("Depth was found to be: ", max_depth)
        if (len(self.nodes) > 1) and (max_depth == 0):
            print("Tree depth not working")
        return max_depth
    
    def get_leaves(self):
        leaves = set(self.nodes)

        for node in self.nodes:
            if node.get_parent() in leaves:
                leaves.remove(node.get_parent())
        
        return leaves
    

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
        #self.control = np.zeros(shape=(4,)) # the control asscoiated with this node
        self.control = control.NormalControl(np.zeros(shape=(4, )))
        self.parent = None # the parent node of this node
    
    def compute_depth(self):
        depth = 0
        curr = self.get_parent()
        if curr == None:
            return 0
        while curr.get_parent() != None:
            depth += 1
            curr = curr.get_parent()

        return depth + 1
    def get_control(self):
        return self.control

    def get_parent(self):
        return self.parent

    def set_control(self, control : control.Control):
        self.control = control
        # print(control)

    def set_parent(self, pnode):
        self.parent = pnode
    
    def get_state(self):
        return self.state
    

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
    def __init__(self, pdef, h, p, d_max):
        self.pdef = pdef
        self.tree = Tree(pdef)
        self.state_sampler = samplers.StateSampler(self.pdef) # state sampler
        self.control_sampler = samplers.ControlSampler(self.pdef) # control sampler
        self.h = h
        self.p = p
        self.d_max = d_max
    """
    Gets the motions to achieve this node 
    in other words, get all parents
    """
    def extract_controls(self, node : Node):
        motions = [] # a list of parent nodes
        curr_node = node
        while curr_node != None:
            motions.insert(0, curr_node)
            curr_node = curr_node.get_parent()

        return motions
    
    def execute_tau(self, tau):
        
        for motion in tau:
            print(motion.get_control())
            self.sim.execute(motion.get_control())
        
        print("Executed a tau")

    """
    Expands the current motion tree

    Input: tree, current motion tree

    Output: expanded tree
    """
    def expand_tree(self, tree):
        # Hyperparams
        m = 5

        # sample random state
        q_rand = self.state_sampler.sample()
        q_near = self.tree.nearest(q_rand)

        near_control, near_state = self.control_sampler.sample_to(q_near, q_rand, k=m)
        if not (near_control is None):
            new_node = Node(near_state)
            new_node.set_control(near_control)
            new_node.set_parent(q_near)
            tree.add(new_node)
        # ensure that v_star can be translated to a valid motion

        return tree
    
    """
    Input: tree, current motion tree

    Inputs from class: h, heuristic fn
                       p, progress threshold
                       d_max, tree size limit
    """
    def evaluate_progress(self, tree : Tree):
        q_new = self.tree.latest
        tau = []
        if self.pdef.goal.is_satisfied(q_new.get_state()):
            tau = self.extract_controls(q_new)
        elif (self.h(tree.nodes[0].get_state()) - self.h(q_new.get_state())) > self.p:
            print("Acting based on progress")
            tau = self.extract_controls(q_new)
        elif (tree.depth() == self.d_max):
            print("Max depth reached")
            leaves = list(tree.get_leaves())
            min_h = np.infty
            min_leaf = None
            for leaf in leaves:
                curr_h = self.h(leaf.get_state())
                if curr_h < min_h:
                    print("Found a min leaf")
                    min_h = curr_h
                    min_leaf = leaf
            tau = self.extract_controls(min_leaf)

        
        return tau
    """
    The dhRRT algorithm 
    
    Inputs: time_budget, self-explanatory
    """
    def solve(self, time_budget):
        # Initialize Tree
        start_node = Node(self.pdef.start_state)
        sim = self.pdef.panda_sim
        self.sim = sim
        start_node.set_parent(None)
        self.tree.add(start_node)

        tau = []
        t_s = time.time()
        deep_plan = []
        plan = []
        solved = False
        while ((time.time() - t_s) < time_budget):
            self.tree = self.expand_tree(self.tree)
            #print("Expanded tree")
            tau = self.evaluate_progress(self.tree)
            #print("Tau from latest evalutation: ", tau)
            if (tau != []):
                deep_plan.append(tau)
                #print("Found a tau: ", tau)
                utils.execute_plan(sim, tau)
                utils.draw_frontier(sim)
                # self.execute_tau(tau)
                q_star = sim.save_state()
                if not self.pdef.is_state_valid(q_star):
                    print("Ended on a bad state")
                if self.pdef.get_goal().is_satisfied(q_star):
                    solved = True
                    break
                self.tree = Tree(self.pdef)
                new_start = Node(q_star)
                new_start.set_parent(None)
                new_start.set_control(np.zeros(shape=(4, )))
                self.tree.add(new_start)
                self.pdef.set_start_state(q_star)
                tau = []
                # q_star = None
        if solved:
            # flatten plan
            for arr in deep_plan:
                for element in arr:
                    plan.append(element)
        end = sim.save_state()
        return solved, plan, end
    
class Heuristic_dhRRT(object):
    """
    From Ren 2022 Rearrangment:
    
    Inputs: start_state, the state we start in
            g, the goal region (pdef.goal)
            h, the heuristic function for evaluating closeness to goal
            p, the progress threshold
            d_max, the tree limit  
    Output: Technically none, results in tau (sequence of controls) being executed
    """
    def __init__(self, pdef, h, p, d_max):
        self.pdef = pdef
        self.tree = Tree(pdef)
        self.state_sampler = samplers.StateSampler(self.pdef) # state sampler
        self.control_sampler = samplers.Greedy_ControlSampler(self.pdef) # control sampler
        self.h = h
        self.p = p
        self.d_max = d_max
        self.greedy_count = 0
        self.total = 0
        self.COLORS = [[1, 0, 0], # RED
                [1, 0.5, 0], # LIGHT ORANGE
                [1, 1, 0], # YELLOW 
                [0.75, 1, 0.25], # YELLOW GREEN
                [0, 1, 0], # GREEN
                [0, 1, 1], # BLUE-GREEN
                [0, 0, 1], # BLUE
                [1, 0, 1] # PURPLE
                ]

        self.color_idx = 0

    def get_color(self):
        color = self.COLORS[self.color_idx]
        self.color_idx = (self.color_idx + 1) % len(self.COLORS)
        return color
    """
    Gets the motions to achieve this node 
    in other words, get all parents
    """
    def extract_controls(self, node : Node):
        motions = [] # a list of parent nodes
        curr_node = node
        while curr_node != None:
            motions.insert(0, curr_node)
            curr_node = curr_node.get_parent()

        return motions
    
    def execute_tau(self, tau):
        
        for motion in tau:
            print(motion.get_control())
            self.sim.execute(motion.get_control())
        
        print("Executed a tau")

    """
    Expands the current motion tree

    Input: tree, current motion tree

    Output: expanded tree
    """
    def expand_tree(self, tree):
        # Hyperparams
        m = 5

        # sample random state
        q_rand = self.state_sampler.sample()
        q_near = self.tree.nearest(q_rand)

        near_control, near_state, greedy = self.control_sampler.sample_to(q_near, q_rand, k=m)
        if greedy:
            self.greedy_count += 1
        self.total += 1
        if not (near_control is None):
            new_node = Node(near_state)
            new_node.set_control(near_control)
            new_node.set_parent(q_near)
            tree.add(new_node)
        # ensure that v_star can be translated to a valid motion

        return tree
    
    """
    Input: tree, current motion tree

    Inputs from class: h, heuristic fn
                       p, progress threshold
                       d_max, tree size limit
    """
    def evaluate_progress(self, tree : Tree):
        q_new = self.tree.latest
        tau = []
        if self.pdef.goal.is_satisfied(q_new.get_state()):
            tau = self.extract_controls(q_new)
        elif (self.h(tree.nodes[0].get_state()) - self.h(q_new.get_state())) > self.p:
            print("Acting based on progress")
            tau = self.extract_controls(q_new)
        elif (tree.depth() == self.d_max):
            print("Max depth reached")
            leaves = list(tree.get_leaves())
            min_h = np.infty
            min_leaf = None
            for leaf in leaves:
                curr_h = self.h(leaf.get_state())
                if curr_h < min_h:
                    print("Found a min leaf")
                    min_h = curr_h
                    min_leaf = leaf
            tau = self.extract_controls(min_leaf)

        
        return tau
    """
    The dhRRT algorithm 
    
    Inputs: time_budget, self-explanatory
    """
    def solve(self, time_budget):
        # Initialize Tree
        start_node = Node(self.pdef.start_state)
        sim = self.pdef.panda_sim
        self.sim = sim
        start_node.set_parent(None)
        self.tree.add(start_node)
        frontier_color = [1, 0, 0]
        exec_num = 0
        draw_freq = 3
        tau = []
        t_s = time.time()
        deep_plan = []
        plan = []
        solved = False
        while ((time.time() - t_s) < time_budget):
            self.tree = self.expand_tree(self.tree)
            #print("Expanded tree")
            tau = self.evaluate_progress(self.tree)
            #print("Tau from latest evalutation: ", tau)
            if (tau != []):
                deep_plan.append(tau)
                print("Found motion")
                utils.execute_plan(sim, tau, sleep_time=0.1)
                print("Done with execution")
                if exec_num % draw_freq:
                    utils.draw_convex_frontier(sim, c=self.get_color())
                    print("Controls chosen from samples :", self.total)
                    print("Number of which are greedy :", self.greedy_count)
                    print("Done with drawing")
                frontier_color[1] += 0.2
                # self.execute_tau(tau)
                q_star = sim.save_state()
                if self.pdef.get_goal().is_satisfied(q_star):
                    solved = True
                    break
                self.tree = Tree(self.pdef)
                new_start = Node(q_star)
                new_start.set_parent(None)
                # new_start.set_control(np.zeros(shape=(4, )))
                self.tree.add(new_start)
                self.pdef.set_start_state(q_star)
                print("Tree and pdef reset")
                gc.collect()
                tau = []
                # q_star = None
        if solved:
            # flatten plan
            for arr in deep_plan:
                for element in arr:
                    plan.append(element)
        end = sim.save_state()
        return solved, plan, end
    