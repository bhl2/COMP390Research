import math
import numpy as np
import pybullet as p
import pybullet_data as pd
import pybullet_utils.bullet_client as bc
import sim
from goal import PackGoal1, PackGoal2, find_packing
'''
Input: n, the number of boxes to generate
       x_bound, [lower, higher] the x bounds to place boxes 
       y_bound, [lower, higher] the y bounds to place boxes

Output: An ndarray of shape (n, 2) of x,y positions for n boxes to be made
'''
def pos_n_boxes(n, x_bound, y_bound):
  arr = np.zeros(shape=(n, 2))

  for i in range(n):
    x = (x_bound[1] - x_bound[0]) * np.random.random_sample() + x_bound[0]
    y = (y_bound[1] - y_bound[0]) * np.random.random_sample() + y_bound[0]
    arr[i] = [x, y]

  return arr
def setup_bullet_client(connection_mode):
  bullet_client = bc.BulletClient(connection_mode=connection_mode)
  bullet_client.resetSimulation()
  bullet_client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
  bullet_client.setAdditionalSearchPath(pd.getDataPath())
  bullet_client.setTimeStep(sim.SimTimeStep)
  bullet_client.setPhysicsEngineParameter(deterministicOverlappingPairs=1) # determinism guaranteed, important
  return bullet_client

def setup_env(panda_sim):
  # set up the environment
  panda_sim.add_object([0.02, 0.02, 0.02], [1.0, 1.0, 0.0, 1.0], [0, 0])
  panda_sim.add_object([0.02, 0.02, 0.02], [0.0, 0.0, 1.0, 1.0], [-0.05, -0.05])
  panda_sim.add_object([0.02, 0.02, 0.02], [0.0, 0.0, 1.0, 1.0], [0, -0.05])
  panda_sim.add_object([0.02, 0.02, 0.02], [0.0, 0.0, 1.0, 1.0], [0.05, -0.05])
  panda_sim.add_object([0.02, 0.02, 0.02], [0.0, 0.0, 1.0, 1.0], [-0.05, 0])
  panda_sim.add_object([0.02, 0.02, 0.02], [0.0, 0.0, 1.0, 1.0], [0.05, 0])
  panda_sim.add_object([0.02, 0.02, 0.02], [0.0, 0.0, 1.0, 1.0], [-0.05, 0.05])
  panda_sim.add_object([0.02, 0.02, 0.02], [0.0, 0.0, 1.0, 1.0], [0, 0.05])
  panda_sim.add_object([0.02, 0.02, 0.02], [0.0, 0.0, 1.0, 1.0], [0.05, 0.05])
"""
Generates a heuristic function for use in dhRRT
Current approach: The sum of the distances of the boxes to the center of the goal region

"""
def make_heuristic(goal : PackGoal1):
  n = goal.n_boxes
  x_g = goal.x_g
  y_g = goal.y_g

  x_center = (x_g[0] + x_g[1]) / 2
  y_center = (y_g[0] + y_g[1]) / 2
  goal_center = [x_center, y_center]
  def h(state):
    stateVec = state["stateVec"]
    
    box_dists = np.zeros(shape=(n, ))
    for i in range(n):
        start_idx = -3*(i+1)
        end_idx = start_idx+2
        pos = stateVec[start_idx:end_idx]
        box_loc = [pos[0], pos[1]]
        box_dists[i] = np.linalg.norm(np.subtract(box_loc, goal_center))
    


    return np.sum(box_dists)

  return h
    
def make_sum_heuristic(goal : PackGoal1):
  # = min(dist from edge, 0 )
  n = goal.n_boxes
  x_g = goal.x_g
  y_g = goal.y_g

  x_center = (x_g[0] + x_g[1]) / 2
  y_center = (y_g[0] + y_g[1]) / 2
  goal_center = [x_center, y_center]
  def h(state):
    stateVec = state["stateVec"]
    
    box_dists = np.zeros(shape=(n,))
    for i in range(n):
        start_idx = -3*(i+1)
        end_idx = start_idx+2
        pos = stateVec[start_idx:end_idx]
        box_dist_x = x_g[1] - pos[0]
        box_dist_y = y_g[1] - pos[1]
        box_dists[i] = math.sqrt((box_dist_x ** 2) + (box_dist_y ** 2))
    return abs(np.sum(box_dists))
  return h

def make_center_heuristic(goal : PackGoal2):
  n = goal.n_boxes

  def h(state):
    stateVec = state["stateVec"]
    
    box_pos_lst = np.zeros(shape=(n, 2))
    for i in range(n):
      start_idx = -3*(i+1)
      end_idx = start_idx+2
      pos = stateVec[start_idx:end_idx]
      box_pos_lst[i, 0] = pos[0]
      box_pos_lst[i, 1] = pos[1]
    box_center = np.average(box_pos_lst, axis=0)
    return abs(np.linalg.norm(box_center - goal.optim_center))
  return h

def setup_390env(panda_sim, n_boxes=3):
  # set up my research environment
  bin_color = [0.4, 0.4, 0.4, 1.0]

  # smaller than [-0.3, 0.3] to ensure nothing starts in a corner
  x_bound = [0, 0.25]
  y_bound = [0, 0.25]
  # vertical walls
  panda_sim.add_obstacle([0.3, 0.01, 0.1], bin_color, [0, 0.31], baseMass=0)
  panda_sim.add_obstacle([0.3, 0.01, 0.1], bin_color, [0, -0.31], baseMass=0)

  # horizontal walls
  panda_sim.add_obstacle([0.01, 0.3, 0.1], bin_color, [0.31, 0], baseMass=0)
  panda_sim.add_obstacle([0.01, 0.3, 0.1], bin_color, [-0.31, 0], baseMass=0)

  # make some boxes and place them randomly!
  box_pos_lst = pos_n_boxes(n_boxes, x_bound, y_bound)

  # hard code test for goal
  # box_pos_lst = [[0.15, 0.2], [0.21, 0.15], [0.23, 0.23]]

  # testing optimal packing 
  #box_pos_lst = find_packing(3, 0.3, -1, 0.3, -1, 0.02)
  # print("Box positions: ", box_pos_lst)
  for pos in box_pos_lst:
    panda_sim.add_object([0.02, 0.02, 0.02], [0.0, 0.0, 1.0, 1.0], pos)
  

  return

def go_to_place(panda_sim, x, y):
  curr_ee, _ = panda_sim.get_ee_pose()
  ee_x, ee_y = curr_ee[0]-0.4, curr_ee[1]-0.2
  vx = x - ee_x
  vy = y - ee_y
  ctrl = [vx, vy, 0, 0.2]
  _ = panda_sim.execute(ctrl)
def execute_plan(panda_sim, plan, sleep_time=0.005):
  for node in plan:
    panda_sim.restore_state(node.state)
    p_from, _ = panda_sim.get_ee_pose()
    ctrl = node.get_control()
    if ctrl is not None:
      _ = panda_sim.execute(ctrl, sleep_time=sleep_time)
      p_to, _ = panda_sim.get_ee_pose()
      draw_line(panda_sim, p_from, p_to, c=[1, 0, 0], w=20)

def extract_reference_waypoints(panda_sim, ctrl):
  wpts_ref = np.empty(shape=(0, 3))
  d = ctrl[3]
  n_steps = int(d / sim.SimTimeStep)
  pos, quat = panda_sim.get_ee_pose()
  euler = panda_sim.bullet_client.getEulerFromQuaternion(quat)
  yaw = euler[2]
  for i in range(n_steps):
    wpt = np.array([pos[0] + (i + 1) * sim.SimTimeStep * ctrl[0],
                    pos[1] + (i + 1) * sim.SimTimeStep * ctrl[1],
                    (yaw + (i + 1) * sim.SimTimeStep * ctrl[2]) % (2 * np.pi)])
    wpts_ref = np.vstack((wpts_ref, wpt.reshape(1, -1)))
  return wpts_ref


def draw_line(panda_sim, p_from, p_to, c, w):
    return panda_sim.bullet_client.addUserDebugLine(p_from, p_to, lineColorRGB=c, lineWidth=w)
