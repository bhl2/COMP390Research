import time
import argparse
import pybullet as p
import pybullet_data as pd
import pybullet_utils.bullet_client as bc
import numpy as np
import sim
from pdef import Bounds, ProblemDefinition
from goal import RelocateGoal, GraspGoal, PackGoal1, PackGoal2
import rrt
import utils
import opt


def setup_pdef(panda_sim):
  pdef = ProblemDefinition(panda_sim)
  dim_state = pdef.get_state_dimension()
  dim_ctrl = pdef.get_control_dimension()

  # define bounds for state and control space
  bounds_state = Bounds(dim_state)
  for j in range(sim.pandaNumDofs):
    bounds_state.set_bounds(j, sim.pandaJointRange[j, 0], sim.pandaJointRange[j, 1])
  for j in range(sim.pandaNumDofs, dim_state):
    if ((j - sim.pandaNumDofs) % 3 == 2):
      bounds_state.set_bounds(j, -np.pi, np.pi)
    else:
      bounds_state.set_bounds(j, -0.3, 0.3)
  pdef.set_state_bounds(bounds_state)

  bounds_ctrl = Bounds(dim_ctrl)
  bounds_ctrl.set_bounds(0, -0.2, 0.2)
  bounds_ctrl.set_bounds(1, -0.2, 0.2)
  bounds_ctrl.set_bounds(2, -1.0, 1.0)
  bounds_ctrl.set_bounds(3, 0.4, 0.6)
  pdef.set_control_bounds(bounds_ctrl)
  return pdef


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--task", type=int, choices=[1, 2, 3, 4, 5, 6, 7])
  args = parser.parse_args()

  # set up the simulation
  pgui = utils.setup_bullet_client(p.GUI)
  panda_sim = sim.PandaSim(pgui)

  # Task 1: Move the Robot with Jacobian-based Projection
  if args.task == 1:
    pdef = setup_pdef(panda_sim)

    ctrls = [[0.02, 0, 0.2, 10],
             [0, 0.02, 0.2, 10],
             [-0.02, 0, -0.2, 10],
             [0, -0.02, -0.2, 10]]
    errs = []
    for _ in range(10):
      for ctrl in ctrls:
        wpts_ref = utils.extract_reference_waypoints(panda_sim, ctrl)
        wpts, _ = panda_sim.execute(ctrl)
        err_pos = np.mean(np.linalg.norm(wpts[:, 0:2] - wpts_ref[:, 0:2], axis=1))
        err_orn = np.mean(np.abs(wpts[:, 2] - wpts_ref[:, 2]))
        print("The average Cartesian error for executing the last control:")
        print("Position: %f meters\t Orientation: %f rads" % (err_pos, err_orn))
        errs.append([err_pos, err_orn])
    errs = np.array(errs)
    print("\nThe average Cartesian error for the entire execution:")
    print("Position: %f meters\t Orientation: %f rads" % (errs[:, 0].mean(), errs[:, 1].mean()))
  
  # Testing Custom Controls 
  elif args.task == 6:
    pdef = setup_pdef(panda_sim)
    start_state = panda_sim.save_state()
    start_xy = np.array([-0.19959354, 0.09872576])

    # Test lifting to a point
    test_heights = [0.1, 0.2, 0.3]
    for test_height in test_heights:
      panda_sim.restore_state(start_state)
      ee_before, _ = panda_sim.get_ee_pose()
      before_xy = np.array([ee_before[0], ee_before[1]])
      print("Start x, y: ", before_xy)
      before_z = ee_before[2]
      panda_sim.execute_lift(test_height)
      ee_after, _ = panda_sim.get_ee_pose()
      after_xy = np.array([ee_after[0], ee_after[1]])
      after_z = ee_after[2]
      z_err = (after_z - before_z) - test_height
      flat_dist = np.linalg.norm(after_xy-before_xy)
      print("Flat Dist Moved: ", flat_dist)
      print("Height Moved: ", (after_z - before_z))
      print("Error: ", z_err)
      print("Percent Error: ", ((z_err/test_height) * 100))
      time.sleep(1)
    print("----------- Done Testing Vertical Lift --------")

    # Test lifting, then moving to a point in the air
    panda_sim.restore_state(start_state)
    panda_sim.execute_lift(0.1)
    ee_before, _ = panda_sim.get_ee_pose()
    print("Before: ", ee_before)
    end_xy = np.array([0.2, 0.2])
    v_xy = end_xy - start_xy
    ctrl = [v_xy[0], v_xy[1], 0, 1]
    panda_sim.execute(ctrl)
    ee_after, _ = panda_sim.get_ee_pose()
    panda_sim.execute_lift(-0.1)

    print("After: ", ee_after)
    time.sleep(1)
  # end if
    
  # Testing Custom Controls 
  elif args.task == 7:
    pdef = setup_pdef(panda_sim)
    start_state = panda_sim.save_state()
    start_xy = np.array([-0.19959354, 0.09872576])

    # Test lifting to a point
    height = 0.2
    start_j_vals, _, _ = panda_sim.get_joint_states()

    panda_sim.execute_lift(0.05)

    end_j_vals, _, _ = panda_sim.get_joint_states()
    print("Start joint vals: ", start_j_vals)

    print("End joint vals: ", end_j_vals)
    time.sleep(1)
  # end if
  elif args.task == 5:
    
    utils.setup_390env(panda_sim)
    pdef = setup_pdef(panda_sim)
    pdef.bounds_ctrl.set_bounds(3, 1, 3)
    panda_sim.execute_lift(-0.2)
    goal = PackGoal1()
    pdef.set_goal(goal)
    h = utils.make_sum_heuristic(goal)
    p_val = 0.01
    d_max = 12
    planner = rrt.dhRRT(pdef, h, p_val, d_max)
    time_st = time.time()
    solved, plan, end = planner.solve(1200.0)
    print("Running time of rrt.dhRRT.solve() for P1 : %f secs" % (time.time() - time_st))
    print("Done with part 1 Woo!")
    panda_sim.restore_state(end)
    pdef2 = setup_pdef(panda_sim)
    utils.go_to_place(panda_sim, 0.01, 0.01) 
    curr_state = panda_sim.save_state()
    pdef2.set_start_state(curr_state)
    print("About to start second step")
    # Finish second step
    time.sleep(1)
    goal = PackGoal2(3, 0.3, -1, 0.3, -1, 0.02)
    pdef2.set_goal(goal)
    pdef2.bounds_ctrl.set_bounds(3, 1, 2)
    h2 = utils.make_center_heuristic(goal)
    p_val = 0.00001
    d_max = 10
    planner = rrt.dhRRT(pdef2, h2, p_val, d_max)
    time_st = time.time()
    solved, plan = planner.solve(1200.0)
    # stateFull = panda_sim.save_state()
    # state = stateFull["stateVec"]
    # print("Should be last box:", state[-3:-1])
    # while ((time.time()-time_st) < 10):
    #   pass
  else:
    # configure the simulation and the problem
    utils.setup_env(panda_sim)
    pdef = setup_pdef(panda_sim)

    # Task 2: Kinodynamic RRT Planning for Relocating
    if args.task == 2:
      goal = RelocateGoal()
      pdef.set_goal(goal)

      planner = rrt.KinodynamicRRT(pdef)
      time_st = time.time()
      solved, plan = planner.solve(120.0)
      print("Running time of rrt.KinodynamicRRT.solve(): %f secs" % (time.time() - time_st))

      if solved:
        print("The Plan has been Found:")
        panda_sim.restore_state(pdef.get_start_state())
        for _ in range(2):
          panda_sim.step()
        panda_sim.restore_state(pdef.get_start_state())
        utils.execute_plan(panda_sim, plan)
        while True:
          pass

    # Task 3: Kinodynamic RRT Planning for Grasping
    elif args.task == 3:
      goal = GraspGoal()
      pdef.set_goal(goal)

      planner = rrt.KinodynamicRRT(pdef)
      time_st = time.time()
      solved, plan = planner.solve(120.0)
      print("Running time of rrt.KinodynamicRRT.solve(): %f secs" % (time.time() - time_st))

      if solved:
        print("The Plan has been Found:")
        panda_sim.restore_state(pdef.get_start_state())
        for _ in range(2):
          panda_sim.step()
        panda_sim.restore_state(pdef.get_start_state())
        utils.execute_plan(panda_sim, plan)
        panda_sim.grasp()
        while True:
          pass

    # Task 4: Trajectory Optimization
    elif args.task == 4:
      ########## TODO ##########
      pass

    
      ##########################
    
    # My Task! 
    elif args.task == 5:
      ########## TODO ##########
      pass

      ##########################
