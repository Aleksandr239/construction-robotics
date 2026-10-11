# Construction robotics

Isaac Sim scene for laying a straight brick wall with a Franka Emika Panda.
The wall comes from Revit. The script reads the brick meshes in the open stage and places the ones the arm can reach.

## Run

1. Open the construction-site USD in Isaac Sim. Physics, the Panda, and the pallet are already in that file.
2. Download `brick-wall/brick_lay_any.py`.
3. On the prim `/World/BrickLay`, set the script path to your copy of that file.
4. In `brick_lay_any.py`, replace `ROOT` with your Isaac Sim folder, the one that contains `extsDeprecated/isaacsim.robot_motion.motion_generation`.
5. Press Play and allow Python scripting. Press Pause to stop.

The wall must be separate meshes of about 20 x 10 x 8 cm, one straight wall, pallet on the opposite side of the robot, same origin as the Revit model. Bricks past the arm's reach stay on the pallet.
