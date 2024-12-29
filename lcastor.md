## LCASTOR

### Install

Our row following package is ag_row, colcon build the workspace to build the package.
The package uses tensorflow the model (`unet.hdf5`) file is not in this repository but available separately.
Put the `unet.hdf5` file in the fodler `hackathon2025_ws/src/ag_row/ag_row/models`.

#### Requirments
- shapley
- tensorflow
- scipy
- matplotlib
- opencv-python
- nvidia-cuda-toolkit

### Lunching
Use trajectory [path recorded01.traj](https://github.com/jondave/hackathon2025_ws/blob/main/demos/hackathon/config/paths/recorded01.traj)

- Launch simulation - `ros2 launch hackathon_bringup simulator.launch.py mode:=simulation_gazebo_classic robot_namespace:=robot demo_config_directory:=~/code/hackathon2025_ws/demos/hackathon/config/robot`

- Launch localisaiton - `ros2 launch tirrex_demo robot_localisation.launch.py demo_config_directory:=~/code/hackathon2025_ws/demos/hackathon/config robot_config_directory:=~/code/hackathon2025_ws/demos/hackathon/config/robot mode:=simulation_gazebo_classic robot_namespace:-robot`

- Launch path following - `ros2 launch tirrex_demo robot_path_following.launch.py demo_config_directory:=~/code/hackathon2025_ws/demos/hackathon/config robot_config_directory:=~/code/hackathon2025_ws/demos/hackathon/config/robot mode:=simulation_gazebo_classic robot_namespace:= robot trajectory_filename:= recorded01.traj`

- Launch evaluation - `ros2 launch hackathon_bringup evaluation.launch.py demo _config_directory:=~/code/hackathon2025_ws/demos/hackathon/config robot_namespace:= robot`

- Launch tensorflow row following - `ros2 launch ag_row row_follow`

- Start automatic ploughing tool raising and lowering - `python3 code/hackathon2025_ws/src/ag_row/tool_up_down.py`

Note absolute paths may be needed in the file paths.

### Running
- The robot will start in path following mode.
- We have not yet integrated switching between path following and our row following yet.
- When the robot gets to the start of the row manually turn off the path following.
- AG Row should take over and drive the robot along the crop rows.
- When the robot gets to the end of the row manually switch on path following.

### TODO
- [x] Integrate AG Row with FIRA robot.
- [x] Automatic raising and lowering of ploughing  tool when inside the field using GPS.
- [] Lidar SLAM inside barn.
- [] Lidar collision avoidance.