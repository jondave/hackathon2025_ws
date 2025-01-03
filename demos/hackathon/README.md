This directory corresponds to the entry point of the simulation.
It contains a `compose.yaml` file that contains all the services to start:
* `simulator`: start the simulator and spawn the robot
* `localisation`: compute absolute localization using GNSS, IMU and odometry measurements
* `path_following`: control the robot to follow the specified trajectory file
* `evaluation`: start evaluation nodes and publish information for rviz
* `rviz`: start evaluation nodes and publish information for rviz


## Running

To start the demo, just execute the following command (from this directory):
```
docker compose up
```

This command will start all the docker services defined in the `compose.yaml` file.
The `bash` service is however not started automatically because it is only used to open a shell in
the docker.


### Keeping the simulator open

You can start the services individually by specifying their name after the `compose up` arguments.
It can be useful to keep the simulator open while restarting the robot programs.
You can start only the simulator by executing:
```
docker compose up -d simulator
```
The `-d` option starts it in the background.
After that, you can start everything else using the command
```
docker compose up
```
This command can be interrupted by typing _Ctrl+C_ and can be executed again while keeping the
simulator open.
When you no longer need the simulator, you can stop it with
```
docker compose stop simulator
```

### Opening a shell in the docker

If you want to open execute some ROS command, you can open a shell in the ROS2 environment using
```
docker compose run bash
```


### Re-starting already created containers

The command `up` allows creating the containers and starting them.
After the containers are created, you can use `start` command to run one or several services in
detached mode and `stop` command to stop them.
At the end, if you want to remove the containers, you can use the `down` command.


### Services with `profiles: [optional]` attribute

The services that specify a `profiles` attribute are not enabled by default.
Usually, this parameter is used to start certain services by specifying the profile name on the
command line with the `-p optional` option.
In our case, this is used to disable automatic startup of these services.
If you want to start them, you have to specify their name in the command line.
For example, to start the view of the robot, you can have to execute
```
docker compose up -d robot_view
```
This will also start the simulator (because of the `depends_on` attribute) if it is not already
started.


## Documentation

* [Scenario of the challenge](/doc/challenge.md)
* [Configuration of the simulation](/doc/challenge_configuration.md)
* [Configuration of the robot](/doc/robot_configuration.md)
* [Configuration of the devices](/doc/devices_configuration.md)
* [Description of the robot control node](/doc/robot_control.md)
* [Description of the provided data files](/doc/plots_surveying.md)
