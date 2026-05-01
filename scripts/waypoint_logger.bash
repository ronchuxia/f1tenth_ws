#!/bin/bash
ros2 run waypoint_logger waypoint_logger --ros-args -p mode:=sim -p file_name:=waypoints.csv

# parameters:
# mode: 
# - sim: use keyboard teleop to log waypoints in simulation
# - rviz: use rviz to log clicked points as waypoints
# - pf: use controller to log waypoints in real world

# file_name: the name of the csv file to save waypoints, default to timestamp.csv
