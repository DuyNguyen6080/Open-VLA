import rtde_control
import rtde_receive
import numpy as np
import json

robot_ip = "192.168.0.205"

print("Connecting to RTDE Control at localhost...")
rtde_c = rtde_control.RTDEControlInterface(robot_ip)
print("Connecting to RTDE Receive at localhost...")
rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)

current_pose = rtde_r.getActualQ()
with open("/Users/dylannguyen/School/Senior-Project/github-edu-UniversalRobot/pose/home_Joint_pose.json", "w") as f:
    loaded_pose = json.dump(current_pose, f)
