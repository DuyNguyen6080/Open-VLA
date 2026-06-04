import rtde_control
import rtde_receive
import numpy as np
import json
from homeutils import parse_args

args = parse_args()
mode = args.posmode

robot_ip = "192.168.0.205"

print("Connecting to RTDE Control at localhost...")
rtde_c = rtde_control.RTDEControlInterface(robot_ip)
print("Connecting to RTDE Receive at localhost...")
rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
# FIX: Invert the logic so "j" maps to "Joint" and "tcp" maps to "TCP"
if mode == "j":
    mode_file = "Joint"
elif mode == "tcp":
    mode_file = "TCP"
else:
    print(f"Error: mode '{mode}' not valid")
    exit(1)


with open(f"/Users/dylannguyen/School/Senior-Project/github-edu-UniversalRobot/pose/home_{mode_file}_pose.json", "r") as f:
    loaded_pose = json.load(f)
if mode == "j":
    rtde_c.moveJ(loaded_pose)
elif mode == "tcp":
    rtde_c.moveL(loaded_pose)
else:
    print("mode not valid")