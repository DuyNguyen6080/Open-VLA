import rtde_control
import rtde_receive
import numpy as np
from scipy.spatial.transform import Rotation as R

class ur_robot:
    
    def __init__(self, robot_ip, scale=1, frequency = 1):
        
        print("Connecting to RTDE Control at localhost...")
        self.rtde_c = rtde_control.RTDEControlInterface(robot_ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip, frequency)
        
        self.scale = scale
        print("Succeses")
    def __del__(self):
        self.rtde_c.disconnect()
        self.rtde_r.disconnect()
    def isProtectiveStopped(self):
        return self.rtde_r.isProtectiveStopped()
    def convert_openvla_to_ur_pose(self, action, current_pose):
        action = np.asarray(action, dtype=float).reshape(-1)
        
        if action.size < 3:
            raise ValueError(f"Expected at least 3 OpenVLA action values, got {action}")

        dx, dy, dz = action[0:3] * self.scale
        droll, dpitch, dyaw = action[3:6] * self.scale
        target_pose = list(current_pose)

        
        target_pose[0] = float(target_pose[0] + dx)
        target_pose[1] = float(target_pose[1] + dy)
        target_pose[2] = float(target_pose[2] + dz)

        # UR current orientation is rotation vector [rx, ry, rz]
        current_rotvec = np.asarray(current_pose[3:6], dtype=float)
        current_rot = R.from_rotvec(current_rotvec)

        # OpenVLA gives roll/pitch/yaw delta
        delta_rot = R.from_euler(
            "xyz",
            [droll, dpitch, dyaw],
            degrees=False
        )

        # METHOD 1:
        # Apply delta rotation in TCP/tool local frame
        target_rot = delta_rot * current_rot

        # Convert back to UR rotation vector
        target_rotvec = target_rot.as_rotvec()

        target_pose[3] = float(target_rotvec[0])
        target_pose[4] = float(target_rotvec[1])
        target_pose[5] = float(target_rotvec[2])

        return target_pose
    def getrtde_r(self):
        return self.rtde_r
    def moveJ(self, Jarray):
        self.rtde_c.moveJ(Jarray)
    def getJ():
        return self.rtde_c.getActualQ()
