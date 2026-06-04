import rtde_control
import rtde_receive
from client import OpenVLAClient
from utils import parse_args
import numpy as np
import os
import cv2
from scipy.spatial.transform import Rotation as R
ROBOT_IP = "127.0.0.1"
MAX_XYZ_DELTA = 0.02   # 2 cm max per step
MAX_RPY_DELTA = 0.05   # ~3 degrees max per step

X_BOUNDS = [-0.4, 0.4]
Y_BOUNDS = [-0.4, 0.4]
Z_BOUNDS = [0.05, 0.45]  # Keeps tool at least 5cm above table surface

# UR5 to UR3e Dataset Scale Factor
# Adjust this if the UR5 model movements feel too over-extended or too small
UR5_TO_UR3E_SCALE = 0.6 
OPENVLA_IMAGE_WIDTH = 224
OPENVLA_IMAGE_HEIGHT = 224

class ur_robot:
    
    def __init__(self, robot_ip):
        
        print("Connecting to RTDE Control at localhost...")
        self.rtde_c = rtde_control.RTDEControlInterface(robot_ip)
        print("Connecting to RTDE Receive at localhost...")
        self.rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
    def __del__(self):
        self.rtde_c.disconnect()
        self.rtde_r.disconnect()
    def convert_openvla_to_ur_pose(self, action, current_pose):
        action = np.asarray(action, dtype=float).reshape(-1)
        
        if action.size < 3:
            raise ValueError(f"Expected at least 3 OpenVLA action values, got {action}")

        dx, dy, dz = action[0:3] * UR5_TO_UR3E_SCALE
        droll, dpitch, dyaw = action[3:6] * UR5_TO_UR3E_SCALE
        target_pose = list(current_pose)

        
        target_pose[0] = float(target_pose[0] - dx)
        target_pose[1] = float(target_pose[1] - dy)
        target_pose[2] = float(target_pose[2] + dz)

        # UR current orientation is rotation vector [rx, ry, rz]
        current_rotvec = np.asarray(current_pose[3:6], dtype=float)
        current_rot = R.from_rotvec(current_rotvec)

        # OpenVLA gives roll/pitch/yaw delta
        delta_rot = R.from_euler(
            "xyz",
            [-dpitch, -droll, dyaw],
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

    def moveJ(self, Jarray):
        self.rtde_c.moveJ(Jarray)
    def getJ():
        return self.rtde_c.getActualQ()

class openvlaserver:
    def __init__(self, host, port):
        self.client = OpenVLAClient(
            host=host,
            port=port,
            image_width=OPENVLA_IMAGE_WIDTH,
            image_height=OPENVLA_IMAGE_HEIGHT,
        )
        self.cap = None
        
    def queryimage(self, nparray):
        print("Reqesting Action...")
        return self.client.request_action(nparray)
    def promptAnglePicture(self):
        print("Press 'd' to set the angle")
        
        if not self.cap.isOpened():
            print("Error: Could not open video capture device.")
            sys.exit(-1)
        
        while True:
            # cap.read() returns a boolean (ret) and the frame matrix (frame)
            ret, frame = self.cap.read()
            
            # If frame read failed (e.g., camera disconnected or video ended)
            if not ret:
                print("Error: Failed to grab frame.")
                break

            # Display the frame in a window named 'Robot Feed'
            cv2.namedWindow("Robot Feed", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Robot Feed", 800, 600)
            cv2.imshow("Robot Feed", frame)

            # Wait for 1 millisecond and check if the user pressed the 'q' key
            if cv2.waitKey(1) & 0xFF == ord('d'):
                cv2.destroyAllWindows()
                return
    
    def open_camera(self, camera_index):
        self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            raise RuntimeError("Error: Could not open video capture device.")

        # Optional warmup: cameras often need a few frames
        for _ in range(5):
            self.cap.read()
        
    def takepicture(self, camera_index):
        
        if self.cap is None:
            raise RuntimeError("Camera is not open. Call open_camera() first.")

        ret, frame = self.cap.read()

        if not ret:
            print("Error: Failed to grab frame.")
            return None

        return frame
    def closeCamera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()

    def __del__(self):
        self.closeCamera()
        if self.client is not None:
            self.client.close()
            self.client = None

def main():
    args = parse_args()
    robot = ur_robot(args.robotip) # contructor call init

    openvla = openvlaserver(host=args.serverip,port=args.serverport)
    openvla.open_camera(args.camera)
    openvla.promptAnglePicture()
    while True:
        frame_np = openvla.takepicture(args.camera)

        if frame_np is None:
            continue

        os.system("cls" if os.name == "nt" else "clear")
        
        action = openvla.queryimage(frame_np)  # blocking call
        
        print("OpenVLA action:", action)

        current_pose = robot.rtde_r.getActualTCPPose()
        target_pose = robot.convert_openvla_to_ur_pose(action, current_pose)
        print("Current TCP pose:", current_pose)
        print("Target TCP pose:", target_pose)

        #input("Press Enter to move robot...")
        robot.rtde_c.moveL(
            target_pose
        )
                
        

        print("After robot move", flush=True)

if __name__ == "__main__":
    main()
