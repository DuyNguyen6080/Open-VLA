import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Robot Vision and Control Script")
    
    # Adds the --camera argument. Defaults to 0 (default system webcam)
    parser.add_argument(
        "--camera", 
        type=int, 
        default=0, 
        required=True,
        help="Camera index integer (e.g., 0, 1) or RTSP/video file path string"
        
    )
    parser.add_argument(
        "--frequency", 
        type=int, 
        default=1, 
        required=False,
        help="Camera index integer (e.g., 0, 1) or RTSP/video file path string"
        
    )
    parser.add_argument(
        "--outfile", 
        type=str, 
        default="robot_data.csv", 
        required=True,
        help="Camera index integer (e.g., 0, 1) or RTSP/video file path string"
        
    )
    parser.add_argument(
        "--serverip", 
        type=str, 
        default="localhost", 
        help="host openvla ip address"
    )
    parser.add_argument(
        "--robotip", 
        type=str, 
        default="localhost", 
        help="host robot ip address"
    )
    parser.add_argument(
        "--serverport", 
        type=int, 
        required=True,
        help="host ip address"
    )
    
    args = parser.parse_args()
    
    # Try converting to integer (for webcams); keep as string if it's a file path/URL
    try:
        args.camera = int(args.camera)
    except ValueError:
        pass  # Leaves it as a string if it's a video file or stream URL
        
    return args

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