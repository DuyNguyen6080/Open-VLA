"""
deploy.py

Provide a lightweight server/client implementation for deploying OpenVLA models (through the HF AutoClass API) over a
REST API. This script implements *just* the server, with specific dependencies and instructions below.

Note that for the *client*, usage just requires numpy/json-numpy, and requests; example usage below!

Dependencies:
    => Server (runs OpenVLA model on GPU): `pip install uvicorn fastapi json-numpy`
    => Client: `pip install requests json-numpy`

Client (Standalone) Usage (assuming a server running on 0.0.0.0:9999):

Note that if your server is not accessible on the open web, you can use ngrok, or forward ports to your client via ssh:
    => `ssh -L 8000:localhost:8000 ssh USER@<SERVER_IP>`
"""


import requests

import numpy as np
import cv2
from ur_robot import ur_robot
import json_numpy
from utils import parse_args, convert_openvla_to_ur_pose
json_numpy.patch()

 
def main():
    args = parse_args()
    robot_ip = args.robotip
    server_ip = args.serverip
    server_port = args.serverport
    camera = args.camera
    outfile = args.outfile

    robot = ur_robot(robot_ip, 0.6)
    robot_r = robot.getrtde_r()
    cap = cv2.VideoCapture(camera,cv2.CAP_AVFOUNDATION)


    record_variables = ["timestamp", "actual_q", "actual_TCP_pose", "actual_current"]
    robot_r.startFileRecording(outfile, record_variables)

    print(f"Data logging initiated into {outfile}...")

    if not cap.isOpened():
        raise RuntimeError("Error: Could not open video capture device.")

    while True: 
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            continue
        frame = cv2.resize(frame,(256, 256) )
        
        cv2.imshow("Iphone",frame)
        cv2.waitKey(1)

        # action == [dx,dy,dz,𝚫𝐫𝐨𝐥𝐥,𝚫𝐩𝐢𝐭𝐜𝐡,𝚫𝐲𝐚𝐰]
        action = requests.post(
        f"{server_ip}/act",
        json={"image": frame, "instruction": "pick up the orange object in the middle of the table", "unnorm_key": "bridge_orig"}
    ).json()
        print("OpenVLA action:", action)

        current_pose = robot.rtde_r.getActualTCPPose()
        target_pose = robot.convert_openvla_to_ur_pose(action, current_pose)
        print("Current TCP pose:", current_pose)
        print("Target TCP pose:", target_pose)

        #input("Press Enter to move robot...")
        robot.rtde_c.moveL(
            target_pose
        )
        if robot.isProtectiveStopped():
            print("Protective stop detected")
            break
    cap.release()
    cv2.destroyAllWindows()
    rtde_r.stopFileRecording()

if __name__ == "__main__":
    main()
    
