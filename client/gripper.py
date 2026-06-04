import time
import robotiq_gripper

ROBOT_IP = "192.168.0.205"   # replace with your UR robot IP
GRIPPER_PORT = 63352        # Robotiq URCap gripper port

gripper = robotiq_gripper.RobotiqGripper()

print("Connecting...")
gripper.connect(ROBOT_IP, GRIPPER_PORT)



print("Closing...")
gripper.move_and_wait_for_pos(120, 255, 255)  # position, speed, force

time.sleep(1)

print("Opening...")
gripper.move_and_wait_for_pos(0, 255, 255)

print("Current position:", gripper.get_current_position())
print("Is open:", gripper.is_open())
print("Is closed:", gripper.is_closed())