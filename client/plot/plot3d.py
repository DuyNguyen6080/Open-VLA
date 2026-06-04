
from plotutils import parse_args
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from PIL import Image

args = parse_args()
inputfile = args.inputfile
outputfile = args.outputfile

# 1. Load the CSV file (Replace with your actual file path if needed)
csv_path = inputfile
df = pd.read_csv(csv_path)

# 2. Extract the specific robot TCP columns
x = df['actual_TCP_pose_0']
y = df['actual_TCP_pose_1']
z = df['actual_TCP_pose_2']

# 2. Extract the specific robot TCP columns
tx = df['target x']
ty = df['target y']
tz = df['target z']


# 3. Initialize the 3D plotting canvas
fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=15, azim=10)

# 4. Plot the data as a continuous 3D line
# 'color' changes the line color, 'linewidth' adjusts thickness
ax.plot(x, y, z, color='blue', linewidth=2, label='TCP Path')
ax.plot(tx, ty, tz, color='grey', linewidth=2, label='TCP target')

# 5. Label the spatial axes and add a title
ax.set_xlabel('TCP X Pose (0)')
ax.set_ylabel('TCP Y Pose (1)')
ax.set_zlabel('TCP Z Pose (2)')
ax.set_title(outputfile)
ax.legend()

# 3. Create a temporary folder to save individual frames
frame_dir = 'temp_frames'
os.makedirs(frame_dir, exist_ok=True)

print("Generating frames for the GIF... Please wait.")
frame_paths = []

# Rotate full 360 degrees, jumping 4 degrees per frame (90 frames total)
for angle in range(0, 360, 4):
    ax.view_init(elev=25, azim=angle)
    
    # Save current frame view
    frame_path = os.path.join(frame_dir, f"frame_{angle:03d}.png")
    plt.savefig(frame_path, dpi=100, bbox_inches='tight')
    frame_paths.append(frame_path)

plt.close() # Close plot window to free memory

# 4. Compile the saved frames into an animated GIF
print("Compiling frames into an animated GIF...")
frames = [Image.open(p) for p in frame_paths]

# Save the GIF to your Desktop
gif_output_path = outputfile
frames[0].save(
    gif_output_path,
    save_all=True,
    format='GIF',
    append_images=frames[1:],
    duration=50,  # Speed of rotation (milliseconds per frame)
    loop=0        # 0 means infinite loop
)

# 5. Clean up temporary frame files
for p in frame_paths:
    os.remove(p)
os.rmdir(frame_dir)

print(f"Success! Your animated GIF is saved at: {os.path.abspath(gif_output_path)}")

angle = 0
while plt.fignum_exists(fig.number):
    # Set the camera view (elevation, azimuth angle)
    ax.view_init(elev=20, azim=angle)
    
    # Redraw the canvas cleanly
    plt.draw()
    plt.pause(0.01)  # Controls the speed of the rotation
    
    # Increment the angle for the next frame
    angle = (angle + 1) % 360
