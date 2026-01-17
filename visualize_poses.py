import numpy as np
import matplotlib.pyplot as plt
try:
    from mpl_toolkits.mplot3d import Axes3D
    HAS_3D = True
except ImportError:
    print("Warning: Could not import Axes3D. 3D trajectory will not be plotted.")
    HAS_3D = False

import argparse
import os

def load_poses(file_path):
    """
    Load KITTI format poses (12 floats per line, flattened 3x4 matrix).
    """
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            T = np.array(values).reshape(3, 4)
            # Add bottom row to make 4x4
            T = np.vstack((T, [0, 0, 0, 1]))
            poses.append(T)
    return np.array(poses)

def visualize_trajectory(poses_file, output_image=None):
    poses = load_poses(poses_file)
    
    # KITTI Optical Frame:
    # X: Right
    # Y: Down
    # Z: Forward
    print(poses[0])
    x_right = poses[:, 0, 3]
    y_down = poses[:, 1, 3]
    z_fwd = poses[:, 2, 3]
    
    # Derived "Height" (ROS Z-equivalent) is -Y (since Y is Down)
    height = -y_down
    
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: Top Down Map (X vs Z)
    # This shows the path on the ground.
    ax1 = fig.add_subplot(131)
    ax1.plot(x_right, z_fwd, label='Path', linewidth=0.5)
    ax1.scatter(x_right[0], z_fwd[0], c='green', marker='o', label='Start')
    ax1.scatter(x_right[-1], z_fwd[-1], c='red', marker='x', label='End')
    ax1.set_title('Top View (Map: X-Right vs Z-Forward)')
    ax1.set_xlabel('X: Right (m)')
    ax1.set_ylabel('Z: Forward (m)')
    ax1.grid(True)
    ax1.axis('equal')
    ax1.legend()
    
    # Plot 2: Elevation Profile (Height vs Forward)
    # This shows how height changes over distance.
    ax2 = fig.add_subplot(132)
    ax2.plot(z_fwd, height, label='Elevation', linewidth=0.5)
    ax2.scatter(z_fwd[0], height[0], c='green', marker='o', label='Start')
    ax2.scatter(z_fwd[-1], height[-1], c='red', marker='x', label='End')
    ax2.set_title('Elevation Profile (Height vs Dist)')
    ax2.set_xlabel('Z: Forward distance (m)')
    ax2.set_ylabel('Height (-Y) (m)')
    ax2.grid(True)
    # ax2.axis('equal') # Elevation might be small, so maybe don't force equal aspect here
    
    # Plot 3: 3D Trajectory
    if HAS_3D:
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.plot(x_right, z_fwd, height, label='Trajectory', linewidth=0.5)
        ax3.scatter(x_right[0], z_fwd[0], height[0], c='green', marker='o', label='Start')
        ax3.scatter(x_right[-1], z_fwd[-1], height[-1], c='red', marker='x', label='End')
        ax3.set_title('3D Trajectory')
        ax3.set_xlabel('X (Right)')
        ax3.set_ylabel('Z (Forward)')
        ax3.set_zlabel('Height (-Y)')
    else:
        # If no 3D, maybe put a text or leave empty
        ax3 = fig.add_subplot(133)
        ax3.text(0.5, 0.5, '3D Plot Unavailable\n(Missing mpl_toolkits.mplot3d)', 
                 horizontalalignment='center', verticalalignment='center')
        ax3.axis('off')
    
    plt.tight_layout()
    
    if output_image:
        print(f"Saving trajectory to {output_image}")
        plt.savefig(output_image)
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize KITTI Poses")
    parser.add_argument("poses_file", help="Path to poses.txt")
    parser.add_argument("--output", help="Output image file (optional)", default="trajectory.png")
    
    args = parser.parse_args()
    visualize_trajectory(args.poses_file, args.output)
