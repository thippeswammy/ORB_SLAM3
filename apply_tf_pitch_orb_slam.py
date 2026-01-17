import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R
import os

def load_kitti_poses(file_path):
    """
    Loads poses from a KITTI format file (12 floats per line).
    Returns a list of 4x4 numpy arrays.
    """
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            if len(values) != 12:
                continue
            # Reshape to 3x4
            pose_3x4 = np.array(values).reshape(3, 4)
            # Make 4x4
            pose_4x4 = np.eye(4)
            pose_4x4[:3, :] = pose_3x4
            poses.append(pose_4x4)
    return poses

def save_kitti_poses(poses, file_path):
    """
    Saves poses to a KITTI format file.
    """
    with open(file_path, 'w') as f:
        for pose in poses:
            # Flatten the top 3x4 submatrix
            flat_pose = pose[:3, :].flatten()
            line = " ".join([f"{x:.12e}" for x in flat_pose])
            f.write(line + "\n")

def main():
    parser = argparse.ArgumentParser(description="Apply Rigid Body Transform (TF) and Pitch to KITTI Trajectory.")
    parser.add_argument("input_file", help="Path to input trajectory file (KITTI format).")
    parser.add_argument("output_file", help="Path to output trajectory file.")
    
    # Transform arguments
    parser.add_argument("--pitch", type=float, default=14.2954, help="Pitch angle in degrees (default: 14.2954).")
    parser.add_argument("--roll", type=float, default=0.0, help="Roll angle in degrees.")
    parser.add_argument("--yaw", type=float, default=0.0, help="Yaw angle in degrees.")
    
    parser.add_argument("--x", type=float, default=0.0, help="Translation X (meters).")
    parser.add_argument("--y", type=float, default=0.0, help="Translation Y (meters).")
    parser.add_argument("--z", type=float, default=0.0, help="Translation Z (meters).")
    
    parser.add_argument("--inverse", action="store_true", help="Apply the INVERSE of the defined transform.")
    parser.add_argument("--pre-multiply", action="store_true", help="Pre-multiply transform (T_new = T_tf * T_old) instead of post-multiply.")

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        return

    # 1. Create Transformation Matrix (T_tf)
    # Rotation (Euler ZYX convention generally safe for small pitch/roll)
    # Assuming the input args define the transform from Frame A to Frame B.
    r = R.from_euler('xyz', [args.roll, args.pitch, args.yaw], degrees=True)
    rot_mat = r.as_matrix()
    
    T_tf = np.eye(4)
    T_tf[:3, :3] = rot_mat
    T_tf[:3, 3] = [args.x, args.y, args.z]

    if args.inverse:
        T_tf = np.linalg.inv(T_tf)

    print(f"Applying Transform:\n{T_tf}")
    print(f"Mode: {'Pre-multiply (Change Frame)' if args.pre_multiply else 'Post-multiply (Move Body)'}")

    # 2. Load Poses
    poses = load_kitti_poses(args.input_file)
    print(f"Loaded {len(poses)} poses.")

    # 3. Apply Transform
    new_poses = []
    for T_old in poses:
        if args.pre_multiply:
            T_new = T_tf @ T_old
        else:
            T_new = T_old @ T_tf
        new_poses.append(T_new)

    # 4. Save
    save_kitti_poses(new_poses, args.output_file)
    print(f"Saved transformed poses to '{args.output_file}'.")

if __name__ == "__main__":
    main()
