import os
import csv
from sensor_msgs.msg import Imu

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageFilter
# NOTE: Using _storage for class definition stability in Humble
from rosbag2_py._storage import StorageOptions, ConverterOptions
# Explicitly import ROS 2 message types
from sensor_msgs.msg import CameraInfo, NavSatFix, NavSatStatus
from sensor_msgs.msg import Image
from geometry_msgs.msg import QuaternionStamped

import pymap3d as pm
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import bisect
import math

# --- CONFIGURATION ---
# NOTE: BAG_PATH must be the FOLDER containing the .db3 file
BAG_PATH = "/media/thippe/project two world's HD/day21/Buggy2_sb_main_gate_circle"
IMAGE_TOPIC = '/camera/color/image_raw'
CALIB_TOPIC = '/camera/color/camera_info'
IMU_TOPIC = '/imu/data'
GNSS_FIX_TOPIC = '/gnss/fix'
GNSS_HEADING_TOPIC = '/gnss/heading'

OUTPUT_DIR = "/media/thippe/SDV/Dataset/buggy/sequences/Buggy2_sb_main_gate_circle"
CALIB_FILE_PATH = os.path.join(OUTPUT_DIR, "calib.txt")
TIMESTAMP_FILE_PATH = os.path.join(OUTPUT_DIR, 'times.txt')
IMU_FILE_PATH = os.path.join(OUTPUT_DIR, 'imu_data.csv')
POSES_FILE_PATH = os.path.join(OUTPUT_DIR, 'poses.txt')

# ============ ANTENNA OFFSET (meters from base_link) ============
# Copied from RecordOrigin.py
ANTENNA_OFFSET_X = 0.0     # Forward
ANTENNA_OFFSET_Y = -0.775  # Right
ANTENNA_OFFSET_Z = 0.0     # Up


# ---------------------

def initialize_reader(bag_path):
    """Initializes and returns a configured SequentialReader."""
    reader = SequentialReader()
    storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    try:
        reader.open(storage_options, converter_options)
    except Exception as e:
        print(f"Error opening bag file: {e}")
        print("Ensure BAG_PATH points to the FOLDER containing the .db3 file.")
        return None
    return reader


def extract_data(bag_path, image_topic, output_dir, timestamp_file):
    """
    Reads a ROS 2 bag, extracts images, and saves them with RELATIVE timestamps.
    """
    reader = initialize_reader(bag_path)
    if reader is None:
        return

    # 1. Setup paths
    image_folder = os.path.join(output_dir, 'image_0')
    os.makedirs(image_folder, exist_ok=True)
    print(f"Saving images to: {image_folder}")

    # Filter messages to only read the image topic
    topic_filter = StorageFilter(topics=[image_topic])
    reader.set_filter(topic_filter)

    # 2. Process Messages
    bridge = CvBridge()
    image_count = 0
    first_timestamp_sec = None  # For calculating relative time

    with open(timestamp_file, 'w') as ts_f:
        while reader.has_next():
            topic, data, timestamp_ns = reader.read_next()

            # Deserialize the message data
            msg = deserialize_message(data, Image)

            # --- RELATIVE TIMESTAMP CALCULATION (KITTI style) ---
            t_abs_sec = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9

            if first_timestamp_sec is None:
                first_timestamp_sec = t_abs_sec
                t_rel_sec = 0.0
            else:
                t_rel_sec = t_abs_sec - first_timestamp_sec

            # Write relative timestamp in scientific notation (KITTI style)
            ts_f.write(f"{t_rel_sec:.10e}\n")
            # --- END RELATIVE TIMESTAMP CALCULATION ---

            # Convert to OpenCV image and save
            try:
                cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
            except Exception as e:
                print(f"CV_Bridge Error for frame {image_count}: {e}")
                continue

            # KITTI format requires 6-digit padding (000000.png)
            filename = os.path.join(image_folder, f"{image_count:06d}.png")
            cv2.imwrite(filename, cv_image)

            image_count += 1
            if image_count % 100 == 0:
                print(f"Processed {image_count} images...", end='\r')

    print(f"\n--- Image Extraction Complete! ---")
    print(f"Total images extracted: {image_count}")
    print(f"Timestamps saved to: {timestamp_file}")


def extract_calib(bag_path, calib_topic, output_file):
    """Reads the first CameraInfo message and writes K and P matrices to KITTI calib.txt format."""

    reader = initialize_reader(bag_path)
    if reader is None:
        return

    # Filter messages to only read the CameraInfo topic
    topic_filter = StorageFilter(topics=[calib_topic])
    reader.set_filter(topic_filter)

    # Check if we found the topic
    if not reader.has_next():
        print(f"Error: No messages found on topic '{calib_topic}'. Check the topic name.")
        return

    # 2. Read the first CameraInfo message
    _, data, _ = reader.read_next()
    msg = deserialize_message(data, CameraInfo)

    # 3. Extract Matrices
    # P matrix (Projection Matrix): 3x4 array in row-major order
    P = np.array(msg.p).reshape((3, 4))

    # 4. Format for KITTI calib.txt (Monocular Case)
    P_formatted = ' '.join([f"{x:.12e}" for x in P.flatten()])

    # 5. Write to File
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        # P0 is the projection matrix for the first (color) camera
        f.write(f"P0: {P_formatted}\n")

        # KITTI placeholder entries (P1, P2, P3 are typically stereo/other cameras)
        P_zero_placeholder = '0.000000000000e+00 ' * 12
        f.write(f"P1: {P_zero_placeholder.strip()}\n")
        f.write(f"P2: {P_zero_placeholder.strip()}\n")
        f.write(f"P3: {P_zero_placeholder.strip()}\n")

        # R0_rect: Rectification matrix. For monocular color camera, use identity.
        R0_rect_identity = "1.000000000000e+00 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00"
        f.write(f"R0_rect: {R0_rect_identity}\n")

        # Tr_velo_to_cam: Transformation from Lidar to Camera frame. (Placeholder)
        Tr_zero_placeholder = '0.000000000000e+00 ' * 12
        f.write(f"Tr_velo_to_cam: {Tr_zero_placeholder.strip()}\n")

    print(f"\n--- Calibration Extraction Complete! ---")
    print(f"Calibration saved to: {output_file}")
    print(f"P0 Matrix (Projection):\n{P}")


def extract_imu(bag_path, imu_topic, output_csv_file):
    """
    Reads a ROS 2 bag, extracts IMU data, and saves it to a CSV file.
    """
    print(f"\n--- Starting IMU Extraction ---")
    print(f"Reading from topic: {imu_topic}")

    reader = initialize_reader(bag_path)
    if reader is None:
        return

    # Filter messages to only read the IMU topic
    topic_filter = StorageFilter(topics=[imu_topic])
    reader.set_filter(topic_filter)

    if not reader.has_next():
        print(f"Error: No messages found on topic '{imu_topic}'.")
        print("Please check the IMU_TOPIC variable in the script.")
        return

    # 1. Setup CSV file
    os.makedirs(os.path.dirname(output_csv_file), exist_ok=True)

    # Define CSV header
    header = [
        'header_stamp_sec',
        'orientation_x', 'orientation_y', 'orientation_z', 'orientation_w',
        'angular_vel_x', 'angular_vel_y', 'angular_vel_z',
        'linear_accel_x', 'linear_accel_y', 'linear_accel_z'
    ]

    msg_count = 0
    with open(output_csv_file, 'w', newline='') as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow(header)

        # 2. Process Messages
        while reader.has_next():
            topic, data, timestamp_ns = reader.read_next()

            # Deserialize the message data
            try:
                msg = deserialize_message(data, Imu)
            except Exception as e:
                print(f"Error deserializing IMU message: {e}")
                continue

            # Extract absolute timestamp from message header
            t_header_sec = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9

            # Extract data
            o = msg.orientation
            w = msg.angular_velocity
            a = msg.linear_acceleration

            # Write data row
            row = [
                f"{t_header_sec:.10f}",
                o.x, o.y, o.z, o.w,
                w.x, w.y, w.z,
                a.x, a.y, a.z
            ]
            writer.writerow(row)

            msg_count += 1
            if msg_count % 500 == 0:
                print(f"Processed {msg_count} IMU messages...", end='\r')

    print(f"\n--- IMU Extraction Complete! ---")
    print(f"Total IMU messages extracted: {msg_count}")
    print(f"IMU data saved to: {output_csv_file}")


def extract_gnss_ground_truth(bag_path, fix_topic, heading_topic, output_file, timestamp_file, image_abs_timestamps=None):
    """
    Generates KITTI-style ground truth (poses.txt).
    - Interpolates GNSS data to image timestamps.
    - Applies Fixed Pitch (Mounting) and Offsets (Camera vs GNSS).
    - Converts to KITTI Optical Frame (Z-Forward).
    - Normalizes to start at Identity OR Gravity-Aligned (Flat).
    """
    print(f"\n--- Starting GNSS Ground Truth Extraction ---")
    
    # Configuration
    FIXED_PITCH_DEG = 14.2954
    FIXED_ROLL_DEG = 0.0
    # Camera is 5cm above VLP, GNSS is 4.5cm above VLP -> Camera is 0.5cm above GNSS
    # Assuming "On top of" implies X,Y alignment (0.0).
    OFFSET_CAM_FROM_GNSS = np.array([0.0, 0.0, 0.005]) 
    
    # ROS (X-Fwd, Y-Left, Z-Up) to KITTI (Z-Fwd, X-Right, Y-Down)
    # R_ros_to_kitti = [[0, -1, 0], [0, 0, -1], [1, 0, 0]]
    R_ros_to_kitti = np.array([
        [0, -1, 0],
        [0, 0, -1],
        [1, 0, 0]
    ])

    # 1. Load Image Timestamps
    if image_abs_timestamps is None:
        if not os.path.exists(timestamp_file):
            print(f"Error: Timestamp file not found: {timestamp_file}")
            print("Run with --images or --all first to generate timestamps, or provide them in memory.")
            return

        print("Reading image timestamps from bag for synchronization...")
        reader_img = initialize_reader(bag_path)
        if reader_img is None:
            return
            
        reader_img.set_filter(StorageFilter(topics=[IMAGE_TOPIC]))
        
        image_abs_timestamps = []
        while reader_img.has_next():
            _, data, _ = reader_img.read_next()
            msg = deserialize_message(data, Image)
            t_sec = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            image_abs_timestamps.append(t_sec)
        
        del reader_img
        print(f"Found {len(image_abs_timestamps)} images.")
    else:
        print(f"Using {len(image_abs_timestamps)} cached image timestamps.")

    # 2. Read GNSS and Heading Data
    reader = initialize_reader(bag_path)
    reader.set_filter(StorageFilter(topics=[fix_topic, heading_topic]))
    
    gnss_data = []      # (t, lat, lon, alt)
    heading_data = []   # (t, qx, qy, qz, qw)
    
    print("Reading GNSS/Heading data...")
    count = 0
    while reader.has_next():
        topic, data, _ = reader.read_next()
        
        if topic == fix_topic:
            msg = deserialize_message(data, NavSatFix)
            if msg.status.status >= NavSatStatus.STATUS_FIX:
                t = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
                gnss_data.append((t, msg.latitude, msg.longitude, msg.altitude))
        
        elif topic == heading_topic:
            msg = deserialize_message(data, QuaternionStamped)
            t = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            heading_data.append((t, msg.quaternion.x, msg.quaternion.y, msg.quaternion.z, msg.quaternion.w))
        
        count += 1
        if count % 1000 == 0:
            print(f"Processed {count} GNSS/Heading messages...", end='\r')
            
    print(f"\nRead {len(gnss_data)} GNSS fixes and {len(heading_data)} heading samples.")
    
    if not gnss_data or not heading_data:
        print("Error: Not enough GNSS or Heading data found.")
        return

    gnss_times = np.array([x[0] for x in gnss_data])
    gnss_vals = np.array([x[1:] for x in gnss_data])
    
    heading_times = np.array([x[0] for x in heading_data])
    heading_quats = np.array([x[1:] for x in heading_data]) # (x, y, z, w)

    # 3. Coordinate Conversion (LLA -> ENU)
    lat0, lon0, alt0 = gnss_data[0][1], gnss_data[0][2], gnss_data[0][3]
    print(f"Global Origin (First Fix): Lat={lat0:.8f}, Lon={lon0:.8f}, Alt={alt0:.2f}")
    
    enu_vals = []
    for lat, lon, alt in gnss_vals:
        e, n, u = pm.geodetic2enu(lat, lon, alt, lat0, lon0, alt0)
        enu_vals.append([e, n, u])
    enu_vals = np.array(enu_vals)

    # 4. Interpolation
    unique_heading_times, unique_indices = np.unique(heading_times, return_index=True)
    unique_heading_quats = heading_quats[unique_indices]
    
    # We need to extract YAW from the quaternions, then rebuild with FIXED_PITCH
    r_obj = R.from_quat(unique_heading_quats)
    euler_angles = r_obj.as_euler('zxy', degrees=True) # Yaw, Pitch, Roll
    yaws = euler_angles[:, 0]
    
    # Rebuild Rotations with Fixed Pitch
    # sequence='zxy' -> Rotate Z(Yaw), then X(Pitch? No), Y(Pitch? ROS Y=Left. Pitch is usually around Y).
    # Standard ROS: Body X-Fwd, Y-Left, Z-Up.
    # Yaw around Z. Pitch around Y. Roll around X.
    # Order: Yaw -> Pitch -> Roll (Intrinsic) gives Body orientation.
    # Euler sequence 'zyx' (Yaw, Pitch, Roll)?
    # Let's use 'zyx' (Z, Y, X).
    # Yaw=Angle around Z (Global). Pitch=Angle around Y (Local). Roll=Angle around X (Local).
    
    new_eulers = np.zeros_like(euler_angles)
    new_eulers[:, 0] = yaws              # Yaw from Bag
    new_eulers[:, 1] = FIXED_PITCH_DEG   # Fixed Pitch
    new_eulers[:, 2] = FIXED_ROLL_DEG    # Fixed Roll
    
    fixed_rotations = R.from_euler('zyx', new_eulers, degrees=True)
    
    # Slerp the FIXED rotations
    slerp = Slerp(unique_heading_times, fixed_rotations)
    
    raw_poses = []
    
    for t_img in image_abs_timestamps:
        # Poses are generated in ROS Body Frame (X-Fwd) first
        
        # --- Interpolate Position (GNSS Antenna) ---
        idx = bisect.bisect_left(gnss_times, t_img)
        if idx == 0:
            pos_gnss = enu_vals[0]
        elif idx >= len(gnss_times):
            pos_gnss = enu_vals[-1]
        else:
            t0, t1 = gnss_times[idx-1], gnss_times[idx]
            p0, p1 = enu_vals[idx-1], enu_vals[idx]
            ratio = (t_img - t0) / (t1 - t0) if (t1 - t0) > 1e-6 else 0
            pos_gnss = p0 + (p1 - p0) * ratio
            
        # --- Interpolate Rotation (Body Frame) ---
        t_clamp = max(min(t_img, unique_heading_times[-1]), unique_heading_times[0])
        rot_body_r = slerp([t_clamp])[0]
        rot_body = rot_body_r.as_matrix()
        
        # --- Apply Camera Offset (Body Frame -> World) ---
        # Cam_Pos = GNSS_Pos + R_body * Offset_Cam_from_GNSS
        offset_world = rot_body @ OFFSET_CAM_FROM_GNSS
        pos_cam = pos_gnss + offset_world
        
        # --- Convert Rotation to KITTI Optical (Z-Fwd) ---
        # We generally have P_world = R_body * P_body
        # And we know P_opt = M * P_body  => P_body = M.T * P_opt
        # So P_world = R_body * M.T * P_opt
        # Thus R_optical = R_body @ R_ros_to_kitti.T
        rot_optical = rot_body @ R_ros_to_kitti.T
        
        # Construct 4x4 Global Pose (in Optical Frame orientation)
        T = np.eye(4)
        T[:3, :3] = rot_optical
        T[:3, 3] = pos_cam
        
        raw_poses.append(T)

    # 5. Normalize Poses (Map to Local Frame)
    if not raw_poses:
        print("Error: No poses generated.")
        return

    # To ensure the trajectory appears "flat" (Gravity Aligned) for visualization and evaluation,
    # we normalize relative to a "Gravity-Aligned" version of the first frame.
    
    # We reconstruct T0_ref using T0's Translation + T0's Yaw only.
    # T0 is raw_poses[0] (Optical Frame).
    
    t0_img = image_abs_timestamps[0]
    
    # Get First Frame's Body Orientation (which we had inside loop, but didn't save)
    # Re-calculate it quickly using slerp
    t_clamp_0 = max(min(t0_img, unique_heading_times[-1]), unique_heading_times[0])
    rot_body_0 = slerp([t_clamp_0])[0]
    yaw_0 = rot_body_0.as_euler('zyx', degrees=True)[0]
    
    # Construct Flat Body Rotation (Yaw only)
    # Pitch=0, Roll=0
    rot_body_flat_0 = R.from_euler('zyx', [yaw_0, 0.0, 0.0], degrees=True).as_matrix()
    
    # Convert Flux Body Rotation to Optical Frame
    rot_opt_flat_0 = rot_body_flat_0 @ R_ros_to_kitti.T
    
    # Construct Reference Pose (T0_ref)
    T0_ref = np.eye(4)
    T0_ref[:3, :3] = rot_opt_flat_0
    T0_ref[:3, 3] = raw_poses[0][:3, 3] # Use exact position of first frame
    
    T0_ref_inv = np.linalg.inv(T0_ref)
    
    print("Normalizing poses to Gravity-Aligned First Frame (Flat Map)...")
    
    final_poses = []
    for T in raw_poses:
        T_local = T0_ref_inv @ T
        
        # Extract 3x4 for KITTI
        T_kitti = T_local[:3, :].flatten()
        final_poses.append(T_kitti)

    # 6. Write to File
    print(f"Writing {len(final_poses)} poses to {output_file}...")
    with open(output_file, 'w') as f:
        for pose in final_poses:
            line = ' '.join([f"{x:.12e}" for x in pose])
            f.write(line + "\n")
            
    print("--- Ground Truth Extraction Complete! ---")


import argparse

# ... existing code ...

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract data from ROS2 bag to KITTI format.")
    parser.add_argument('--images', action='store_true', help="Extract Images")
    parser.add_argument('--calib', action='store_true', help="Extract Calibration")
    parser.add_argument('--imu', action='store_true', help="Extract IMU")
    parser.add_argument('--gnss', action='store_true', help="Extract GNSS Ground Truth")
    parser.add_argument('--all', action='store_true', help="Extract All (default if no other flags set)")
    
    args = parser.parse_args()
    
    # If no specific flags are set, or if --all is set, run everything
    run_all = args.all or not (args.images or args.calib or args.imu or args.gnss)

    # Initialize ROS 2 context
    rclpy.init(args=None)
    
    cached_timestamps = None

    if run_all or args.calib:
        extract_calib(BAG_PATH, CALIB_TOPIC, CALIB_FILE_PATH)
        
    if run_all or args.images:
        cached_timestamps = extract_data(BAG_PATH, IMAGE_TOPIC, OUTPUT_DIR, TIMESTAMP_FILE_PATH)
        
    if run_all or args.imu:
        extract_imu(BAG_PATH, IMU_TOPIC, IMU_FILE_PATH)
    
    if run_all or args.gnss:
        # Run GNSS GT Extraction
        # Pass cached_timestamps to avoid re-reading images
        extract_gnss_ground_truth(BAG_PATH, GNSS_FIX_TOPIC, GNSS_HEADING_TOPIC, POSES_FILE_PATH, TIMESTAMP_FILE_PATH, 
                                  image_abs_timestamps=cached_timestamps)
    
    rclpy.shutdown()

