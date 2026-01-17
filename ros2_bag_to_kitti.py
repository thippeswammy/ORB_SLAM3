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
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image

# --- CONFIGURATION ---
# NOTE: BAG_PATH must be the FOLDER containing the .db3 file
BAG_PATH = "/media/thippe/SDV/Dataset/BUGGY_DATA_ROS/Buggy3_sb_port_calib"
IMAGE_TOPIC = '/camera/color/image_raw'
CALIB_TOPIC = '/camera/color/camera_info'
IMU_TOPIC = '/imu/data'

OUTPUT_DIR = "/media/thippe/SDV/Dataset/buggy/sequences/Buggy3_sb_port_calib"
CALIB_FILE_PATH = os.path.join(OUTPUT_DIR, "calib.txt")
TIMESTAMP_FILE_PATH = os.path.join(OUTPUT_DIR, 'times.txt')
IMU_FILE_PATH = os.path.join(OUTPUT_DIR, 'imu_data.csv')


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


if __name__ == '__main__':
    # Initialize ROS 2 context
    rclpy.init(args=None)

    # Run the extraction for Calibration and Data
    extract_calib(BAG_PATH, CALIB_TOPIC, CALIB_FILE_PATH)
    extract_data(BAG_PATH, IMAGE_TOPIC, OUTPUT_DIR, TIMESTAMP_FILE_PATH)
    extract_imu(BAG_PATH, IMU_TOPIC, IMU_FILE_PATH)
    rclpy.shutdown()
