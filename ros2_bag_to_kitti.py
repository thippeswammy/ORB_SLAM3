import rclpy
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageFilter
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import argparse

# --- CONFIGURATION ---
IMAGE_TOPIC = '/realsense/color/image_raw'  # <--- REPLACE with your image topic
BAG_PATH = "./ws_slam/BUGGY_DATA/simple_data_1"  # <--- REPLACE with the path to your bag folder
OUTPUT_DIR = "./dataset/buggy/seq/00"


# ---------------------

def extract_data(bag_path, image_topic, output_dir):
    """Reads a ROS 2 bag, extracts images, and saves them with timestamps in KITTI format."""

    # 1. Setup paths
    image_folder = os.path.join(output_dir, 'image_0')
    timestamp_file = os.path.join(output_dir, 'timestamps.txt')

    os.makedirs(image_folder, exist_ok=True)
    print(f"Saving images to: {image_folder}")

    # 2. Initialize Bag Reader
    reader = SequentialReader()
    try:
        reader.open(
            storage_options={'uri': bag_path, 'storage_id': 'sqlite3'},
            converter_options={'input_serialization_format': 'cdr', 'output_serialization_format': 'cdr'}
        )
    except Exception as e:
        print(f"Error opening bag file: {e}")
        print("Ensure BAG_PATH points to the FOLDER containing the .db3 file.")
        return

    # Filter messages to only read the required image topic
    topic_filter = StorageFilter(topics=[image_topic])
    reader.set_filter(topic_filter)

    # 3. Process Messages
    bridge = CvBridge()
    image_count = 0

    with open(timestamp_file, 'w') as ts_f:
        while reader.has_next():
            topic, data, timestamp = reader.read_next()

            # Deserialize the message data
            msg = deserialize_message(data, Image)

            # Get timestamp in seconds (KITTI format)
            # ROS 2 timestamp is nanoseconds from epoch
            t_sec = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            ts_f.write(f"{t_sec:.6f}\n")

            # Convert to OpenCV image and save
            try:
                # Use 'bgr8' for color images
                cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
            except Exception as e:
                print(f"CV_Bridge Error: {e}")
                continue

            # KITTI format requires 6-digit padding (000000.png)
            filename = os.path.join(image_folder, f"{image_count:06d}.png")
            cv2.imwrite(filename, cv_image)

            image_count += 1
            if image_count % 100 == 0:
                print(f"Processed {image_count} images...", end='\r')

    print(f"\n--- Extraction Complete! ---")
    print(f"Total images extracted: {image_count}")
    print(f"Timestamps saved to: {timestamp_file}")
    print(f"Images saved to: {image_folder}")


if __name__ == '__main__':
    # Initialize ROS 2 context (needed for rclpy/serialization)
    rclpy.init(args=None)

    # 1. Check/Update the CONFIGURATION section at the top of the script!

    # 2. Run the extraction
    extract_data(BAG_PATH, IMAGE_TOPIC, OUTPUT_DIR)

    rclpy.shutdown()
