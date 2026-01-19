# ORB-SLAM3 Usage Instructions

This document provides consolidated instructions for building and running ORB-SLAM3, based on recent updates and `build_cmd.txt`.

## 1. Building ORB-SLAM3

### Prerequisites

Ensure you have the necessary dependencies installed (OpenCV, Pangolin, Eigen3, etc.).

### Build Steps

**Clean Build:**

```bash
cd ~/test_bot/src/ORB_SLAM3
rm -rf build
mkdir build && cd build
cmake .. -DOpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4 -DCMAKE_BUILD_TYPE=Release
make -j4
```

**Build Thirdparty Libraries:**
If this is your first time or you need to rebuild dependencies:

*DBoW2:*

```bash
cd ~/test_bot/src/ORB_SLAM3/Thirdparty/DBoW2
rm -rf build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

*g2o:*

```bash
cd ~/test_bot/src/ORB_SLAM3/Thirdparty/g2o
rm -rf build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

*Sophus:*

```bash
cd ~/test_bot/src/ORB_SLAM3/Thirdparty/Sophus
rm -rf build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

## 2. Running Examples

### Monocular Examples

**KITTI Dataset (Sequence 00):**

```bash
./Examples/Monocular/mono_kitti ./Vocabulary/ORBvoc.txt ./Examples/Monocular/KITTI00.yaml ./dataset/kitti/seq/00/
```

**Buggy Dataset:**

```bash
./Examples/Monocular/mono_buggy ./Vocabulary/ORBvoc.txt ./Examples/Monocular/buggy.yaml ./dataset/buggy/seq/00/
```

*Note: Adjust paths to your specific dataset location.*

### Stereo-Inertial Examples

**TUM-VI Dataset:**

```bash
./Examples/Stereo-Inertial/stereo_inertial_tum_vi \
    Vocabulary/ORBvoc.txt \
    Examples/Stereo-Inertial/TUM-VI.yaml \
    ./dataset/TUM/dataset-corridor4_512_16/dso/cam0/images \
    ./dataset/TUM/dataset-corridor4_512_16/dso/cam1/images \
    ./Examples/Stereo-Inertial/TUM_TimeStamps/dataset-corridor4_512.txt \
    ./Examples/Stereo-Inertial/TUM_IMU/dataset-corridor4_512.txt \
    ./trajectory_corridor4.txt
```

## 3. Helper Scripts


**Bag to KITTI GNSS Conversion:**
`ros2_bag_to_kitti_gnss.py` is available for converting ROS 2 bag files containing GNSS data to KITTI format.

```bash
python3 ros2_bag_to_kitti_gnss.py --bag <path_to_bag> --output <output_dir>
```

**Pose Visualization:**
`visualize_poses.py` can be used to visualize trajectory poses.

```bash
python3 visualize_poses.py --file <path_to_pose_file>
```
