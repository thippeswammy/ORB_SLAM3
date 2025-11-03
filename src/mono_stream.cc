#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <iomanip>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

#include "System.h"

using namespace std;

class ORBSLAM3Node
{
public:
    ORBSLAM3Node(const string &strVocFile, const string &strSettingsFile)
    {
        // Initialize ORB-SLAM3 in MONOCULAR mode
        mpSLAM = new ORB_SLAM3::System(strVocFile, strSettingsFile, ORB_SLAM3::System::MONOCULAR, true);
        imageScale = mpSLAM->GetImageScale();
    }

    void ImageCallback(const sensor_msgs::ImageConstPtr &msg)
    {
        cv::Mat im;
        try
        {
            im = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8)->image;
        }
        catch (cv_bridge::Exception &e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        if (im.empty())
        {
            ROS_WARN("Received empty image frame!");
            return;
        }

        if (imageScale != 1.f)
        {
            int width = im.cols * imageScale;
            int height = im.rows * imageScale;
            cv::resize(im, im, cv::Size(width, height));
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        double tframe = msg->header.stamp.toSec();
        mpSLAM->TrackMonocular(im, tframe, vector<ORB_SLAM3::IMU::Point>(), "ros_stream_frame");

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
        ROS_INFO("Frame processed in %.3f s", ttrack);
    }

    void Shutdown()
    {
        mpSLAM->Shutdown();
        mpSLAM->SaveTrajectoryKITTI("KeyFrameTrajectory.txt");
        ROS_INFO("Trajectory saved to KeyFrameTrajectory.txt");
    }

private:
    ORB_SLAM3::System *mpSLAM;
    float imageScale;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "orbslam3_mono_stream");

    if (argc != 3)
    {
        cerr << endl
             << "Usage: rosrun orbslam3_ros mono_stream path_to_vocabulary path_to_settings" << endl;
        return 1;
    }

    ORBSLAM3Node node(argv[1], argv[2]);

    ros::NodeHandle nh;
    ros::Subscriber sub = nh.subscribe("/camera/image_raw", 1, &ORBSLAM3Node::ImageCallback, &node);

    ROS_INFO("ORB-SLAM3 Node started. Subscribed to /camera/image_raw");
    ros::spin();

    node.Shutdown();
    return 0;
}
