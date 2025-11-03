#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <iomanip>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

using namespace std;

// Load images and timestamps from KITTI sequence
void LoadImages(const string &strPathToSequence, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    ifstream fTimes(strPathToSequence + "/times.txt");
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes, s);
        if(!s.empty())
        {
            stringstream ss(s);
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    string strPrefixLeft = strPathToSequence + "/image_0/";
    const int nTimes = vTimestamps.size();
    vstrImageFilenames.resize(nTimes);

    for(int i = 0; i < nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageFilenames[i] = strPrefixLeft + ss.str() + ".png";
    }
}

int main(int argc, char **argv)
{
    if(argc != 2)
    {
        cerr << "Usage: ./orb_pose_estimation path_to_sequence" << endl;
        return 1;
    }

    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    LoadImages(string(argv[1]), vstrImageFilenames, vTimestamps);

    int nImages = vstrImageFilenames.size();
    if(nImages < 2)
    {
        cerr << "Need at least 2 images for pose estimation!" << endl;
        return 1;
    }

    // Create ORB detector
    cv::Ptr<cv::ORB> orb = cv::ORB::create();

    cv::Mat prevImg = cv::imread(vstrImageFilenames[0], cv::IMREAD_GRAYSCALE);
    if(prevImg.empty()) { cerr << "Failed to load first image." << endl; return 1; }

    vector<cv::KeyPoint> prevKp;
    cv::Mat prevDesc;
    orb->detectAndCompute(prevImg, cv::Mat(), prevKp, prevDesc);

    cv::Mat K = (cv::Mat_<double>(3,3) << 
        718.856, 0, 607.1928,
        0, 718.856, 185.2157,
        0, 0, 1); // Example KITTI calibration

    for(int i = 1; i < nImages; i++)
    {
        cv::Mat img = cv::imread(vstrImageFilenames[i], cv::IMREAD_GRAYSCALE);
        if(img.empty()) { cerr << "Failed to load image " << i << endl; break; }

        vector<cv::KeyPoint> kp;
        cv::Mat desc;
        orb->detectAndCompute(img, cv::Mat(), kp, desc);

        // Match descriptors
        cv::BFMatcher matcher(cv::NORM_HAMMING, true);
        vector<cv::DMatch> matches;
        matcher.match(prevDesc, desc, matches);

        if(matches.size() < 8) { cerr << "Not enough matches!" << endl; continue; }

        // Extract matched points
        vector<cv::Point2f> pts1, pts2;
        for(auto &m : matches)
        {
            pts1.push_back(prevKp[m.queryIdx].pt);
            pts2.push_back(kp[m.trainIdx].pt);
        }

        // Compute Essential matrix
        cv::Mat E = cv::findEssentialMat(pts2, pts1, K, cv::RANSAC);
        cv::Mat R, t;
        cv::recoverPose(E, pts2, pts1, K, R, t);

        cout << "Frame " << i-1 << " -> " << i << " pose:" << endl;
        cout << "R = " << endl << R << endl;
        cout << "t = " << endl << t << endl;

        prevImg = img;
        prevKp = kp;
        prevDesc = desc;
    }

    return 0;
}

