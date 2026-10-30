// #include <chrono>
// #include <memory>
#include "sensor_msgs/msg/image.hpp"
#include <cv_bridge/cv_bridge.hpp> // cv_bridge converts between ROS 2 image messages and OpenCV image representations
#include <fstream>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <random>
#include <rclcpp/clock.hpp>
#include <rclcpp/rclcpp.hpp> // IWYU pragma: export
#include <sstream>
#include <std_msgs/msg/string.hpp>
#include <string>
#include <unordered_map>

class FrontendClass {
public:
  FrontendClass() { loadCameraInsintric(); }

  static std::unordered_map<std::string, cv::Mat>
  loadCalibKitti(const std::string &path) {
    std::unordered_map<std::string, cv::Mat> m;
    std::ifstream f(path);
    for (std::string line; std::getline(f, line);) {
      auto colon = line.find(':');
      if (colon == std::string::npos)
        continue;
      std::istringstream iss(line.substr(colon + 1));
      cv::Mat a(3, 4, CV_64F);
      for (auto it = a.begin<double>(); it != a.end<double>(); ++it)
        iss >> *it;
      m[line.substr(0, colon)] = a;
    }
    return m;
  }

  static cv::Mat rosImage2OpencvImage(void) {

    cv::Mat my_image =
        cv::imread("/workspace/src/KITTI_data/data_odometry_color/dataset/"
                   "sequences/00/image_2/000000.png",
                   cv::IMREAD_COLOR);

    return my_image;
  }

  void loadCameraInsintric(std::string path =
                               "/workspace/src/KITTI_data/data_odometry_calib/"
                               "dataset/sequences/00/calib.txt") {
    auto c = loadCalibKitti(path);

    const cv::Mat &P2 = c.at("P2");

    // Strip the 4th column -> 3x3 K
    K = P2.colRange(0, 3).clone();
  }

  std::pair<std::vector<cv::KeyPoint>, cv::Mat> getFeatures(cv::Mat img) {
    auto detector = cv::SIFT::create();

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptor;

    detector->detectAndCompute(img, cv::noArray(), keypoints, descriptor);

    return {keypoints, descriptor};
  }

  std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>>
  getBestFeatureMatches(std::vector<cv::KeyPoint> keypoint1,
                        cv::Mat descriptor1,
                        std::vector<cv::KeyPoint> keypoint2,
                        cv::Mat descriptor2) {

    cv::BFMatcher matcher(cv::NORM_L2); // cv::FlannBasedMatcher for big sets
    std::vector<std::vector<cv::DMatch>> knn;
    int numberOfNeigbors = 2;
    matcher.knnMatch(descriptor1, descriptor2, knn, numberOfNeigbors);

    const float ratio = 0.7f;
    std::vector<cv::DMatch> matches;
    for (const auto &m : knn) {
      if (m.size() == 2 && m[0].distance < ratio * m[1].distance) {
        matches.push_back(m[0]);
      }
    }

    std::size_t minimumValueForEssentialMatrixEquation = 8;
    if (matches.size() < minimumValueForEssentialMatrixEquation) {
      throw std::runtime_error("Not enough matches");
    }

    // p1[i] <-> p2[i] <-> matches[i]: same order, so masks line up
    std::vector<cv::Point2f> p1, p2;
    for (const auto &m : matches) {
      p1.push_back(keypoint1[m.queryIdx].pt);
      p2.push_back(keypoint2[m.trainIdx].pt);
    }

    return {p1, p2};
  }

  static void drawEpilines(cv::Mat &img, const std::vector<cv::Vec3f> &lines,
                           const std::vector<cv::Point2f> &pts) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(50, 255);

    for (size_t i = 0; i < lines.size(); ++i) {
      cv::Scalar color(dist(rng), dist(rng), dist(rng));
      const cv::Vec3f &l = lines[i]; // a*x + b*y + c = 0

      // Intersect the line with the left (x=0) and right (x=w) image borders.
      // Requires b != 0, i.e. the line is not vertical.
      if (std::abs(l[1]) < 1e-6f)
        continue;

      cv::Point p0(0, cvRound(-l[2] / l[1]));
      cv::Point p1(img.cols, cvRound(-(l[2] + l[0] * img.cols) / l[1]));

      cv::line(img, p0, p1, color, 1, cv::LINE_AA);
      cv::circle(img, pts[i], 5, color, -1, cv::LINE_AA);
    }
  }

  void drawEpilinesShowImage(cv::Mat img1, cv::Mat img2,
                             std::vector<cv::Point2f> in1,
                             std::vector<cv::Point2f> in2, cv::Mat F) {
    // Draw a subset so the image stays readable
    const size_t maxLines = 50;
    if (in1.size() > maxLines) {
      in1.resize(maxLines);
      in2.resize(maxLines);
    }

    // 3. Epilines
    // Points from image 1 -> lines in image 2
    std::vector<cv::Vec3f> lines2;
    cv::computeCorrespondEpilines(in1, 1, F, lines2);
    // Points from image 2 -> lines in image 1
    std::vector<cv::Vec3f> lines1;
    cv::computeCorrespondEpilines(in2, 2, F, lines1);

    // 4. Draw
    cv::Mat out1 = img1.clone(), out2 = img2.clone();
    drawEpilines(out1, lines1, in1); // lines in img1, points of img1
    drawEpilines(out2, lines2, in2); // lines in img2, points of img2

    cv::Mat combined;
    cv::vconcat(out1, out2, combined);
    cv::imshow("Epipolar lines (left | right)", combined);
    cv::waitKey(0);
  }

  cv::Mat poseEstimation(cv::Mat pastImage, cv::Mat currentImage) {

    // TODO: past image only need to be ran 1st time.
    auto [keypoint1, descriptor1] = getFeatures(pastImage);
    auto [keypoint2, descriptor2] = getFeatures(currentImage);

    auto [point1, point2] =
        getBestFeatureMatches(keypoint1, descriptor1, keypoint2, descriptor2);
    cv::Mat mask;
    cv::Mat E =
        cv::findEssentialMat(point1, point2, K, cv::RANSAC, 0.999, 1.0, mask);

    std::vector<cv::Point2f> in1, in2;
    for (int i = 0; i < mask.rows; ++i)
      if (mask.at<uchar>(i)) {
        in1.push_back(point1[i]);
        in2.push_back(point2[i]);
      }

    /*
    cv::Mat F = K.inv().t() * E * K.inv();
    drawEpilinesShowImage(pastImage, currentImage, in1, in2, F);
    */
    cv::Mat R, t;
    int nGood = cv::recoverPose(E, in1, in2, R, t);
    std::cout << "nGood = " << nGood << std::endl;
    std::cout << "R = " << R << std::endl;
    std::cout << "t = " << t << std::endl;

    // Transform R and t to homogeneous matrix
    cv::Mat hNew = cv::Mat::eye(4, 4, CV_64F);

    // Copy the 3x3 rotation matrix into the top-left block
    R.copyTo(hNew(cv::Rect(0, 0, 3, 3)));

    // Copy the 3x1 translation vector into the top-right column
    t.copyTo(hNew(cv::Rect(3, 0, 1, 3)));

    return hNew;
  };

private:
  cv::Mat K{};
  cv::Mat hPrev{};
};

class MinimalPublisher : public rclcpp::Node {
public:
  MinimalPublisher() : Node("minimal_publisher"), count_(0) {
    publisher_ = this->create_publisher<std_msgs::msg::String>(
        "/vslam/fronend/camera", 1);
    publisherImage_ = this->create_publisher<sensor_msgs::msg::Image>(
        "/vslam/frontend/image", 1);

    auto timer_callback = [this]() -> void {
      auto message = std_msgs::msg::String();
      message.data = "Hello, world! " + std::to_string(this->count_++);
      publisher_->publish(message);

      cv::Mat img = FrontendClass::rosImage2OpencvImage();

      // Write message to be sent. Member function toImageMsg() converts a
      // CvImage into a ROS image message
      auto msg_ = std::make_unique<sensor_msgs::msg::Image>();
      cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", img)
          .toImageMsg(*msg_);
      msg_->header.stamp = rclcpp::Clock().now();

      publisherImage_->publish(*msg_);
    };
    timer_ = this->create_wall_timer(std::chrono::milliseconds(1000),
                                     timer_callback);
  }

private:
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisherImage_;
  size_t count_;
};

int main(int argc, char *argv[]) {

  FrontendClass front;

  cv::Mat my_image =
      cv::imread("/workspace/src/KITTI_data/data_odometry_color/dataset/"
                 "sequences/00/image_2/000000.png",
                 cv::IMREAD_COLOR);

  cv::Mat my_image1 =
      cv::imread("/workspace/src/KITTI_data/data_odometry_color/dataset/"
                 "sequences/00/image_2/000002.png",
                 cv::IMREAD_COLOR);

  front.poseEstimation(my_image, my_image1);

  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinimalPublisher>());
  rclcpp::shutdown();
  return 0;
}
