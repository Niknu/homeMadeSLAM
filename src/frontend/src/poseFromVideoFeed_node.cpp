#include "sensor_msgs/msg/image.hpp"
#include <cv_bridge/cv_bridge.hpp> // cv_bridge converts between ROS 2 image messages and OpenCV image representations
#include <deque>
#include <fstream>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <map>
#include <optional>
#include <random>
#include <rclcpp/rclcpp.hpp> // IWYU pragma: export
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <sstream>
#include <string>
#include <tf2/LinearMath/Matrix3x3.hpp>
#include <tf2/LinearMath/Quaternion.hpp>
#include <tf2_ros/transform_broadcaster.hpp>
#include <unordered_map>

// Result of one processed frame, everything expressed in the world frame
// (= optical frame of the first camera).
struct FrameResult {
  cv::Mat T_wc; // 4x4 pose of the current camera in the world
  std::vector<cv::Point3f> pointsWorld;
};

class FrontendClass {
public:
  FrontendClass() { loadCameraInsintric(); }

  void setK(const cv::Mat &newK) { K = newK.clone(); }
  void setTranslationScale(double s) { translationScale = s; }
  // Force the length of the next frame-to-frame translation (e.g. from ground
  // truth). Used once by the next processFrame call.
  void setNextStepLength(double length) { nextStepLength = length; }

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

    cv::Mat gray = img;
    if (img.channels() == 3)
      cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    detector->detectAndCompute(gray, cv::noArray(), keypoints, descriptor);

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

  std::tuple<cv::Mat, cv::Mat, std::vector<cv::Point2f>,
             std::vector<cv::Point2f>>
  poseEstimation(cv::Mat pastImage, cv::Mat currentImage) {

    // TODO: past image only need to be ran 1st time.
    auto [keypoint1, descriptor1] = getFeatures(pastImage);
    auto [keypoint2, descriptor2] = getFeatures(currentImage);

    auto [point1, point2] =
        getBestFeatureMatches(keypoint1, descriptor1, keypoint2, descriptor2);
    cv::Mat mask;
    cv::Mat E =
        cv::findEssentialMat(point1, point2, K, cv::RANSAC, 0.999, 1.0, mask);

    // find valid points
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
    // recoverPose uses K to normalise the points; passing only E would assume
    // focal=1, pp=(0,0).
    cv::recoverPose(E, in1, in2, K, R, t);

    /*
    // Transform R and t to homogeneous matrix
    cv::Mat hNew = cv::Mat::eye(4, 4, CV_64F);

    // Copy the 3x3 rotation matrix into the top-left block
    R.copyTo(hNew(cv::Rect(0, 0, 3, 3)));

    // Copy the 3x1 translation vector into the top-right column
    t.copyTo(hNew(cv::Rect(3, 0, 1, 3)));
    */

    return {R, t, in1, in2};
  };

  std::pair<cv::Mat, std::vector<cv::Point3f>>
  scaling(cv::Mat &t, std::vector<cv::Point3f> &euclideanPoints) {

    std::vector<double> depths;
    std::vector<cv::Point3f> filtered3dPoints;
    // Only keep the depths value that is positive(z)
    for (const auto &point : euclideanPoints) {
      if (point.z <= 0)
        continue;

      depths.push_back(point.z);
      filtered3dPoints.push_back(point);
    }

    if (depths.empty()) {
      throw std::runtime_error("No triangulated points in front of camera");
    }

    std::nth_element(depths.begin(), depths.begin() + depths.size() / 2,
                     depths.end());
    double medianDepth = depths[depths.size() / 2];

    double s = 1.0 / medianDepth;

    // Apply to points AND translation (both, or the map becomes inconsistent)
    // Points far beyond the median depth are mostly low-parallax noise.
    const float maxScaledDepth = 10.0f;
    std::vector<cv::Point3f> scaledPoints;
    scaledPoints.reserve(filtered3dPoints.size());
    for (auto &X : filtered3dPoints) {
      X *= s;
      if (X.z <= maxScaledDepth)
        scaledPoints.push_back(X);
    }
    cv::Mat t_scaled = t * s;

    return {t_scaled, scaledPoints};
  };

  std::pair<cv::Mat, std::vector<cv::Point3f>>
  triangulationAndScaling(cv::Mat projection0, cv::Mat projection1,
                          std::vector<cv::Point2f> point0,
                          std::vector<cv::Point2f> point1, cv::Mat t) {

    cv::Mat homogeneousPoints;

    cv::triangulatePoints(projection0, projection1, point0, point1,
                          homogeneousPoints);
    // Point3f output requires float input to convertPointsFromHomogeneous
    homogeneousPoints.convertTo(homogeneousPoints, CV_32F);

    // TODO: Remove points with too low parallax

    std::vector<cv::Point3f> euclideanPoints;

    cv::convertPointsFromHomogeneous(homogeneousPoints.t(), euclideanPoints);

    auto [tScaled, scaled3dPoints] = scaling(t, euclideanPoints);

    // NOTE: consider: Always filter by depth sign, reprojection error, and
    // parallax afterwards.

    return {tScaled, scaled3dPoints};
  };

  // Returns the current camera pose and the newly triangulated points, both in
  // the world frame. Throws if the frame could not be tracked; in that case the
  // previous image and pose are kept so the next frame is matched against it.
  FrameResult processFrame(cv::Mat imageCurrent) {
    if (!initialized) {
      imagePrev = imageCurrent;
      initialized = true;
      return {T_cw.inv(), {}};
    }

    // R, t: motion from the previous camera to the current one (x2 = R*x1 + t)
    auto [R, t, point1, point2] = poseEstimation(imagePrev, imageCurrent);

    // Triangulate in the previous camera frame: P0 = K[I|0], P1 = K[R|t]
    cv::Mat projectionPrev = K * cv::Mat::eye(3, 4, CV_64F);
    cv::Mat RtCurrent;
    cv::hconcat(R, t, RtCurrent);
    cv::Mat projectionCurrent = K * RtCurrent;

    auto [tScaled, scaled3dPoints] = triangulationAndScaling(
        projectionPrev, projectionCurrent, point1, point2, t);

    tScaled *= translationScale;
    double pointScale = translationScale;
    if (nextStepLength) {
      const double norm = cv::norm(tScaled);
      if (norm > 1e-9) {
        const double k = *nextStepLength / norm;
        tScaled *= k;
        pointScale *= k;
      }
      nextStepLength.reset();
    }

    // Chain the relative motion onto the world pose
    cv::Mat T_rel = cv::Mat::eye(4, 4, CV_64F);
    R.copyTo(T_rel(cv::Rect(0, 0, 3, 3)));
    tScaled.copyTo(T_rel(cv::Rect(3, 0, 1, 3)));

    cv::Mat T_wc_prev = T_cw.inv();
    T_cw = T_rel * T_cw;

    // Previous camera frame -> world frame
    std::vector<cv::Point3f> pointsWorld;
    pointsWorld.reserve(scaled3dPoints.size());
    const cv::Matx44d T(T_wc_prev);
    for (const auto &p : scaled3dPoints) {
      const double s = pointScale;
      cv::Vec4d Xw = T * cv::Vec4d(p.x * s, p.y * s, p.z * s, 1.0);
      pointsWorld.emplace_back(Xw[0], Xw[1], Xw[2]);
    }

    imagePrev = imageCurrent;

    return {T_cw.inv(), pointsWorld};
  };

  void localMapping(void) {};
  void keyframeSelection(void) {};

private:
  cv::Mat imagePrev{};
  cv::Mat K{};
  cv::Mat T_cw{cv::Mat::eye(4, 4, CV_64F)}; // world -> current camera
  double translationScale{1.0};
  std::optional<double> nextStepLength{};
  bool initialized{false};
};

class FrontendNode : public rclcpp::Node {
public:
  FrontendNode() : Node("slam_frontend") {
    const auto imageTopic = declare_parameter<std::string>(
        "image_topic", "/kitti/camera/left/image_raw");
    const auto cameraInfoTopic = declare_parameter<std::string>(
        "camera_info_topic", "/kitti/camera/left/camera_info");
    worldFrame_ = declare_parameter<std::string>("world_frame", "world");
    cameraFrame_ =
        declare_parameter<std::string>("camera_frame", "camera_optical_frame");
    front_.setTranslationScale(
        declare_parameter<double>("translation_scale", 1.0));
    alignWorldWithRos_ = declare_parameter<bool>("align_world_with_ros", true);
    const auto gtTopic = declare_parameter<std::string>(
        "ground_truth_topic", "/kitti/ground_truth/pose_world");
    useGtScale_ = declare_parameter<bool>("use_gt_scale", false);
    mapPointStride_ = std::max<int64_t>(
        1, declare_parameter<int64_t>("map_point_stride", 2));
    mapPublishEveryN_ = std::max<int64_t>(
        1, declare_parameter<int64_t>("map_publish_every_n_frames", 5));
    maxMapPoints_ = static_cast<std::size_t>(std::max<int64_t>(
        1, declare_parameter<int64_t>("max_map_points", 300000)));

    // Optical frame of the first camera (x right, y down, z forward) ->
    // ROS world (x forward, y left, z up).
    A_ = cv::Matx44d::eye();
    if (alignWorldWithRos_) {
      A_ = cv::Matx44d(0, 0, 1, 0,  //
                       -1, 0, 0, 0, //
                       0, -1, 0, 0, //
                       0, 0, 0, 1);
    }

    posePub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
        "/vslam/frontend/pose", 10);
    pathPub_ =
        create_publisher<nav_msgs::msg::Path>("/vslam/frontend/path", 10);
    mapPub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
        "/vslam/frontend/map", 1);
    tfBroadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    cameraInfoSub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
        cameraInfoTopic, 10,
        [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) {
          if (haveCameraInfo_)
            return;
          cv::Mat K(3, 3, CV_64F);
          for (int i = 0; i < 9; ++i)
            K.at<double>(i / 3, i % 3) = msg->k[i];
          front_.setK(K);
          haveCameraInfo_ = true;
          RCLCPP_INFO_STREAM(get_logger(), "Camera intrinsics from "
                                               "camera_info:\n"
                                               << K);
        });

    // Ground truth positions, keyed by stamp, so each processed image can look
    // up the true step length (only used with use_gt_scale).
    gtSub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        gtTopic, 50, [this](geometry_msgs::msg::PoseStamped::ConstSharedPtr m) {
          const auto &p = m->pose.position;
          gtPositions_[rclcpp::Time(m->header.stamp).nanoseconds()] =
              cv::Vec3d(p.x, p.y, p.z);
          while (gtPositions_.size() > 500)
            gtPositions_.erase(gtPositions_.begin());
        });

    // Depth 1: if SIFT is slower than the camera, drop old frames.
    imageSub_ = create_subscription<sensor_msgs::msg::Image>(
        imageTopic, rclcpp::QoS(1),
        [this](sensor_msgs::msg::Image::ConstSharedPtr msg) {
          onImage(msg);
        });

    RCLCPP_INFO(get_logger(), "Listening for images on %s",
                imageTopic.c_str());
  }

private:
  void onImage(const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
    cv::Mat img;
    try {
      img = cv_bridge::toCvShare(msg, "bgr8")->image.clone();
    } catch (const cv_bridge::Exception &e) {
      RCLCPP_WARN(get_logger(), "cv_bridge: %s", e.what());
      return;
    }

    const int64_t stampNs = rclcpp::Time(msg->header.stamp).nanoseconds();
    if (useGtScale_ && prevStampNs_) {
      auto cur = gtPositions_.find(stampNs);
      auto prev = gtPositions_.find(*prevStampNs_);
      if (cur != gtPositions_.end() && prev != gtPositions_.end()) {
        front_.setNextStepLength(cv::norm(cur->second - prev->second));
      } else {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                             "use_gt_scale: no ground truth for this stamp");
      }
    }

    FrameResult result;
    try {
      result = front_.processFrame(img);
      prevStampNs_ = stampNs;
    } catch (const std::exception &e) {
      RCLCPP_WARN(get_logger(), "Frame skipped: %s", e.what());
      return;
    }

    const auto &stamp = msg->header.stamp;
    publishPose(cv::Matx44d(result.T_wc), stamp);
    addToMap(result.pointsWorld);

    if (++frameCount_ % mapPublishEveryN_ == 0)
      publishMap(stamp);
  }

  void publishPose(const cv::Matx44d &T_wc, const builtin_interfaces::msg::Time &stamp) {
    const cv::Matx44d T = A_ * T_wc;

    tf2::Matrix3x3 rot(T(0, 0), T(0, 1), T(0, 2), //
                       T(1, 0), T(1, 1), T(1, 2), //
                       T(2, 0), T(2, 1), T(2, 2));
    tf2::Quaternion q;
    rot.getRotation(q);

    geometry_msgs::msg::PoseStamped pose;
    pose.header.stamp = stamp;
    pose.header.frame_id = worldFrame_;
    pose.pose.position.x = T(0, 3);
    pose.pose.position.y = T(1, 3);
    pose.pose.position.z = T(2, 3);
    pose.pose.orientation.x = q.x();
    pose.pose.orientation.y = q.y();
    pose.pose.orientation.z = q.z();
    pose.pose.orientation.w = q.w();
    posePub_->publish(pose);

    path_.header = pose.header;
    path_.poses.push_back(pose);
    pathPub_->publish(path_);

    geometry_msgs::msg::TransformStamped tf;
    tf.header = pose.header;
    tf.child_frame_id = cameraFrame_;
    tf.transform.translation.x = pose.pose.position.x;
    tf.transform.translation.y = pose.pose.position.y;
    tf.transform.translation.z = pose.pose.position.z;
    tf.transform.rotation = pose.pose.orientation;
    tfBroadcaster_->sendTransform(tf);
  }

  void addToMap(const std::vector<cv::Point3f> &pointsWorld) {
    for (std::size_t i = 0; i < pointsWorld.size();
         i += static_cast<std::size_t>(mapPointStride_)) {
      const auto &p = pointsWorld[i];
      const cv::Vec4d X = A_ * cv::Vec4d(p.x, p.y, p.z, 1.0);
      map_.emplace_back(X[0], X[1], X[2]);
    }
    while (map_.size() > maxMapPoints_)
      map_.pop_front();
  }

  void publishMap(const builtin_interfaces::msg::Time &stamp) {
    sensor_msgs::msg::PointCloud2 cloud;
    cloud.header.stamp = stamp;
    cloud.header.frame_id = worldFrame_;

    sensor_msgs::PointCloud2Modifier modifier(cloud);
    modifier.setPointCloud2FieldsByString(1, "xyz");
    modifier.resize(map_.size());

    sensor_msgs::PointCloud2Iterator<float> x(cloud, "x"), y(cloud, "y"),
        z(cloud, "z");
    for (const auto &p : map_) {
      *x = p.x;
      *y = p.y;
      *z = p.z;
      ++x, ++y, ++z;
    }
    mapPub_->publish(cloud);
  }

  FrontendClass front_;

  std::string worldFrame_;
  std::string cameraFrame_;
  bool alignWorldWithRos_{true};
  int64_t mapPointStride_{2};
  int64_t mapPublishEveryN_{5};
  std::size_t maxMapPoints_{300000};
  cv::Matx44d A_;

  bool haveCameraInfo_{false};
  bool useGtScale_{false};
  std::map<int64_t, cv::Vec3d> gtPositions_;
  std::optional<int64_t> prevStampNs_;
  int64_t frameCount_{0};
  nav_msgs::msg::Path path_;
  std::deque<cv::Point3f> map_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr imageSub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cameraInfoSub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr gtSub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr posePub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pathPub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr mapPub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tfBroadcaster_;
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FrontendNode>());
  rclcpp::shutdown();
  return 0;
}
