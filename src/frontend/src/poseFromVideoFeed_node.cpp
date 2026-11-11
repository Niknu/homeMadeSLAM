#include "sensor_msgs/msg/image.hpp"
#include <algorithm>
#include <cmath>
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
  bool standingStill{false}; // pose held, keyframe kept
  bool scaleReset{false};    // tracking lost, scale re-guessed from last step
  double stepLength{0.0};    // camera motion since the previous keyframe
  int trackedPoints{0};      // PnP inliers (0 if the essential matrix was used)
};

// A processed frame. points[i] is the world point seen by keypoints[i], if
// one has been triangulated; these points carry the map scale forward.
struct TrackedFrame {
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  std::vector<std::optional<cv::Point3d>> points;
  cv::Mat T_cw; // world -> camera
};

class FrontendClass {
public:
  FrontendClass() { loadCameraInsintric(); }

  void setK(const cv::Mat &newK) { K = newK.clone(); }
  // Length of the first step; it defines the unit of the whole map.
  void setTranslationScale(double s) { translationScale = s; }
  // Force the length of the next initialisation step (e.g. from ground
  // truth). Ignored while the frame is tracked against the map.
  void setNextStepLength(double length) { nextStepLength = length; }
  void setTrackingThresholds(int minInliers, double reprojPx,
                             double minParallax) {
    minPnpInliers = minInliers;
    pnpReprojPx = reprojPx;
    minParallaxDeg = minParallax;
  }
  void setStandstillThresholds(double flowPx, double staticFraction,
                               double minParallaxPx) {
    standstillFlowPx = flowPx;
    standstillStaticFraction = staticFraction;
    keyframeMinParallaxPx = minParallaxPx;
  }

  // With (almost) no baseline the essential matrix is degenerate: RANSAC fits
  // noise or a moving object and recoverPose returns an arbitrary rotation.
  // The car counts as standing still if a large share of the matches does not
  // move at all (static background, even while other cars drive by), or if
  // the median flow is too small to triangulate from.
  bool isStandingStill(const std::vector<cv::Point2f> &p1,
                       const std::vector<cv::Point2f> &p2) const {
    std::vector<double> flow;
    flow.reserve(p1.size());
    std::size_t staticMatches = 0;
    for (std::size_t i = 0; i < p1.size(); ++i) {
      flow.push_back(cv::norm(p2[i] - p1[i]));
      if (flow.back() < standstillFlowPx)
        ++staticMatches;
    }

    if (staticMatches >= standstillStaticFraction * flow.size())
      return true;

    std::nth_element(flow.begin(), flow.begin() + flow.size() / 2, flow.end());
    return flow[flow.size() / 2] < keyframeMinParallaxPx;
  }

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

  // queryIdx indexes descriptor1, trainIdx descriptor2
  std::vector<cv::DMatch> getBestFeatureMatches(const cv::Mat &descriptor1,
                                                const cv::Mat &descriptor2) {

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

    return matches;
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

  // Relative motion from the essential matrix (x2 = R*x1 + t), |t| = 1: the
  // scale is unknown, so this is only used to (re-)initialise the map.
  std::pair<cv::Mat, cv::Mat>
  poseEstimation(const std::vector<cv::Point2f> &point1,
                 const std::vector<cv::Point2f> &point2) {

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

    return {R, t};
  };

  static cv::Vec3d cameraCentre(const cv::Mat &T_cw) {
    const cv::Matx44d T(T_cw);
    const cv::Matx33d R = T.get_minor<3, 3>(0, 0);
    return -(R.t() * cv::Vec3d(T(0, 3), T(1, 3), T(2, 3)));
  }

  // Pose of the current frame from the world points seen by the previous one.
  // Because those points are already in the map's unit, the scale carries
  // over. Inlier points are handed on to cur.points.
  bool trackWithPnp(const std::vector<cv::DMatch> &matches,
                    TrackedFrame &cur) const {
    std::vector<cv::Point3d> object;
    std::vector<cv::Point2d> image;
    std::vector<const cv::DMatch *> used;
    for (const auto &m : matches) {
      if (const auto &X = prev.points[m.queryIdx]) {
        object.push_back(*X);
        image.emplace_back(cur.keypoints[m.trainIdx].pt);
        used.push_back(&m);
      }
    }
    if (static_cast<int>(object.size()) < minPnpInliers)
      return false;

    // Seed with the previous pose; the camera moves little between frames.
    cv::Mat rvec, tvec = prev.T_cw(cv::Rect(3, 0, 1, 3)).clone();
    cv::Rodrigues(prev.T_cw(cv::Rect(0, 0, 3, 3)), rvec);
    std::vector<int> inliers;
    if (!cv::solvePnPRansac(object, image, K, cv::noArray(), rvec, tvec, true,
                            100, pnpReprojPx, 0.999, inliers) ||
        static_cast<int>(inliers.size()) < minPnpInliers)
      return false;

    cv::Mat R;
    cv::Rodrigues(rvec, R);
    cur.T_cw = cv::Mat::eye(4, 4, CV_64F);
    R.copyTo(cur.T_cw(cv::Rect(0, 0, 3, 3)));
    tvec.copyTo(cur.T_cw(cv::Rect(3, 0, 1, 3)));

    for (int i : inliers)
      cur.points[used[i]->trainIdx] = prev.points[used[i]->queryIdx];
    return true;
  }

  // Triangulates the matches that have no world point yet, from the two known
  // poses, so the points come out directly in the world frame and unit.
  // Returns the subset with enough parallax for the map.
  std::vector<cv::Point3f>
  triangulateNew(const std::vector<cv::DMatch> &matches,
                 TrackedFrame &cur) const {
    std::vector<cv::Point2f> p0, p1;
    std::vector<int> curIdx;
    for (const auto &m : matches) {
      if (cur.points[m.trainIdx])
        continue;
      p0.push_back(prev.keypoints[m.queryIdx].pt);
      p1.push_back(cur.keypoints[m.trainIdx].pt);
      curIdx.push_back(m.trainIdx);
    }
    if (p0.empty())
      return {};

    cv::Mat X;
    cv::triangulatePoints(K * prev.T_cw.rowRange(0, 3),
                          K * cur.T_cw.rowRange(0, 3), p0, p1, X);
    X.convertTo(X, CV_64F);

    const cv::Matx44d T0(prev.T_cw), T1(cur.T_cw);
    const cv::Matx33d Kx(K);
    const cv::Vec3d C0 = cameraCentre(prev.T_cw), C1 = cameraCentre(cur.T_cw);
    const double maxCosParallax = std::cos(minParallaxDeg * CV_PI / 180.0);

    // In front of the camera and close to the measured pixel
    auto reprojects = [&](const cv::Matx44d &T, const cv::Vec3d &Xw,
                          const cv::Point2f &px) {
      const cv::Vec4d Xc = T * cv::Vec4d(Xw[0], Xw[1], Xw[2], 1.0);
      if (Xc[2] <= 0)
        return false;
      const cv::Vec3d uv = Kx * cv::Vec3d(Xc[0], Xc[1], Xc[2]);
      return std::hypot(uv[0] / uv[2] - px.x, uv[1] / uv[2] - px.y) <
             pnpReprojPx;
    };

    std::vector<cv::Point3f> newPoints;
    for (int i = 0; i < X.cols; ++i) {
      const double w = X.at<double>(3, i);
      if (std::abs(w) < 1e-12)
        continue;
      const cv::Vec3d Xw(X.at<double>(0, i) / w, X.at<double>(1, i) / w,
                         X.at<double>(2, i) / w);
      if (!reprojects(T0, Xw, p0[i]) || !reprojects(T1, Xw, p1[i]))
        continue;

      // Every point is used for tracking: filtering on the (noisy) parallax
      // keeps mostly points whose depth came out too small, and the scale
      // then shrinks frame by frame.
      cur.points[curIdx[i]] = cv::Point3d(Xw);

      // Only well-conditioned points go to the published map; with a small
      // angle between the two rays the depth is mostly noise.
      const cv::Vec3d r0 = Xw - C0, r1 = Xw - C1;
      if (r0.dot(r1) / (cv::norm(r0) * cv::norm(r1)) <= maxCosParallax)
        newPoints.emplace_back(Xw[0], Xw[1], Xw[2]);
    }
    return newPoints;
  }

  // Returns the current camera pose and the newly triangulated points, both in
  // the world frame. Throws if the frame could not be tracked; in that case the
  // previous frame is kept so the next frame is matched against it.
  FrameResult processFrame(const cv::Mat &imageCurrent) {
    TrackedFrame cur;
    std::tie(cur.keypoints, cur.descriptors) = getFeatures(imageCurrent);
    cur.points.resize(cur.keypoints.size());

    if (!initialized) {
      cur.T_cw = cv::Mat::eye(4, 4, CV_64F);
      prev = std::move(cur);
      initialized = true;
      return {prev.T_cw.inv(), {}};
    }

    const auto matches = getBestFeatureMatches(prev.descriptors, cur.descriptors);
    std::vector<cv::Point2f> p1, p2;
    for (const auto &m : matches) {
      p1.push_back(prev.keypoints[m.queryIdx].pt);
      p2.push_back(cur.keypoints[m.trainIdx].pt);
    }

    if (isStandingStill(p1, p2)) {
      // Hold the pose and keep prev as keyframe, so parallax can build up
      nextStepLength.reset();
      return {prev.T_cw.inv(), {}, true};
    }

    FrameResult result;
    if (trackWithPnp(matches, cur)) {
      result.trackedPoints = static_cast<int>(
          std::count_if(cur.points.begin(), cur.points.end(),
                        [](const auto &p) { return p.has_value(); }));
    } else {
      // (Re-)initialise from the essential matrix. The first step length sets
      // the map unit; after a tracking loss the last step length is reused so
      // the unit stays roughly continuous.
      auto [R, t] = poseEstimation(p1, p2);
      double step = scaleInitialized ? lastStepLength : translationScale;
      if (nextStepLength)
        step = *nextStepLength;
      result.scaleReset = scaleInitialized;
      scaleInitialized = true;

      cv::Mat T_rel = cv::Mat::eye(4, 4, CV_64F);
      R.copyTo(T_rel(cv::Rect(0, 0, 3, 3)));
      cv::Mat(t * step).copyTo(T_rel(cv::Rect(3, 0, 1, 3)));
      cur.T_cw = T_rel * prev.T_cw;
    }
    nextStepLength.reset();

    result.pointsWorld = triangulateNew(matches, cur);
    result.stepLength =
        cv::norm(cameraCentre(cur.T_cw) - cameraCentre(prev.T_cw));
    lastStepLength = result.stepLength;
    result.T_wc = cur.T_cw.inv();

    prev = std::move(cur);
    return result;
  };

  void localMapping(void) {};
  void keyframeSelection(void) {};

private:
  TrackedFrame prev{}; // last keyframe
  cv::Mat K{};
  double translationScale{1.0};
  std::optional<double> nextStepLength{};
  double lastStepLength{1.0};
  double standstillFlowPx{1.0};
  double standstillStaticFraction{0.3};
  double keyframeMinParallaxPx{5.0};
  int minPnpInliers{30};
  double pnpReprojPx{2.0};
  double minParallaxDeg{1.0};
  bool initialized{false};
  bool scaleInitialized{false};
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
    front_.setStandstillThresholds(
        declare_parameter<double>("standstill_flow_px", 1.0),
        declare_parameter<double>("standstill_static_fraction", 0.3),
        declare_parameter<double>("keyframe_min_parallax_px", 5.0));
    front_.setTrackingThresholds(
        static_cast<int>(declare_parameter<int64_t>("min_pnp_inliers", 30)),
        declare_parameter<double>("pnp_reprojection_px", 2.0),
        declare_parameter<double>("min_parallax_deg", 1.0));
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
      // While standing still the keyframe is kept, so the GT step length must
      // still be measured from the keyframe's stamp.
      if (!result.standingStill)
        prevStampNs_ = stampNs;
    } catch (const std::exception &e) {
      RCLCPP_WARN(get_logger(), "Frame skipped: %s", e.what());
      return;
    }

    if (result.scaleReset)
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "Tracking lost, re-initialised scale from last "
                           "step length");
    if (!result.standingStill)
      RCLCPP_DEBUG(get_logger(), "step %.3f, tracked points %d, new points %zu",
                   result.stepLength, result.trackedPoints,
                   result.pointsWorld.size());

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
