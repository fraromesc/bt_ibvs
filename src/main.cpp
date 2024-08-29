#include <Eigen/Dense>
#include <cv_bridge/cv_bridge.h>
#include <functional>
#include <geometry_msgs/PolygonStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Float64.h>
#include <std_srvs/Trigger.h>

cv::Point findCentroid(const std::vector<cv::Point> &points) {
  cv::Point center(0, 0);
  for(const auto &point : points) {
    center.x += point.x;
    center.y += point.y;
  }
  center.x /= points.size();
  center.y /= points.size();
  return center;
}

double calculateAngle(const cv::Point &point, const cv::Point &center) {
  return atan2(point.y - center.y, point.x - center.x);
}

void sortPointsClockwise(std::vector<cv::Point> &points) {
  auto topLeft = *std::min_element(points.begin(), points.end(), [](const cv::Point &a, const cv::Point &b) {
    return (a.y < b.y) || (a.y == b.y && a.x < b.x);
  });

  cv::Point center = findCentroid(points);

  std::sort(points.begin(), points.end(), [&center, &topLeft](const cv::Point &a, const cv::Point &b) {
    double angleA = calculateAngle(a, center);
    double angleB = calculateAngle(b, center);
    if(a == topLeft) angleA = -1;
    if(b == topLeft) angleB = -1;

    return angleA < angleB;
  });
}

class AutoLanding {
  enum class Mode : uint8_t { WAITING,
                              APPROACH,
                              FINE };

public:
  AutoLanding() {
    imageSub_ = nh_.subscribe("/webcam/image_raw", 1, &AutoLanding::imageCallback, this);
    altitudeSub_ = nh_.subscribe("/mavros/global_position/rel_alt", 1, &AutoLanding::altitudeCallback, this);
    cameraInfoSub_ = nh_.subscribe("/webcam/camera_info", 1, &AutoLanding::cameraInforCallback, this);
    imagePub_ = nh_.advertise<sensor_msgs::Image>("/landing/debug", 1);
    featuresPub_ = nh_.advertise<geometry_msgs::PolygonStamped>("/landing/features", 1);
    statusPub_ = nh_.advertise<std_msgs::Float32MultiArray>("/landing/status", 1);
    refVelPub_ = nh_.advertise<geometry_msgs::TwistStamped>("/mavros/setpoint_velocity/cmd_vel", 1);
    approachSrv_ = nh_.advertiseService("/landing/mode/approach", &AutoLanding::triggerApproachMode, this);
    fineSrv_ = nh_.advertiseService("/landing/mode/fine", &AutoLanding::triggerFineMode, this);
    status_.data.resize(3);
  }

  inline bool triggerApproachMode(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res) {
    return triggerMode(req, res, Mode::APPROACH);
  }

  inline bool triggerFineMode(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res) {
    return triggerMode(req, res, Mode::FINE);
  }

  bool triggerMode(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res, const Mode mode) {
    if(altitude_ < 0.15) {
      res.success = false;
      res.message = "Failed to change mode. Altitude value very low.";
      currentMode_ = Mode::WAITING;
    } else if(fx_ < 0 || fy_ < 0) {
      res.success = false;
      res.message = "Failed to change mode. Camera info not available.";
      currentMode_ = Mode::WAITING;
    } else {
      res.success = true;
      switch(mode) {
      case Mode::APPROACH:
        ROS_WARN("APPROACH mode enabled.");
        res.message = "APPROACH mode enabled";
        break;
      case Mode::FINE:
        ROS_WARN("FINE mode enabled.");
        res.message = "FINE mode enabled";
        break;
      default:
        break;
      }
      currentMode_ = mode;
    }
    return true;
  }

  void cameraInforCallback(const sensor_msgs::CameraInfoConstPtr &msg) {
    fx_ = msg->K[0];
    fy_ = msg->K[4];
    cx_ = msg->K[2];
    cy_ = msg->K[5];
  }

  void altitudeCallback(const std_msgs::Float64ConstPtr &msg) {
    altitude_ = msg->data;
  }

  void imageCallback(const sensor_msgs::ImageConstPtr &msg) {
    /***--- In ---***/
    cv::Mat img = (cv_bridge::toCvCopy(msg, msg->encoding)->image);

    /***--- Checks ---***/
    if(currentMode_ == Mode::WAITING) {
      std::array<cv::Mat, 2> masks = getMasks(img);
      status_.data[0] = static_cast<double>(cv::countNonZero(masks[0])) / (masks[0].rows * masks[0].cols);
      status_.data[1] = static_cast<double>(cv::countNonZero(masks[1])) / (masks[1].rows * masks[1].cols);
      status_.data[2] = 0;
      statusPub_.publish(status_);
      return;
    }
    if(altitude_ < 0.15) {
      ROS_WARN("Altitude lower than 0.15 meters. Returning to WAITING mode.");
      currentMode_ = Mode::WAITING;
      return;
    }
    if(fx_ < 0 || fy_ < 0 || cx_ < 0 || cy_ < 0) {
      ROS_WARN("Camera not configured. Returning to WAITING mode.");
      currentMode_ = Mode::WAITING;
      return;
    }

    /***--- Corner detection ---***/
    std::array<cv::Mat, 2> masks = getMasks(img);
    std::vector<cv::Point> corners = getSquareCorners(img, currentMode_ == Mode::APPROACH ? masks[0] : masks[1]);
    if(corners.size() != 4) {
      imagePub_.publish(cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::RGB8, img).toImageMsg());
      return;
    }
    sortPointsClockwise(corners);

    /***--- Compute reference ---***/
    if(ref_.empty()) {
      const double square_size = 0.3 * msg->height;
      ref_.emplace_back(msg->width / 2.0 - square_size, msg->height / 2.0 - square_size);
      ref_.emplace_back(msg->width / 2.0 + square_size, msg->height / 2.0 - square_size);
      ref_.emplace_back(msg->width / 2.0 + square_size, msg->height / 2.0 + square_size);
      ref_.emplace_back(msg->width / 2.0 - square_size, msg->height / 2.0 + square_size);
    }
    for(int i = 0; i < ref_.size(); i++) {
      cv::circle(img, ref_[i], 5, cv::Scalar(255, 255, 0), -1);
      cv::line(img, ref_[i], corners[i], cv::Scalar(255, 255, 0), 2);
    }

    /***--- IBVS ---***/
    Eigen::Matrix<double, 2 * 4, 6> J1;
    Eigen::Matrix<double, 2 * 4, 6> J2;
    Eigen::Matrix<double, 2 * 4, 1> e;
    J1 << computeJacobian(corners[0], altitude_, fx_, fy_, cx_, cy_),
        computeJacobian(corners[1], altitude_, fx_, fy_, cx_, cy_),
        computeJacobian(corners[2], altitude_, fx_, fy_, cx_, cy_),
        computeJacobian(corners[3], altitude_, fx_, fy_, cx_, cy_);
    J2 << computeJacobian(ref_[0], 0.5, fx_, fy_, cx_, cy_),
        computeJacobian(ref_[1], 0.5, fx_, fy_, cx_, cy_),
        computeJacobian(ref_[2], 0.5, fx_, fy_, cx_, cy_),
        computeJacobian(ref_[3], 0.5, fx_, fy_, cx_, cy_);
    e << ref_[0].x - corners[0].x,
        ref_[0].y - corners[0].y,
        ref_[1].x - corners[1].x,
        ref_[1].y - corners[1].y,
        ref_[2].x - corners[2].x,
        ref_[2].y - corners[2].y,
        ref_[3].x - corners[3].x,
        ref_[3].y - corners[3].y;
    Eigen::Matrix<double, 6, 1> nu = 0.2 * (J1.completeOrthogonalDecomposition().pseudoInverse() + J2.completeOrthogonalDecomposition().pseudoInverse()) * e;
    geometry_msgs::TwistStamped msgout;
    msgout.header.stamp = ros::Time::now();
    msgout.header.frame_id = "/cam";
    msgout.twist.linear.x = -nu[1];
    msgout.twist.linear.y = -nu[0];
    msgout.twist.linear.z = -nu[2];
    msgout.twist.angular.x = -nu[4];
    msgout.twist.angular.y = -nu[3];
    msgout.twist.angular.z = -nu[5];
    if(msgout.twist.linear.x > 0.2) {
      msgout.twist.linear.x = 0.2;
    }
    if(msgout.twist.linear.y > 0.2) {
      msgout.twist.linear.y = 0.2;
    }
    if(msgout.twist.linear.z > 0.75) {
      msgout.twist.linear.z = 0.75;
    }
    if(msgout.twist.linear.x < -0.2) {
      msgout.twist.linear.x = -0.2;
    }
    if(msgout.twist.linear.y < -0.2) {
      msgout.twist.linear.y = -0.2;
    }
    if(msgout.twist.linear.z < -0.75) {
      msgout.twist.linear.z = -0.75;
    }
    refVelPub_.publish(msgout);

    /***--- Status ---***/
    status_.data[0] = static_cast<double>(cv::countNonZero(masks[0])) / (masks[0].rows * masks[0].cols);
    status_.data[1] = static_cast<double>(cv::countNonZero(masks[1])) / (masks[1].rows * masks[1].cols);
    status_.data[2] = cv::norm(ref_[0] - corners[0]) + cv::norm(ref_[1] - corners[1]) + cv::norm(ref_[2] - corners[2]) + cv::norm(ref_[3] - corners[3]);
    statusPub_.publish(status_);

    /***--- Features ---***/
    geometry_msgs::PolygonStamped msg_features;
    msg_features.header.stamp = ros::Time::now();
    msg_features.header.frame_id = "/cam";
    for(int i = 0; i < 4; i++) {
      msg_features.polygon.points.emplace_back();
      msg_features.polygon.points.back().x = corners[i].x;
      msg_features.polygon.points.back().y = corners[i].y;
    }
    for(int i = 0; i < 4; i++) {
      msg_features.polygon.points.emplace_back();
      msg_features.polygon.points.back().x = ref_[i].x;
      msg_features.polygon.points.back().y = ref_[i].y;
    }
    featuresPub_.publish(msg_features);

    /***--- Out ---***/
    imagePub_.publish(cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::RGB8, img).toImageMsg());
  }

private:
  ros::NodeHandle nh_;
  ros::Subscriber imageSub_;
  ros::Subscriber altitudeSub_;
  ros::Subscriber cameraInfoSub_;
  ros::Publisher statusPub_;
  ros::Publisher imagePub_;
  ros::Publisher featuresPub_;
  ros::Publisher refVelPub_;
  ros::ServiceServer approachSrv_;
  ros::ServiceServer fineSrv_;
  Mode currentMode_ = Mode::WAITING;
  std_msgs::Float32MultiArray status_;
  std::vector<cv::Point> ref_;
  double altitude_;
  double fx_{}, fy_{}, cx_{}, cy_{};

  std::array<cv::Mat, 2> getMasks(cv::Mat &img) {
    std::array<cv::Mat, 2> mask;
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

    cv::inRange(hsv, cv::Scalar(45, 150, 150), cv::Scalar(75, 255, 255), mask[0]);

    cv::Mat mask1, mask2;
    cv::inRange(hsv, cv::Scalar(0, 80, 80), cv::Scalar(30, 255, 255), mask1);
    cv::inRange(hsv, cv::Scalar(110, 80, 80), cv::Scalar(180, 255, 255), mask2);
    cv::bitwise_or(mask1, mask2, mask[1]);

    return mask;
  }

  std::vector<cv::Point> getSquareCorners(cv::Mat &img, cv::Mat &mask) {
    cv::Mat overlay;
    cv::cvtColor(mask, overlay, cv::COLOR_GRAY2BGR);
    overlay.setTo(cv::Scalar(255, 255, 255), mask);
    const double alpha = 0.5;
    cv::addWeighted(overlay, alpha, img, 1 - alpha, 0, img);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Point> approx;
    for(const std::vector<cv::Point> &contour : contours) {
      cv::approxPolyDP(contour, approx, cv::arcLength(contour, true) * 0.02, true);
      if(approx.size() == 4) {
        for(const cv::Point &point : approx) {
          cv::circle(img, point, 10, cv::Scalar(127, 0, 255), -1);
        }
        for(size_t i = 0; i < 4; ++i) {
          cv::line(img, approx[i], approx[(i + 1) % 4], cv::Scalar(255, 128, 0), 2);
        }
        return approx;
      }
    }
    return {};
  }

  inline Eigen::Matrix<double, 2, 6> computeJacobian(const cv::Point &p, const double z, const double fx, const double fy, const double cx, const double cy) {
    const double u = p.x - cx;
    const double v = p.y - cy;
    return (Eigen::Matrix<double, 2, 6>() << -fx / z, 0, u / z, u * v / fy, -(fx * fx + u * u) / fx, (fx / fy) * v,
            0, -fy / z, v / z, (fy * fy + v * v) / fy, -u * v / fx, -(fy / fx) * u)
        .finished();
  }
};

int main(int argc, char **argv) {
  ros::init(argc, argv, "landing");
  AutoLanding al;
  ros::spin();
  return 0;
}
