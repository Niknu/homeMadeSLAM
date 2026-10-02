"""Publish KITTI odometry data as ROS 2 topics."""

from glob import glob
import pathlib

import cv2
from geometry_msgs.msg import PoseStamped, TransformStamped
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float64
from tf2_ros import TransformBroadcaster


class KittiPublisher(Node):
    """Publish KITTI poses, images, calibration and timestamps."""

    def __init__(self):
        """Initialize node, load KITTI data and set up publishers."""
        super().__init__("kitti_publisher")

        self.declare_parameter("sequence", "00")
        self.declare_parameter("kitti_path", "/workspace/src/KITTI_data")
        self.declare_parameter("playback_rate", 10.0)
        self.declare_parameter("loop", False)
        self.declare_parameter("child_frame", "camera_link")

        self.sequence = str(self.get_parameter("sequence").value)
        self.kitti_path = pathlib.Path(self.get_parameter("kitti_path").value)
        self.playback_rate = float(self.get_parameter("playback_rate").value)
        self.loop = self.parse_bool(self.get_parameter("loop").value)
        self.child_frame = self.get_parameter("child_frame").value

        self.frame_id = "kitti_frame"

        self.current_frame = 0

        self.load_ground_truth_poses()
        self.load_timestamps()
        self.load_camera_calibration()
        self.get_image_files()

        self.pose_pub = self.create_publisher(
            PoseStamped, "/kitti/ground_truth/pose", 10
        )
        self.image_pub = self.create_publisher(
            Image, "/kitti/camera/left/image_raw", 10
        )
        self.camera_info_pub = self.create_publisher(
            CameraInfo, "/kitti/camera/left/camera_info", 10
        )
        self.timestamp_pub = self.create_publisher(Float64, "/kitti/timestamps", 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        timer_period = 1.0 / self.playback_rate
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info(
            f"Published {len(self.image_files)} frames for sequence {self.sequence}"
        )

    def parse_bool(self, value):
        """Return a value as a boolean regardless of its type."""
        if isinstance(value, bool):
            return value
        return str(value).lower() in ("true", "1", "yes", "on")

    def load_ground_truth_poses(self):
        """Load all ground truth poses for the target sequence."""
        poses_file = (
            self.kitti_path
            / "data_odometry_poses"
            / "dataset"
            / "poses"
            / f"{self.sequence}.txt"
        )
        self.poses = []
        with open(poses_file, "r") as file:
            for line in file:
                if line.strip():
                    row = list(map(float, line.split()))
                    self.poses.append(np.array(row).reshape(3, 4))

    def load_timestamps(self):
        """Load the timestamp for every frame."""
        times_file = (
            self.kitti_path
            / "data_odometry_color"
            / "dataset"
            / "sequences"
            / self.sequence
            / "times.txt"
        )
        self.timestamps = []
        with open(times_file, "r") as file:
            for line in file:
                if line.strip():
                    self.timestamps.append(float(line))

    def load_camera_calibration(self):
        """Load the P2 projection matrix for the left camera."""
        calib_file = (
            self.kitti_path
            / "data_odometry_color"
            / "dataset"
            / "sequences"
            / self.sequence
            / "calib.txt"
        )
        self.p2 = None
        with open(calib_file, "r") as file:
            for line in file:
                if line.startswith("P2:"):
                    row = list(map(float, line.split(":")[1].split()))
                    self.p2 = np.array(row).reshape(3, 4)
                    break

    def get_image_files(self):
        """List all left camera image files for the sequence."""
        image_folder = (
            self.kitti_path
            / "data_odometry_color"
            / "dataset"
            / "sequences"
            / self.sequence
            / "image_2"
        )
        self.image_files = sorted(glob(str(image_folder / "*.png")))

    def matrix_to_quaternion(self, R):
        """Convert a 3x3 rotation matrix to a quaternion."""
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        if trace > 0.0:
            s = np.sqrt(trace + 1.0) * 2.0
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        return x, y, z, w

    def load_image(self, index):
        """Load a KITTI image and wrap it in a ROS 2 Image message."""
        image_path = self.image_files[index]
        cv_image = cv2.imread(image_path)
        msg = Image()
        msg.height = cv_image.shape[0]
        msg.width = cv_image.shape[1]
        msg.encoding = "bgr8"
        msg.is_bigendian = False
        msg.step = cv_image.shape[1] * 3
        msg.data = cv_image.tobytes()
        return msg

    def build_camera_info(self):
        """Build a CameraInfo message from the P2 matrix."""
        msg = CameraInfo()
        msg.header.frame_id = self.frame_id
        msg.height = 376
        msg.width = 1241
        msg.distortion_model = "plumb_bob"

        fx = self.p2[0, 0]
        fy = self.p2[1, 1]
        cx = self.p2[0, 2]
        cy = self.p2[1, 2]

        msg.k = [
            fx,
            0.0,
            cx,
            0.0,
            fy,
            cy,
            0.0,
            0.0,
            1.0,
        ]
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        msg.p = self.p2.flatten().tolist()
        return msg

    def timer_callback(self):
        """Publish the next frame of KITTI data."""
        if self.current_frame >= len(self.image_files):
            if self.loop:
                self.current_frame = 0
            else:
                self.get_logger().info("Finished publishing all frames")
                self.timer.cancel()
                return

        index = self.current_frame
        stamp = self.get_clock().now().to_msg()
        pose_matrix = self.poses[index]

        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = self.frame_id
        pose_msg.pose.position.x = float(pose_matrix[0, 3])
        pose_msg.pose.position.z = float(pose_matrix[1, 3])
        pose_msg.pose.position.y = float(pose_matrix[2, 3])

        R = pose_matrix[:3, :3]
        x, y, z, w = self.matrix_to_quaternion(R)
        pose_msg.pose.orientation.x = x
        pose_msg.pose.orientation.y = y
        pose_msg.pose.orientation.z = z
        pose_msg.pose.orientation.w = w

        self.pose_pub.publish(pose_msg)

        image_msg = self.load_image(index)
        image_msg.header.stamp = stamp
        image_msg.header.frame_id = self.frame_id
        self.image_pub.publish(image_msg)

        cam_info = self.build_camera_info()
        cam_info.header.stamp = stamp
        self.camera_info_pub.publish(cam_info)

        timestamp_msg = Float64()
        timestamp_msg.data = float(self.timestamps[index])
        self.timestamp_pub.publish(timestamp_msg)

        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self.frame_id
        t.child_frame_id = self.child_frame
        t.transform.translation.x = float(pose_matrix[0, 3])
        t.transform.translation.z = float(pose_matrix[1, 3])
        t.transform.translation.y = float(pose_matrix[2, 3])
        t.transform.rotation.x = x
        t.transform.rotation.y = y
        t.transform.rotation.z = z
        t.transform.rotation.w = w
        self.tf_broadcaster.sendTransform(t)

        self.current_frame += 1


def main(args=None):
    """Run the KITTI publisher node."""
    rclpy.init(args=args)
    node = KittiPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
