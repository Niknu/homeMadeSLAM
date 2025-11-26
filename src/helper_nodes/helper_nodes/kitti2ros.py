import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3

import pathlib
import numpy as np


class MinimalPublisher(Node):
    def __init__(self, kitti_path, seqence_no):
        super().__init__("kitti_publisher")

        self.seqence_no = seqence_no
        gt_file_seqence_no = self.seqence_no + ".txt"

        self.kitti_folder = pathlib.Path(kitti_path)
        self.kitti_folder_gt = (
            self.kitti_folder
            / "data_odometry_poses"
            / "dataset"
            / "poses"
            / gt_file_seqence_no
        )
        self.kitti_folder_images = (
            self.kitti_folder
            / "data_odometry_color"
            / "dataset"
            / "sequences"
            / self.seqence_no
            / "image_2"
        )

        self.gt_file_line_no = 0
        timer_period = 0.1  # seconds
        #        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_ground_truth_pos()

    def get_new_image(self):
        pass

    def get_new_timestamp(self):
        pass

    def get_ground_truth_pos(self):
        # data_odometry_poses/dataset/poses/<sequenceNO>.txt
        # format: 3x4 rotmatix r11 r12 r13 t1 r21 r22 r23 t2 r31 r32 r33 t3

        gt_text = self.read_line(self.kitti_folder_gt, self.gt_file_line_no)
        print(gt_text)
        row = list(map(float, gt_text.split()))
        T = np.array(row).reshape(3, 4)
        print(T)
        T_hom = np.vstack([T, [0, 0, 0, 1]])
        print(T_hom)

    def read_line(self, path, line_no) -> str:
        with open(path, "r") as file:
            for i, line in enumerate(file):
                if i == line_no:
                    return line.rstrip()

    # def timer_callback(self):


def main(args=None):
    rclpy.init(args=args)

    kitti_path = "/workspace/src/KITTI_data"
    minimal_publisher = MinimalPublisher(kitti_path, seqence_no="00")

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
