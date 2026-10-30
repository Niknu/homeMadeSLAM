from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="frontend",
                executable="poseFromVideoFeed_node",
                name="pose_from_video_feed",
                output="screen",
                emulate_tty=True,
            ),
        ]
    )
