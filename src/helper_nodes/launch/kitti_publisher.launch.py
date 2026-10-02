from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import Parameter


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "sequence",
                default_value="00",
                description="KITTI sequence number",
            ),
            DeclareLaunchArgument(
                "kitti_path",
                default_value="/workspace/src/KITTI_data",
                description="Path to KITTI dataset",
            ),
            DeclareLaunchArgument(
                "playback_rate",
                default_value="20.0",
                description="Publishing rate in Hz",
            ),
            DeclareLaunchArgument(
                "loop",
                default_value="false",
                description="Loop playback",
            ),
            DeclareLaunchArgument(
                "child_frame",
                default_value="camera_link",
                description="Child frame id for TF",
            ),
            Node(
                package="helper_nodes",
                executable="kitti2ros",
                name="kitti_publisher",
                output="screen",
                parameters=[
                    Parameter(
                        "sequence",
                        LaunchConfiguration("sequence"),
                        value_type=str,
                    ),
                    Parameter(
                        "kitti_path",
                        LaunchConfiguration("kitti_path"),
                        value_type=str,
                    ),
                    Parameter(
                        "playback_rate",
                        LaunchConfiguration("playback_rate"),
                        value_type=float,
                    ),
                    Parameter(
                        "loop",
                        LaunchConfiguration("loop"),
                        value_type=bool,
                    ),
                    Parameter(
                        "child_frame",
                        LaunchConfiguration("child_frame"),
                        value_type=str,
                    ),
                ],
            ),
        ]
    )
