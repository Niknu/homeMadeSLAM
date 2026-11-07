import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import (
    AnyLaunchDescriptionSource,
    PythonLaunchDescriptionSource,
)
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import Parameter
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    frontend_share = get_package_share_directory('frontend')
    helper_share = get_package_share_directory('helper_nodes')

    kitti_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(helper_share, 'launch', 'kitti_publisher.launch.py')
        ),
        launch_arguments={
            'sequence': LaunchConfiguration('sequence'),
            'kitti_path': LaunchConfiguration('kitti_path'),
            'playback_rate': LaunchConfiguration('playback_rate'),
            'loop': LaunchConfiguration('loop'),
            'child_frame': LaunchConfiguration('child_frame'),
        }.items(),
    )

    frontend_node = Node(
        package='frontend',
        executable='poseFromVideoFeed_node',
        name='slam_frontend',
        output='screen',
        emulate_tty=True,
        parameters=[
            Parameter(
                'image_topic',
                LaunchConfiguration('image_topic'),
                value_type=str,
            ),
            Parameter(
                'camera_info_topic',
                LaunchConfiguration('camera_info_topic'),
                value_type=str,
            ),
            Parameter(
                'world_frame',
                LaunchConfiguration('world_frame'),
                value_type=str,
            ),
            Parameter(
                'camera_frame',
                LaunchConfiguration('camera_frame'),
                value_type=str,
            ),
            Parameter(
                'translation_scale',
                LaunchConfiguration('translation_scale'),
                value_type=float,
            ),
            Parameter(
                'align_world_with_ros',
                LaunchConfiguration('align_world_with_ros'),
                value_type=bool,
            ),
            Parameter(
                'map_point_stride',
                LaunchConfiguration('map_point_stride'),
                value_type=int,
            ),
            Parameter(
                'map_publish_every_n_frames',
                LaunchConfiguration('map_publish_every_n_frames'),
                value_type=int,
            ),
            Parameter(
                'max_map_points',
                LaunchConfiguration('max_map_points'),
                value_type=int,
            ),
            Parameter(
                'use_gt_scale',
                LaunchConfiguration('use_gt_scale'),
                value_type=bool,
            ),
        ],
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', os.path.join(frontend_share, 'rviz', 'slam.rviz')],
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    foxglove_bridge = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            # Resolved lazily, so foxglove:=false works without the package
            PathJoinSubstitution(
                [
                    FindPackageShare('foxglove_bridge'),
                    'launch',
                    'foxglove_bridge_launch.xml',
                ]
            )
        ),
        launch_arguments={'port': LaunchConfiguration('foxglove_port')}.items(),
        condition=IfCondition(LaunchConfiguration('foxglove')),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'sequence',
                default_value='00',
                description='KITTI sequence number',
            ),
            DeclareLaunchArgument(
                'kitti_path',
                default_value='/workspace/src/KITTI_data',
                description='Path to KITTI dataset',
            ),
            DeclareLaunchArgument(
                'playback_rate',
                default_value='10.0',
                description='KITTI playback rate in Hz',
            ),
            DeclareLaunchArgument(
                'loop',
                default_value='false',
                description='Loop the KITTI sequence',
            ),
            DeclareLaunchArgument(
                'child_frame',
                default_value='camera_link',
                description='TF child frame used by the KITTI publisher',
            ),
            DeclareLaunchArgument(
                'image_topic',
                default_value='/kitti/camera/left/image_raw',
                description='KITTI image topic',
            ),
            DeclareLaunchArgument(
                'camera_info_topic',
                default_value='/kitti/camera/left/camera_info',
                description='KITTI camera_info topic',
            ),
            DeclareLaunchArgument(
                'world_frame',
                default_value='world',
                description='World/map frame used by the frontend',
            ),
            DeclareLaunchArgument(
                'camera_frame',
                default_value='camera_optical_frame',
                description='Optical camera frame published by the frontend',
            ),
            DeclareLaunchArgument(
                'translation_scale',
                default_value='1.0',
                description=(
                    'Constant scale applied to every monocular translation. '
                    'Monocular SLAM is only defined up to scale.'
                ),
            ),
            DeclareLaunchArgument(
                'align_world_with_ros',
                default_value='true',
                description=(
                    'Align the optical axes of the first camera with a ROS-style '
                    'world frame (x forward, y left, z up)'
                ),
            ),
            DeclareLaunchArgument(
                'map_point_stride',
                default_value='2',
                description='Add every Nth triangulated point to the map',
            ),
            DeclareLaunchArgument(
                'map_publish_every_n_frames',
                default_value='5',
                description='Throttle PointCloud2 publication',
            ),
            DeclareLaunchArgument(
                'max_map_points',
                default_value='300000',
                description='Maximum number of points kept in the map',
            ),
            DeclareLaunchArgument(
                'use_gt_scale',
                default_value='false',
                description=(
                    'Take each step length from KITTI ground truth, so the VO '
                    'trajectory can be compared to /kitti/ground_truth/path'
                ),
            ),
            DeclareLaunchArgument(
                'rviz',
                default_value='false',
                description='Start RViz with the SLAM configuration',
            ),
            DeclareLaunchArgument(
                'foxglove',
                default_value='true',
                description='Start foxglove_bridge (connect Foxglove to ws://localhost:<port>)',
            ),
            DeclareLaunchArgument(
                'foxglove_port',
                default_value='8765',
                description='foxglove_bridge WebSocket port',
            ),
            kitti_launch,
            frontend_node,
            rviz_node,
            foxglove_bridge,
        ]
    )
