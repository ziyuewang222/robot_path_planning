import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # 默认指向本仓库 ros2_ws 同级的 regional_legacy 目录，可用 REGIONAL_LEGACY_ROOT 覆盖。
    default_algorithm_root = os.environ.get(
        "REGIONAL_LEGACY_ROOT",
        os.path.abspath(
            os.path.join(
                get_package_share_directory("regional_explorer"),
                "..",
                "..",
                "..",
                "..",
                "..",
                "regional_legacy",
            )
        ),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("algorithm_root", default_value=default_algorithm_root),
            DeclareLaunchArgument("map_topic", default_value="map"),
            DeclareLaunchArgument("robot_frame", default_value="base_link"),
            DeclareLaunchArgument("replan_rate_hz", default_value="1.0"),
            DeclareLaunchArgument("bfs_weight", default_value="8.0"),
            DeclareLaunchArgument("turn_weight", default_value="0.35"),
            DeclareLaunchArgument("obstacle_weight", default_value="1.0"),
            DeclareLaunchArgument("obstacle_safe_distance", default_value="6.0"),
            Node(
                package="regional_explorer",
                executable="frontier_explorer_node",
                name="frontier_explorer",
                output="screen",
                parameters=[
                    {
                        "algorithm_root": LaunchConfiguration("algorithm_root"),
                        "map_topic": LaunchConfiguration("map_topic"),
                        "robot_frame": LaunchConfiguration("robot_frame"),
                        "replan_rate_hz": LaunchConfiguration("replan_rate_hz"),
                        "bfs_weight": LaunchConfiguration("bfs_weight"),
                        "turn_weight": LaunchConfiguration("turn_weight"),
                        "obstacle_weight": LaunchConfiguration("obstacle_weight"),
                        "obstacle_safe_distance": LaunchConfiguration("obstacle_safe_distance"),
                    }
                ],
            ),
        ]
    )
